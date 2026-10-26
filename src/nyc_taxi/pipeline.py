"""Closed loop: train champion, replay months, monitor, retrain challenger, promote via alias."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from nyc_taxi.baseline import MedianBaseline
from nyc_taxi.config import CHALLENGER_ALIAS, CHAMPION_ALIAS, Settings
from nyc_taxi.data import load_file, load_month, utc_now_iso
from nyc_taxi.drift import drift_report
from nyc_taxi.features import TARGET, build_features
from nyc_taxi.model import (
    TRAIN_PARAMS,
    VALID_FRAC,
    evaluate,
    load_model_dir,
    predict,
    save_model_dir,
    time_split,
    train_model,
)
from nyc_taxi.registry import (
    download_alias,
    get_alias_version,
    log_and_register,
    set_alias,
)
from nyc_taxi.zones import load_centroids

API_BATCH_SIZE = 1000
RECENT_DAYS_FRAC = 0.75
DRIFT_SAMPLE_ROWS = 50_000
CI_HOLDOUT_FRAC = 0.2
RETRAIN_WAIT_SECONDS = 900
API_RELOAD_WAIT_SECONDS = 180
POLL_SECONDS = 5


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def load_alias(alias: str, settings: Settings) -> tuple[Any, dict[str, Any], str]:
    """Load (booster, meta, version) for a registry alias; falls back to models/champion locally."""
    version = get_alias_version(alias, settings.tracking_uri)
    if version is not None:
        dst = Path(settings.models_dir) / "registry" / alias
        root = download_alias(alias, dst, settings.tracking_uri)
        booster, meta = load_model_dir(root)
        return booster, meta, version
    local = Path(settings.models_dir) / "champion"
    if alias == CHAMPION_ALIAS and (local / "model.txt").exists():
        booster, meta = load_model_dir(local)
        return booster, meta, str(meta.get("model_version") or "local")
    raise FileNotFoundError(f"no model under alias '{alias}': run `python -m nyc_taxi train` first")


def predict_remote(api_url: str, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Predict through the serving API in batches; returns predictions and served version."""
    preds = np.full(len(df), np.nan)
    served_version = "unknown"
    url = f"{api_url.rstrip('/')}/predict"
    for start in range(0, len(df), API_BATCH_SIZE):
        chunk = df.iloc[start : start + API_BATCH_SIZE]
        trips = [
            {
                "request_id": rid,
                "pickup_datetime": pickup.isoformat(),
                "PULocationID": int(pu),
                "DOLocationID": int(do),
            }
            for rid, pickup, pu, do in zip(
                chunk["request_id"],
                chunk["tpep_pickup_datetime"],
                chunk["PULocationID"],
                chunk["DOLocationID"],
                strict=True,
            )
        ]
        resp = requests.post(url, json={"trips": trips}, timeout=120)
        resp.raise_for_status()
        body = resp.json()
        served_version = str(body["model_version"])
        by_id = {p["request_id"]: p["duration_min"] for p in body["predictions"]}
        preds[start : start + len(chunk)] = [by_id[rid] for rid in chunk["request_id"]]
    return preds, served_version


def train_champion(settings: Settings) -> dict[str, Any]:
    """Train on the training month, register it and point the champion alias at it."""
    df = load_month(settings.train_month, settings.data_dir, settings.sample_rows, settings.seed)
    booster = train_model(df, settings.seed)
    head, tail = time_split(df, VALID_FRAC)
    val_metrics = evaluate(tail[TARGET].to_numpy(), predict(booster, tail))
    baseline = MedianBaseline().fit(head)
    base_mae = evaluate(tail[TARGET].to_numpy(), baseline.predict(tail))["mae"]

    meta: dict[str, Any] = {
        "model_version": None,
        "trained_on": [settings.train_month],
        "metrics": val_metrics,
        "baseline_mae": base_mae,
        "created_at": utc_now_iso(),
    }
    version = log_and_register(booster, meta, val_metrics, TRAIN_PARAMS, settings.tracking_uri)
    set_alias(version, CHAMPION_ALIAS, settings.tracking_uri)
    save_model_dir(
        booster, Path(settings.models_dir) / "champion", {**meta, "model_version": version}
    )
    return {"model_version": version, "mae": val_metrics["mae"], "baseline_mae": base_mae}


def retrain(month: str, settings: Settings, ref_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Train a challenger on training month + early days of `month`; promote if it wins holdout."""
    if ref_df is None:
        ref_df = load_month(settings.train_month, settings.data_dir, settings.sample_rows)
    new = load_month(month, settings.data_dir, settings.sample_rows, settings.seed)
    days = new["tpep_pickup_datetime"].dt.normalize()
    unique_days = np.sort(days.unique())
    k = max(1, int(len(unique_days) * RECENT_DAYS_FRAC))
    cut = unique_days[k - 1]
    recent = new[days <= cut]
    holdout = new[days > cut]
    if recent.empty or holdout.empty:
        raise ValueError(f"month {month} has too few days to split for retraining")

    challenger_train = pd.concat([ref_df, recent]).sort_values(
        "tpep_pickup_datetime", kind="stable"
    )
    challenger = train_model(challenger_train.reset_index(drop=True), settings.seed)

    champion, _, champion_version = load_alias(CHAMPION_ALIAS, settings)
    y = holdout[TARGET].to_numpy()
    champion_mae = evaluate(y, predict(champion, holdout))["mae"]
    challenger_metrics = evaluate(y, predict(challenger, holdout))
    promoted = challenger_metrics["mae"] < champion_mae * (1 - settings.promotion_margin)

    meta: dict[str, Any] = {
        "model_version": None,
        "trained_on": [settings.train_month, month],
        "trigger_month": month,
        "metrics": challenger_metrics,
        "champion_holdout_mae": champion_mae,
        "created_at": utc_now_iso(),
    }
    version = log_and_register(
        challenger, meta, challenger_metrics, TRAIN_PARAMS, settings.tracking_uri
    )
    set_alias(version, CHAMPION_ALIAS if promoted else CHALLENGER_ALIAS, settings.tracking_uri)

    record = {
        "month": month,
        "timestamp": utc_now_iso(),
        "previous_champion": champion_version,
        "challenger_version": version,
        "champion_mae": champion_mae,
        "challenger_mae": challenger_metrics["mae"],
        "promoted": promoted,
        "new_champion": version if promoted else champion_version,
    }
    _append_jsonl(Path(settings.reports_dir) / "promotions.jsonl", record)
    return record


def wait_for_retrain(
    month: str, champion_version: str, settings: Settings
) -> dict[str, Any] | None:
    """Poll the registry until a retrain Job registers a new champion or challenger."""
    challenger_before = get_alias_version(CHALLENGER_ALIAS, settings.tracking_uri)
    deadline = time.monotonic() + RETRAIN_WAIT_SECONDS
    while time.monotonic() < deadline:
        champion_now = get_alias_version(CHAMPION_ALIAS, settings.tracking_uri)
        challenger_now = get_alias_version(CHALLENGER_ALIAS, settings.tracking_uri)
        if champion_now != champion_version or challenger_now != challenger_before:
            promoted = champion_now != champion_version
            new_version = champion_now if promoted else challenger_now
            _, meta, _ = load_alias(CHAMPION_ALIAS if promoted else CHALLENGER_ALIAS, settings)
            record = {
                "month": month,
                "timestamp": utc_now_iso(),
                "source": "retrain-job",
                "previous_champion": champion_version,
                "challenger_version": new_version,
                "champion_mae": meta.get("champion_holdout_mae"),
                "challenger_mae": meta.get("metrics", {}).get("mae"),
                "promoted": promoted,
                "new_champion": champion_now,
            }
            _append_jsonl(Path(settings.reports_dir) / "promotions.jsonl", record)
            return record
        time.sleep(POLL_SECONDS)
    return None


def wait_for_api_version(api_url: str, version: str) -> bool:
    """Wait until every API replica serves `version` (several consecutive matching probes)."""
    deadline = time.monotonic() + API_RELOAD_WAIT_SECONDS
    streak = 0
    while time.monotonic() < deadline:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
        streak = streak + 1 if resp.ok and str(resp.json().get("model_version")) == version else 0
        if streak >= 6:
            return True
        time.sleep(2)
    return False


def _post_evaluation(monitor_url: str, payload: dict[str, Any]) -> None:
    resp = requests.post(f"{monitor_url.rstrip('/')}/evaluations", json=payload, timeout=30)
    resp.raise_for_status()


def _drift_for_month(
    champion: Any,
    ref_df: pd.DataFrame,
    df: pd.DataFrame,
    preds: np.ndarray,
    out_html: Path,
    seed: int,
) -> dict[str, Any]:
    """Drift of current features + predictions against the training month."""
    centroids = load_centroids()
    ref_feat = build_features(ref_df, centroids)
    ref_feat["prediction"] = predict(champion, ref_df)
    cur_feat = build_features(df, centroids)
    cur_feat["prediction"] = preds
    if len(cur_feat) > DRIFT_SAMPLE_ROWS:
        cur_feat = cur_feat.sample(n=DRIFT_SAMPLE_ROWS, random_state=seed)
    if len(ref_feat) > DRIFT_SAMPLE_ROWS:
        ref_feat = ref_feat.sample(n=DRIFT_SAMPLE_ROWS, random_state=seed)
    return drift_report(ref_feat, cur_feat, out_html)


def _write_outputs(reports: Path, history: list[dict[str, Any]]) -> None:
    (reports / "history.json").write_text(json.dumps(history, indent=2, default=str))
    _plot_mae(reports / "mae_by_month.png", history)
    _write_markdown(reports / "replay.md", history)


def _plot_mae(path: Path, history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = [row["month"] for row in history]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [row["mae"] for row in history], marker="o", label="champion MAE")
    ax.plot(x, [row["baseline_mae"] for row in history], marker="s", label="baseline MAE")
    for idx, row in enumerate(history):
        if row["promoted"]:
            ax.axvline(idx, color="tab:green", linestyle="--", alpha=0.7)
    ax.set_xticks(x, months)
    ax.set_ylabel("MAE (minutes)")
    ax.set_title("Replay MAE by month (dashed line = promotion)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_markdown(path: Path, history: list[dict[str, Any]]) -> None:
    lines = [
        "# Replay report",
        "",
        "| month | version | MAE | baseline MAE | ref MAE | drift share | degraded "
        "| promoted | drift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in history:
        lines.append(
            f"| {row['month']} | {row['model_version']} | {row['mae']:.3f} | "
            f"{row['baseline_mae']:.3f} | {row['ref_mae']:.3f} | {row['drift_share']:.2f} | "
            f"{'yes' if row['degraded'] else 'no'} | {'yes' if row['promoted'] else 'no'} | "
            f"[html](drift/{row['month']}.html) |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_replay(
    settings: Settings, api_url: str | None = None, monitor_url: str | None = None
) -> list[dict[str, Any]]:
    """Replay each configured month against the champion; retrain and promote on degradation."""
    reports = Path(settings.reports_dir)
    (reports / "drift").mkdir(parents=True, exist_ok=True)
    ref_df = load_month(
        settings.train_month, settings.data_dir, settings.sample_rows, settings.seed
    )
    baseline = MedianBaseline().fit(ref_df)
    history: list[dict[str, Any]] = []

    for month in settings.replay_months:
        champion, meta, version = load_alias(CHAMPION_ALIAS, settings)
        df = load_month(month, settings.data_dir, settings.sample_rows, settings.seed)
        local_preds = predict(champion, df)
        if api_url:
            preds, served_version = predict_remote(api_url, df)
        else:
            preds, served_version = local_preds, version

        y = df[TARGET].to_numpy()
        model_metrics = evaluate(y, preds)
        base_mae = evaluate(y, baseline.predict(df))["mae"]
        drift = _drift_for_month(
            champion, ref_df, df, local_preds, reports / "drift" / f"{month}.html", settings.seed
        )
        ref_mae = float(meta["metrics"]["mae"])
        degraded = model_metrics["mae"] > ref_mae * settings.degradation_factor
        row: dict[str, Any] = {
            "month": month,
            "model_version": served_version,
            "mae": model_metrics["mae"],
            "baseline_mae": base_mae,
            "ref_mae": ref_mae,
            "drift_share": drift["drift_share"],
            "n_drifted": drift["n_drifted"],
            "n": model_metrics["n"],
            "degraded": degraded,
            "retrained": False,
            "promoted": False,
        }
        if monitor_url:
            _post_evaluation(
                monitor_url,
                {
                    "replay_month": month,
                    "model_version": served_version,
                    "mae": model_metrics["mae"],
                    "baseline_mae": base_mae,
                    "ref_mae": ref_mae,
                    "drift_share": drift["drift_share"],
                    "n": model_metrics["n"],
                },
            )
        if degraded and monitor_url:
            # Ops mode: the monitor's alert launches the retrain Job; we only observe the registry.
            record = wait_for_retrain(month, version, settings)
            row["retrained"] = record is not None
            row["promoted"] = bool(record and record["promoted"])
            if row["promoted"] and api_url:
                wait_for_api_version(api_url, record["new_champion"])
        elif degraded:
            record = retrain(month, settings, ref_df)
            row["retrained"] = True
            row["promoted"] = bool(record["promoted"])
        history.append(row)

    _write_outputs(reports, history)
    return history


def evaluate_ci(fixture: str | Path, settings: Settings) -> int:
    """Gate: the model must beat the median baseline on a time holdout; writes models/."""
    df = load_file(fixture, settings.train_month, seed=settings.seed)
    fit_part, holdout = time_split(df, CI_HOLDOUT_FRAC)
    gate_model = train_model(fit_part, settings.seed)
    y = holdout[TARGET].to_numpy()
    model_metrics = evaluate(y, predict(gate_model, holdout))
    base_mae = evaluate(y, MedianBaseline().fit(fit_part).predict(holdout))["mae"]
    passed = model_metrics["mae"] < base_mae

    final = train_model(df, settings.seed)
    meta = {
        "model_version": "ci",
        "trained_on": [settings.train_month],
        "metrics": model_metrics,
        "baseline_mae": base_mae,
        "created_at": utc_now_iso(),
    }
    save_model_dir(final, Path(settings.models_dir) / "champion", meta)
    summary = {"passed": passed, "model_mae": model_metrics["mae"], "baseline_mae": base_mae}
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1
