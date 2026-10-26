"""Data drift reports with Evidently (HTML report + a small JSON-friendly summary)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DRIFT_COLUMNS = ["hour", "dow", "centroid_km", "PULocationID", "DOLocationID", "prediction"]


def _metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat list of metric entries from an Evidently snapshot dict."""
    return [m for m in payload.get("metrics", []) if isinstance(m, dict)]


def _drift_count_value(payload: dict[str, Any]) -> dict[str, Any]:
    """Value of the DriftedColumnsCount metric ({count, share}), empty if absent."""
    for metric in _metrics(payload):
        if str(metric.get("metric_name", "")).startswith("DriftedColumnsCount"):
            value = metric.get("value")
            if isinstance(value, dict):
                return value
    return {}


def _drifted_columns(payload: dict[str, Any]) -> list[str]:
    """Columns whose ValueDrift score reaches its configured threshold (Evidently rule)."""
    drifted: list[str] = []
    for metric in _metrics(payload):
        config = metric.get("config", {})
        if not str(metric.get("metric_name", "")).startswith("ValueDrift("):
            continue
        score = metric.get("value")
        threshold = config.get("threshold")
        if isinstance(score, (int, float)) and threshold is not None and score >= threshold:
            drifted.append(str(config.get("column")))
    return sorted(set(drifted))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: str | Path) -> dict:
    """Evidently DataDriftPreset on the reference vs current frames; writes out_html."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    ref = reference[DRIFT_COLUMNS].copy()
    cur = current[DRIFT_COLUMNS].copy()
    for frame in (ref, cur):
        frame[["PULocationID", "DOLocationID"]] = frame[["PULocationID", "DOLocationID"]].astype(
            "float64"
        )

    snapshot = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))

    payload = snapshot.dict()
    value = _drift_count_value(payload)
    return {
        "drift_share": float(value.get("share", 0.0)),
        "drifted_columns": _drifted_columns(payload),
        "n_drifted": int(value.get("count", 0)),
    }
