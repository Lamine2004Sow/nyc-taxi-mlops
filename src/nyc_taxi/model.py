"""LightGBM duration model: training, prediction, metrics and on-disk persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from nyc_taxi.features import CATEGORICAL, FEATURES, TARGET, build_features

TRAIN_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1,
}
NUM_ROUNDS = 300
EARLY_STOPPING_ROUNDS = 30
VALID_FRAC = 0.1
MODEL_FILE = "model.txt"
META_FILE = "meta.json"


def time_split(df: pd.DataFrame, tail_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by pickup time: returns (head, tail) where tail holds the last tail_frac rows."""
    ordered = df.sort_values("tpep_pickup_datetime", kind="stable")
    cut = int(len(ordered) * (1.0 - tail_frac))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def train_model(train_df: pd.DataFrame, seed: int = 42) -> lgb.Booster:
    """Train a L1-objective booster; early-stops on the last 10% (time-ordered) of train_df."""
    head, tail = time_split(train_df, VALID_FRAC)
    dtrain = lgb.Dataset(
        build_features(head),
        label=head[TARGET].to_numpy(),
        categorical_feature=CATEGORICAL,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        build_features(tail),
        label=tail[TARGET].to_numpy(),
        reference=dtrain,
        categorical_feature=CATEGORICAL,
        free_raw_data=False,
    )
    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(0),
    ]
    return lgb.train(
        {**TRAIN_PARAMS, "seed": seed},
        dtrain,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predicted trip durations in minutes."""
    return np.asarray(booster.predict(build_features(df)), dtype=float)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, 90th percentile absolute error and sample size."""
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_err = np.abs(err)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "p90_abs_err": float(np.percentile(abs_err, 90)),
        "n": int(len(err)),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def save_model_dir(booster: lgb.Booster, out_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Write model.txt and meta.json into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / MODEL_FILE))
    full_meta = {"features": FEATURES, **meta}
    (out / META_FILE).write_text(json.dumps(full_meta, indent=2, default=_json_default))
    return out


def load_model_dir(path: str | Path) -> tuple[lgb.Booster, dict[str, Any]]:
    """Load (booster, meta) from a directory written by save_model_dir."""
    root = Path(path)
    booster = lgb.Booster(model_file=str(root / MODEL_FILE))
    meta = json.loads((root / META_FILE).read_text())
    return booster, meta
