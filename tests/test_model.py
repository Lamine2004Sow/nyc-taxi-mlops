"""Tests for training, prediction, metrics and model persistence."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pytest

from nyc_taxi.baseline import MedianBaseline
from nyc_taxi.features import FEATURES
from nyc_taxi.model import (
    evaluate,
    load_model_dir,
    predict,
    save_model_dir,
    time_split,
    train_model,
)


def test_time_split_is_ordered_and_complete(sample_df):
    head, tail = time_split(sample_df, 0.2)
    assert len(head) + len(tail) == len(sample_df)
    assert head["tpep_pickup_datetime"].max() <= tail["tpep_pickup_datetime"].min()


def test_train_model_returns_booster(sample_df):
    booster = train_model(sample_df, seed=42)
    assert isinstance(booster, lgb.Booster)
    assert booster.feature_name() == FEATURES


def test_model_beats_median_baseline_on_holdout(sample_df):
    fit_part, holdout = time_split(sample_df, 0.2)
    booster = train_model(fit_part, seed=42)
    y = holdout["duration_min"].to_numpy()
    model_mae = evaluate(y, predict(booster, holdout))["mae"]
    base_mae = evaluate(y, MedianBaseline().fit(fit_part).predict(holdout))["mae"]
    assert model_mae < base_mae


def test_predict_returns_one_value_per_row(sample_df, trained_model_dir):
    booster, _ = load_model_dir(trained_model_dir)
    preds = predict(booster, sample_df.head(50))
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


def test_evaluate_metrics():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    pred = np.array([12.0, 18.0, 30.0, 50.0])
    metrics = evaluate(y, pred)
    assert metrics["mae"] == pytest.approx(3.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt((4 + 4 + 0 + 100) / 4))
    assert metrics["n"] == 4
    assert set(metrics) == {"mae", "rmse", "p90_abs_err", "n"}


def test_save_and_load_roundtrip(tmp_path, sample_df):
    booster = train_model(sample_df, seed=1)
    out = save_model_dir(booster, tmp_path / "m", {"model_version": "7", "trained_on": ["x"]})
    assert (out / "model.txt").exists()
    loaded, meta = load_model_dir(out)
    assert meta["model_version"] == "7"
    assert meta["features"] == FEATURES
    np.testing.assert_allclose(predict(booster, sample_df), predict(loaded, sample_df), rtol=1e-6)
    assert json.loads((out / "meta.json").read_text())["trained_on"] == ["x"]
