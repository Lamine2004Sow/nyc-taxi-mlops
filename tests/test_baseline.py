"""Tests for the median baseline and its JSON persistence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nyc_taxi.baseline import MedianBaseline


def _frame(durations, pu, do, hours):
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                pd.Timestamp("2020-01-01") + pd.Timedelta(hours=h) for h in hours
            ],
            "PULocationID": pu,
            "DOLocationID": do,
            "duration_min": durations,
        }
    )


def test_predicts_group_median_when_group_known():
    train = _frame([10.0, 20.0, 30.0, 100.0], [1, 1, 1, 1], [2, 2, 2, 2], [8, 8, 8, 9])
    model = MedianBaseline().fit(train)
    preds = model.predict(_frame([0.0, 0.0], [1, 1], [2, 2], [8, 9]))
    assert preds[0] == 20.0  # median of (10, 20, 30) for hour 8
    assert preds[1] == 100.0  # only one sample for hour 9


def test_falls_back_to_pair_then_global():
    train = _frame([10.0, 30.0], [1, 1], [2, 2], [8, 8])
    model = MedianBaseline().fit(train)
    # unseen hour -> pair median (20)
    pair_pred = model.predict(_frame([0.0], [1], [2], [3]))[0]
    assert pair_pred == 20.0
    # unseen pair -> global median (20)
    global_pred = model.predict(_frame([0.0], [5], [6], [3]))[0]
    assert global_pred == 20.0


def test_predict_before_fit_raises():
    try:
        MedianBaseline().predict(_frame([1.0], [1], [2], [0]))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_roundtrip_json_keeps_predictions(sample_df):
    model = MedianBaseline().fit(sample_df)
    payload = json.dumps(model.to_dict())
    restored = MedianBaseline.from_dict(json.loads(payload))
    np.testing.assert_allclose(model.predict(sample_df), restored.predict(sample_df))


def test_baseline_beats_global_constant_on_fixture(sample_df):
    model = MedianBaseline().fit(sample_df)
    preds = model.predict(sample_df)
    constant = np.full(len(sample_df), sample_df["duration_min"].median())
    y = sample_df["duration_min"].to_numpy()
    assert np.abs(y - preds).mean() <= np.abs(y - constant).mean()
