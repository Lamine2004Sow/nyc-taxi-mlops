"""Tests for the Evidently drift report wrapper."""

from __future__ import annotations

from nyc_taxi.drift import DRIFT_COLUMNS, drift_report
from nyc_taxi.features import build_features


def _with_prediction(df):
    feats = build_features(df)
    feats["prediction"] = df["duration_min"].to_numpy()
    return feats


def test_no_drift_on_identical_halves(tmp_path, sample_df):
    frame = _with_prediction(sample_df)
    half = len(frame) // 2
    summary = drift_report(frame.iloc[:half], frame.iloc[:half], tmp_path / "same.html")
    assert summary["drift_share"] == 0.0
    assert summary["n_drifted"] == 0
    assert summary["drifted_columns"] == []
    assert (tmp_path / "same.html").exists()


def test_detects_shift_in_centroid_distance(tmp_path, sample_df):
    frame = _with_prediction(sample_df)
    half = len(frame) // 2
    ref = frame.iloc[:half]
    cur = frame.iloc[half:].copy()
    cur["centroid_km"] = cur["centroid_km"] * 3.0
    summary = drift_report(ref, cur, tmp_path / "shift.html")
    assert "centroid_km" in summary["drifted_columns"]
    assert summary["n_drifted"] >= 1
    assert 0.0 < summary["drift_share"] <= 1.0
    assert set(summary["drifted_columns"]) <= set(DRIFT_COLUMNS)
