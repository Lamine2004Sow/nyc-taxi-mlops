"""Tests for MLflow registration and alias handling (file-based tracking store)."""

from __future__ import annotations

import numpy as np

from nyc_taxi.model import load_model_dir, predict, train_model
from nyc_taxi.registry import download_alias, get_alias_version, log_and_register, set_alias


def test_register_alias_and_download(tmp_path, tracking_uri, sample_df):
    booster = train_model(sample_df, seed=3)
    meta = {"trained_on": ["2020-01"], "metrics": {"mae": 1.0}, "model_version": None}
    version = log_and_register(booster, meta, {"mae": 1.0}, {"num_leaves": 63}, tracking_uri)
    assert version == "1"
    assert get_alias_version("champion", tracking_uri) is None

    set_alias(version, "champion", tracking_uri)
    assert get_alias_version("champion", tracking_uri) == version

    root = download_alias("champion", tmp_path / "dl", tracking_uri)
    loaded, downloaded_meta = load_model_dir(root)
    assert downloaded_meta["model_version"] == version
    np.testing.assert_allclose(predict(booster, sample_df), predict(loaded, sample_df), rtol=1e-6)


def test_challenger_alias_does_not_move_champion(tmp_path, tracking_uri, sample_df):
    booster = train_model(sample_df, seed=3)
    meta = {"trained_on": ["2020-01"], "metrics": {"mae": 1.0}, "model_version": None}
    first = log_and_register(booster, meta, {"mae": 1.0}, {}, tracking_uri)
    second = log_and_register(booster, meta, {"mae": 0.9}, {}, tracking_uri)
    set_alias(first, "champion", tracking_uri)
    set_alias(second, "challenger", tracking_uri)
    assert get_alias_version("champion", tracking_uri) == first
    assert get_alias_version("challenger", tracking_uri) == second
