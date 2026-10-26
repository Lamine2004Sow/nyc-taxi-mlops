"""Shared fixtures: fixture trips, centroids, a trained model directory and synthetic months."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from nyc_taxi.data import clean, load_file
from nyc_taxi.model import evaluate, predict, save_model_dir, train_model
from nyc_taxi.zones import load_centroids

TESTS_DIR = Path(__file__).parent
FIXTURE = TESTS_DIR / "fixtures" / "sample.parquet"


def _load_make_sample():
    spec = importlib.util.spec_from_file_location(
        "make_sample", TESTS_DIR / "fixtures" / "make_sample.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


make_sample = _load_make_sample()


@pytest.fixture(scope="session")
def raw_sample_df() -> pd.DataFrame:
    """The fixture exactly as stored (raw TLC schema, uncleaned)."""
    return pd.read_parquet(FIXTURE)


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """Cleaned fixture trips for 2020-01."""
    return load_file(FIXTURE, "2020-01")


@pytest.fixture(scope="session")
def centroids() -> pd.DataFrame:
    """Packaged zone centroids indexed by LocationID."""
    return load_centroids()


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, sample_df) -> Path:
    """A model directory (model.txt + meta.json) trained on the fixture."""
    booster = train_model(sample_df, seed=42)
    metrics = evaluate(sample_df["duration_min"].to_numpy(), predict(booster, sample_df))
    out = tmp_path_factory.mktemp("model") / "champion"
    save_model_dir(
        booster,
        out,
        {
            "model_version": "test",
            "trained_on": ["2020-01"],
            "metrics": metrics,
            "baseline_mae": 0.0,
        },
    )
    return out


@pytest.fixture
def tracking_uri(tmp_path) -> str:
    """Isolated file-based MLflow tracking store."""
    return f"file://{tmp_path}/mlruns"


@pytest.fixture
def synthetic_month() -> Callable[..., pd.DataFrame]:
    """Build cleaned synthetic trips for a month; shift > 1 simulates degradation."""

    def _make(month: str, n: int = 1500, seed: int = 1, shift: float = 1.0) -> pd.DataFrame:
        raw = make_sample.make_trips(n, month, seed=seed, shift=shift, junk_frac=0.02)
        return clean(raw, month)

    return _make
