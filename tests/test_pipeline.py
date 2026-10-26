"""Pipeline tests: offline replay with forced degradation, retraining, promotion and CI gate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from nyc_taxi import __main__ as cli
from nyc_taxi import pipeline
from nyc_taxi.config import Settings
from nyc_taxi.registry import get_alias_version

from .conftest import FIXTURE


@pytest.fixture
def settings(tmp_path, tracking_uri) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        models_dir=str(tmp_path / "models"),
        tracking_uri=tracking_uri,
        train_month="2020-01",
        replay_months=("2020-02", "2020-03"),
        sample_rows=1500,
        degradation_factor=0.5,  # forced: every replay month counts as degraded
        promotion_margin=0.02,
        seed=42,
    )


@pytest.fixture
def fake_months(monkeypatch, synthetic_month):
    """Replace the network-backed loader: March 2020 durations are 60% longer (drift)."""

    def _load(month, data_dir, sample_rows=None, seed=42):
        shift = 1.6 if month == "2020-03" else 1.0
        seed = {"2020-01": 1, "2020-02": 2, "2020-03": 3}[month]
        return synthetic_month(month, n=1500, seed=seed, shift=shift)

    monkeypatch.setattr(pipeline, "load_month", _load)


def test_replay_retrains_and_promotes(settings, fake_months):
    trained = pipeline.train_champion(settings)
    assert trained["model_version"] == "1"
    assert get_alias_version("champion", settings.tracking_uri) == "1"

    history = pipeline.run_replay(settings)

    reports = Path(settings.reports_dir)
    assert [row["month"] for row in history] == ["2020-02", "2020-03"]
    assert (reports / "history.json").exists()
    assert (reports / "mae_by_month.png").exists()
    assert (reports / "replay.md").exists()
    assert (reports / "drift" / "2020-02.html").exists()
    assert (reports / "drift" / "2020-03.html").exists()

    lines = (reports / "promotions.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    promoted = [r for r in records if r["promoted"]]
    assert promoted, "the shifted month should produce a promoted challenger"
    assert get_alias_version("champion", settings.tracking_uri) == promoted[-1]["new_champion"]
    assert promoted[0]["previous_champion"] == "1"

    saved = json.loads((reports / "history.json").read_text())
    assert all(row["degraded"] for row in saved)
    assert saved[1]["retrained"] is True
    assert saved[1]["promoted"] == promoted[-1]["promoted"]


def test_retrain_without_win_only_moves_challenger(settings, fake_months):
    pipeline.train_champion(settings)
    strict = replace(settings, promotion_margin=0.9)  # challenger can never win by 90%
    record = pipeline.retrain("2020-02", strict)
    assert record["promoted"] is False
    assert record["previous_champion"] == "1"
    assert get_alias_version("champion", strict.tracking_uri) == "1"
    assert get_alias_version("challenger", strict.tracking_uri) == record["challenger_version"]


def test_evaluate_ci_passes_on_fixture(tmp_path, settings):
    gate_settings = replace(settings, models_dir=str(tmp_path / "ci_models"))
    assert pipeline.evaluate_ci(FIXTURE, gate_settings) == 0
    assert (tmp_path / "ci_models" / "champion" / "model.txt").exists()
    meta = json.loads((tmp_path / "ci_models" / "champion" / "meta.json").read_text())
    assert meta["metrics"]["mae"] < meta["baseline_mae"]


def test_cli_evaluate_ci_exit_code(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(
        cli, "get_settings", lambda: replace(settings, models_dir=str(tmp_path / "m"))
    )
    assert cli.main(["evaluate-ci", "--fixture", str(FIXTURE)]) == 0
