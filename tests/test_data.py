"""Tests for raw trip loading, cleaning, sampling and downloads."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from nyc_taxi import data
from nyc_taxi.data import clean, download_month, load_file, raw_path

from .conftest import FIXTURE


def _raw_rows() -> pd.DataFrame:
    base = pd.Timestamp("2020-01-15 08:00")
    rows = [
        # good trip, 10 minutes
        (base, base + pd.Timedelta(minutes=10), 10, 20),
        # too short (0 minutes)
        (base, base, 10, 20),
        # too long (4 hours, above the 180 minute cap)
        (base, base + pd.Timedelta(hours=4), 10, 20),
        # unknown pickup zone 264
        (base, base + pd.Timedelta(minutes=10), 264, 20),
        # unknown dropoff zone 265
        (base, base + pd.Timedelta(minutes=10), 10, 265),
        # pickup outside the month
        (pd.Timestamp("2019-12-31 23:00"), pd.Timestamp("2019-12-31 23:20"), 10, 20),
        # missing zone
        (base, base + pd.Timedelta(minutes=10), None, 20),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
        ],
    )


def test_clean_keeps_only_valid_in_month_trips():
    cleaned = clean(_raw_rows(), "2020-01")
    assert len(cleaned) == 1
    assert cleaned["duration_min"].iloc[0] == pytest.approx(10.0)


def test_clean_adds_request_id_and_is_stable():
    first = clean(_raw_rows(), "2020-01")
    second = clean(_raw_rows(), "2020-01")
    assert first["request_id"].str.len().eq(16).all()
    assert first["request_id"].tolist() == second["request_id"].tolist()


def test_clean_request_id_is_sha1_prefix():
    cleaned = clean(_raw_rows(), "2020-01")
    row = cleaned.iloc[0]
    raw = f"{row.tpep_pickup_datetime.isoformat()}|{row.tpep_dropoff_datetime.isoformat()}|10|20|0"
    assert row.request_id == hashlib.sha1(raw.encode()).hexdigest()[:16]


def test_load_file_reads_only_needed_columns(sample_df):
    assert set(sample_df.columns) == {
        "request_id",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "duration_min",
    }
    assert "trip_distance" not in sample_df.columns
    assert "fare_amount" not in sample_df.columns


def test_load_file_casts_ids_and_datetimes(sample_df):
    assert str(sample_df["PULocationID"].dtype) == "int32"
    assert str(sample_df["DOLocationID"].dtype) == "int32"
    assert str(sample_df["tpep_pickup_datetime"].dtype).startswith("datetime64")


def test_load_file_fixture_range(sample_df, raw_sample_df):
    assert 0 < len(sample_df) < len(raw_sample_df)
    assert sample_df["PULocationID"].between(1, 263).all()
    assert sample_df["duration_min"].between(1, 180).all()


def test_load_file_sampling_is_deterministic():
    first = load_file(FIXTURE, "2020-01", sample_rows=500, seed=7)
    second = load_file(FIXTURE, "2020-01", sample_rows=500, seed=7)
    assert len(first) == 500
    assert first["request_id"].tolist() == second["request_id"].tolist()


def test_download_month_skips_existing_file(tmp_path, monkeypatch):
    path = raw_path("2020-01", str(tmp_path))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"already here")

    def _no_network(*args, **kwargs):
        raise AssertionError("network must not be used when the file exists")

    monkeypatch.setattr(data.requests, "get", _no_network)
    assert download_month("2020-01", str(tmp_path)) == path


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self._payload


def test_download_month_writes_manifest(tmp_path, monkeypatch):
    payload = b"PAR1-fake-parquet-bytes"
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: _FakeResponse(payload))
    path = download_month("2020-02", str(tmp_path))
    assert path.read_bytes() == payload
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    entry = manifest["2020-02"]
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["bytes"] == len(payload)
    assert entry["url"].endswith("yellow_tripdata_2020-02.parquet")
