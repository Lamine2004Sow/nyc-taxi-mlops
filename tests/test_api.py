"""Tests for the prediction API (FastAPI TestClient, in-process model)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nyc_taxi.api.main import ApiConfig, create_app
from nyc_taxi.api.model_store import ModelStore
from nyc_taxi.data import clean
from nyc_taxi.model import save_model_dir, train_model

FIXTURE = Path(__file__).parent / "fixtures" / "sample.parquet"


def _train_into(out_dir: Path, seed: int, version: str) -> Path:
    raw = pd.read_parquet(FIXTURE)
    df = clean(raw, "2020-01")
    booster = train_model(df, seed=seed)
    meta = {
        "model_version": version,
        "trained_on": ["2020-01"],
        "features": [],
        "metrics": {},
        "baseline_mae": 0.0,
        "created_at": "2026-01-01T00:00:00Z",
    }
    save_model_dir(booster, out_dir, meta)
    return out_dir


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("models")
    return {
        "v1": _train_into(base / "v1", seed=42, version="1"),
        "v2": _train_into(base / "v2", seed=7, version="2"),
    }


@pytest.fixture
def api(model_dirs: dict[str, Path], tmp_path: Path):
    """Yield (client, store, log_path) for an app serving model v1."""
    store = ModelStore(cache_dir=tmp_path / "cache")
    store.reload(model_dirs["v1"])
    log_path = tmp_path / "predictions.jsonl"
    cfg = ApiConfig(
        model_dir=None,
        model_alias="champion",
        tracking_uri=None,
        poll_seconds=30,
        prediction_log_path=log_path,
        cache_dir=tmp_path / "cache",
    )
    app = create_app(config=cfg, store=store)
    with TestClient(app) as client:
        yield client, store, log_path


def _trip(pu: int = 161, do: int = 236, request_id: str | None = None) -> dict:
    trip: dict = {
        "pickup_datetime": "2020-02-14T18:30:00",
        "PULocationID": pu,
        "DOLocationID": do,
    }
    if request_id is not None:
        trip["request_id"] = request_id
    return trip


def test_health_reports_loaded_model(api) -> None:
    client, _, _ = api
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "1", "model_loaded": True}


def test_health_503_without_model(tmp_path: Path) -> None:
    cfg = ApiConfig(None, "champion", None, 30, tmp_path / "p.jsonl", tmp_path / "c")
    with TestClient(create_app(config=cfg, store=ModelStore(tmp_path / "c"))) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"trips": [_trip()]}).status_code == 503


def test_predict_single_generates_request_id(api) -> None:
    client, _, _ = api
    resp = client.post("/predict", json={"trips": [_trip()]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "1"
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert isinstance(pred["request_id"], str) and pred["request_id"]
    assert isinstance(pred["duration_min"], float)


def test_predict_batch_preserves_order_and_ids(api) -> None:
    client, _, _ = api
    trips = [_trip(request_id="a"), _trip(pu=132, do=230, request_id="b"), _trip(request_id="c")]
    resp = client.post("/predict", json={"trips": trips})
    assert resp.status_code == 200
    assert [p["request_id"] for p in resp.json()["predictions"]] == ["a", "b", "c"]


def test_predict_invalid_zone_is_422(api) -> None:
    client, _, _ = api
    resp = client.post("/predict", json={"trips": [_trip(pu=300)]})
    assert resp.status_code == 422


def test_predict_empty_batch_is_422(api) -> None:
    client, _, _ = api
    assert client.post("/predict", json={"trips": []}).status_code == 422


def test_metrics_exposes_route_labels_and_counts_422(api) -> None:
    client, _, _ = api
    client.post("/predict", json={"trips": [_trip()]})
    client.post("/predict", json={"trips": [_trip(pu=300)]})
    text = client.get("/metrics").text
    assert 'http_requests_total{endpoint="/predict",method="POST",status="200"} 1.0' in text
    assert 'http_requests_total{endpoint="/predict",method="POST",status="422"} 1.0' in text
    assert 'endpoint="/health"' not in text
    assert "predictions_total 1.0" in text
    assert 'model_info{version="1"} 1.0' in text
    assert "prediction_duration_minutes_count 1.0" in text
    assert 'http_request_duration_seconds_count{endpoint="/predict"}' in text


def test_prediction_log_written(api) -> None:
    client, _, log_path = api
    client.post("/predict", json={"trips": [_trip(request_id="r1"), _trip(request_id="r2")]})
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [line["request_id"] for line in lines] == ["r1", "r2"]
    expected = {
        "request_id",
        "model_version",
        "pickup_datetime",
        "PULocationID",
        "DOLocationID",
        "duration_min",
        "ts",
    }
    assert set(lines[0]) == expected
    assert lines[0]["model_version"] == "1"
    assert lines[0]["PULocationID"] == 161


def test_hot_swap_changes_served_version(api, model_dirs: dict[str, Path]) -> None:
    client, store, _ = api
    assert client.get("/health").json()["model_version"] == "1"
    store.reload(model_dirs["v2"])
    assert client.get("/health").json()["model_version"] == "2"
    resp = client.post("/predict", json={"trips": [_trip()]})
    assert resp.json()["model_version"] == "2"
    text = client.get("/metrics").text
    assert 'model_info{version="2"} 1.0' in text
    # The initial load happened before the app bound its metrics, so only the swap counts.
    assert "model_reloads_total 1.0" in text
