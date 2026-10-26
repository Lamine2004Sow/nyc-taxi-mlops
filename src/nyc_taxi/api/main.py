"""FastAPI prediction service: POST /predict, GET /health, GET /metrics."""

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from nyc_taxi.api.metrics import build_api_metrics
from nyc_taxi.api.model_store import ModelStore, is_remote_tracking
from nyc_taxi.api.schemas import (
    HealthResponse,
    PredictionOut,
    PredictRequest,
    PredictResponse,
    TripIn,
)
from nyc_taxi.model import predict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiConfig:
    """Runtime settings, normally read from the environment."""

    model_dir: str | None
    model_alias: str
    tracking_uri: str | None
    poll_seconds: float
    prediction_log_path: Path
    cache_dir: Path

    @classmethod
    def from_env(cls) -> "ApiConfig":
        return cls(
            model_dir=os.getenv("MODEL_DIR", "models/champion"),
            model_alias=os.getenv("MODEL_ALIAS", "champion"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
            poll_seconds=float(os.getenv("MODEL_POLL_SECONDS", "30")),
            prediction_log_path=Path(
                os.getenv("PREDICTION_LOG_PATH", "predictions/predictions.jsonl")
            ),
            cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "models/cache")),
        )


class PredictionLog:
    """Appends one JSON line per prediction. Thread-safe."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, records: list[dict[str, object]]) -> None:
        if not records:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(r) + "\n" for r in records)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(payload)


def _bootstrap_model(store: ModelStore, cfg: ApiConfig) -> None:
    """Load the model at startup: MLflow alias first (if remote), then MODEL_DIR fallback."""
    if is_remote_tracking(cfg.tracking_uri) and cfg.tracking_uri is not None:
        try:
            store.sync_alias(cfg.model_alias, cfg.tracking_uri)
        except Exception:
            log.exception("could not load alias %s from MLflow", cfg.model_alias)
    if not store.loaded and cfg.model_dir:
        path = Path(cfg.model_dir)
        if path.exists():
            store.reload(path)
        else:
            log.warning("MODEL_DIR %s not found; API will report model_loaded=false", path)


def _endpoint_label(request: Request) -> str:
    """Route template (e.g. /predict), never the raw URL, to keep label cardinality bounded."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if path else "unmatched"


def _to_frame(trips: list[TripIn]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime([t.pickup_datetime for t in trips]),
            "PULocationID": np.asarray([t.PULocationID for t in trips], dtype="int32"),
            "DOLocationID": np.asarray([t.DOLocationID for t in trips], dtype="int32"),
        }
    )


def create_app(config: ApiConfig | None = None, store: ModelStore | None = None) -> FastAPI:
    """Build the API app. Pass `store` to inject a pre-loaded model (used in tests)."""
    cfg = config or ApiConfig.from_env()
    metrics = build_api_metrics()
    model_store = store or ModelStore(cache_dir=cfg.cache_dir)
    model_store.bind_metrics(metrics)
    pred_log = PredictionLog(cfg.prediction_log_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(_bootstrap_model, model_store, cfg)
        if is_remote_tracking(cfg.tracking_uri) and cfg.tracking_uri is not None:
            model_store.start_polling(cfg.model_alias, cfg.tracking_uri, cfg.poll_seconds)
        try:
            yield
        finally:
            model_store.stop()

    app = FastAPI(title="nyc-taxi duration API", version="0.1.0", lifespan=lifespan)
    app.state.store = model_store
    app.state.metrics = metrics

    @app.middleware("http")
    async def record_http(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            endpoint = _endpoint_label(request)
            metrics.http_requests.labels(endpoint, request.method, str(status)).inc()
            metrics.http_duration.labels(endpoint).observe(time.perf_counter() - start)

    @app.get("/health", response_model=None)
    def health() -> HealthResponse | JSONResponse:
        current = model_store.current()
        if current is None:
            body = HealthResponse(status="no_model", model_version=None, model_loaded=False)
            return JSONResponse(status_code=503, content=body.model_dump())
        _, meta = current
        return HealthResponse(
            status="ok", model_version=str(meta.get("model_version")), model_loaded=True
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict_trips(body: PredictRequest) -> PredictResponse:
        current = model_store.current()
        if current is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        booster, meta = current
        version = str(meta.get("model_version"))
        ids = [t.request_id or uuid.uuid4().hex for t in body.trips]
        preds = np.asarray(predict(booster, _to_frame(body.trips)), dtype=float)

        metrics.predictions.inc(len(ids))
        for value in preds:
            metrics.prediction_minutes.observe(float(value))
        ts = datetime.now(UTC).isoformat()
        pred_log.append(
            [
                {
                    "request_id": rid,
                    "model_version": version,
                    "pickup_datetime": trip.pickup_datetime.isoformat(),
                    "PULocationID": trip.PULocationID,
                    "DOLocationID": trip.DOLocationID,
                    "duration_min": float(value),
                    "ts": ts,
                }
                for rid, trip, value in zip(ids, body.trips, preds, strict=True)
            ]
        )
        return PredictResponse(
            model_version=version,
            predictions=[
                PredictionOut(request_id=rid, duration_min=float(value))
                for rid, value in zip(ids, preds, strict=True)
            ],
        )

    @app.get("/metrics", response_model=None)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
