"""Monitor FastAPI app: POST /evaluations, POST /alerts, GET /metrics, GET /health."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from nyc_taxi.monitor.jobs import ALERTNAME, RetrainLauncher
from nyc_taxi.monitor.metrics import MonitorMetrics
from nyc_taxi.monitor.schemas import AlertmanagerPayload, Evaluation


def _record_evaluation(metrics: MonitorMetrics, ev: Evaluation) -> None:
    metrics.model_mae.labels(replay_month=ev.replay_month, model_version=ev.model_version).set(
        ev.mae
    )
    metrics.model_mae_latest.set(ev.mae)
    metrics.baseline_mae_latest.set(ev.baseline_mae)
    metrics.model_ref_mae.set(ev.ref_mae)
    metrics.drift_share_latest.set(ev.drift_share)
    metrics.replay_month_info.clear()
    metrics.replay_month_info.labels(month=ev.replay_month).set(1)


def create_app(
    launcher: RetrainLauncher | None = None, metrics: MonitorMetrics | None = None
) -> FastAPI:
    """Build the monitor app. Inject `launcher` (with a fake k8s client) in tests."""
    if launcher is None:
        launcher = RetrainLauncher.from_env(metrics)
    metrics = metrics or launcher.metrics

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.last_evaluation = None
        yield

    app = FastAPI(title="nyc-taxi monitor", version="0.1.0", lifespan=lifespan)
    app.state.launcher = launcher
    app.state.metrics = metrics
    app.state.last_evaluation = None

    @app.post("/evaluations")
    def post_evaluation(ev: Evaluation) -> dict[str, Any]:
        _record_evaluation(metrics, ev)
        app.state.last_evaluation = ev
        return {"status": "ok", "replay_month": ev.replay_month}

    @app.post("/alerts")
    def post_alerts(payload: AlertmanagerPayload) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        for alert in payload.alerts:
            metrics.alerts_received.labels(status=alert.status).inc()
            alertname = alert.labels.get("alertname")
            if alert.status != "firing":
                decisions.append({"alertname": alertname, "action": "ignored", "reason": "status"})
                continue
            if alertname != ALERTNAME:
                decisions.append({"alertname": alertname, "action": "ignored", "reason": "name"})
                continue
            last = app.state.last_evaluation
            if last is None:
                decisions.append(
                    {"alertname": alertname, "action": "ignored", "reason": "no evaluation yet"}
                )
                continue
            decisions.append(asdict(launcher.handle_alert(last.replay_month)))
        return {"decisions": decisions}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "retrain_mode": launcher.mode}

    @app.get("/metrics", response_model=None)
    def prometheus_metrics() -> Response:
        return Response(content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
