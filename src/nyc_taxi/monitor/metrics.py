"""Prometheus metrics for the monitor service (one registry per instance)."""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge


@dataclass(frozen=True)
class MonitorMetrics:
    """Metric handles bound to one registry."""

    registry: CollectorRegistry
    model_mae: Gauge
    model_mae_latest: Gauge
    baseline_mae_latest: Gauge
    model_ref_mae: Gauge
    drift_share_latest: Gauge
    replay_month_info: Gauge
    alerts_received: Counter
    retrain_jobs_triggered: Counter


def build_monitor_metrics() -> MonitorMetrics:
    """Create the monitor metric set on a fresh registry."""
    registry = CollectorRegistry()
    return MonitorMetrics(
        registry=registry,
        model_mae=Gauge(
            "model_mae",
            "MAE of the served model per replay month",
            ["replay_month", "model_version"],
            registry=registry,
        ),
        model_mae_latest=Gauge(
            "model_mae_latest", "MAE of the latest evaluated month", registry=registry
        ),
        baseline_mae_latest=Gauge(
            "baseline_mae_latest", "Baseline MAE of the latest evaluated month", registry=registry
        ),
        model_ref_mae=Gauge(
            "model_ref_mae", "Reference (validation) MAE of the served model", registry=registry
        ),
        drift_share_latest=Gauge(
            "drift_share_latest", "Share of drifted columns, latest evaluation", registry=registry
        ),
        replay_month_info=Gauge(
            "replay_month_info", "Latest replayed month (value 1)", ["month"], registry=registry
        ),
        alerts_received=Counter(
            "alerts_received_total",
            "Alertmanager alerts received by status",
            ["status"],
            registry=registry,
        ),
        retrain_jobs_triggered=Counter(
            "retrain_jobs_triggered_total", "Retrain Jobs created", registry=registry
        ),
    )
