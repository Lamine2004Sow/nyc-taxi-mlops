"""Prometheus metrics for the prediction API.

Each app instance gets its own CollectorRegistry so tests never collide.
"""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

MINUTE_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180)


@dataclass(frozen=True)
class ApiMetrics:
    """Metric handles bound to one registry."""

    registry: CollectorRegistry
    http_requests: Counter
    http_duration: Histogram
    predictions: Counter
    prediction_minutes: Histogram
    model_info: Gauge
    model_reloads: Counter


def build_api_metrics() -> ApiMetrics:
    """Create the API metric set on a fresh registry."""
    registry = CollectorRegistry()
    return ApiMetrics(
        registry=registry,
        http_requests=Counter(
            "http_requests_total",
            "HTTP requests by route, method and status",
            ["endpoint", "method", "status"],
            registry=registry,
        ),
        http_duration=Histogram(
            "http_request_duration_seconds",
            "HTTP request latency by route",
            ["endpoint"],
            registry=registry,
        ),
        predictions=Counter(
            "predictions_total", "Trip duration predictions served", registry=registry
        ),
        prediction_minutes=Histogram(
            "prediction_duration_minutes",
            "Distribution of predicted trip durations (minutes)",
            buckets=MINUTE_BUCKETS,
            registry=registry,
        ),
        model_info=Gauge(
            "model_info",
            "Model currently served, one series per version",
            ["version"],
            registry=registry,
        ),
        model_reloads=Counter(
            "model_reloads_total", "Model loads/hot-swaps performed", registry=registry
        ),
    )
