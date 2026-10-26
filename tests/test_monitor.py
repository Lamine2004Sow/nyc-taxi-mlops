"""Tests for the monitor service: evaluations, alert handling, retrain Job creation."""

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nyc_taxi.monitor.jobs import APP_LABEL, RetrainLauncher, build_retrain_job
from nyc_taxi.monitor.main import create_app
from nyc_taxi.monitor.metrics import build_monitor_metrics


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeBatchApi:
    """Stands in for kubernetes BatchV1Api."""

    def __init__(self, active: bool = False) -> None:
        self.created: list[Any] = []
        self.active = active

    def create_namespaced_job(self, namespace: str, body: Any) -> Any:
        self.created.append(body)
        return body

    def list_namespaced_job(self, namespace: str, label_selector: str) -> Any:
        assert label_selector == f"app={APP_LABEL}"
        items = [SimpleNamespace(status=SimpleNamespace(active=1))] if self.active else []
        return SimpleNamespace(items=items)


EVALUATION = {
    "replay_month": "2020-03",
    "model_version": "1",
    "mae": 4.2,
    "baseline_mae": 5.0,
    "ref_mae": 3.0,
    "drift_share": 0.4,
    "n": 1000,
}


def _alert(status: str = "firing", alertname: str = "ModelMaeDegraded") -> dict:
    return {
        "status": status,
        "labels": {"alertname": alertname, "severity": "warning"},
        "annotations": {},
        "fingerprint": "abc",
    }


def _payload(*alerts: dict) -> dict:
    return {"version": "4", "status": "firing", "alerts": list(alerts)}


@pytest.fixture
def setup():
    """Yield (client, fake_api, clock, metrics) with a k8s-mode launcher."""
    fake = FakeBatchApi()
    clock = FakeClock()
    metrics = build_monitor_metrics()
    launcher = RetrainLauncher(
        mode="k8s",
        namespace="nyc-taxi",
        image="nyc-taxi:test",
        cooldown_seconds=600,
        mlflow_uri="http://mlflow:5000",
        batch_api=fake,
        metrics=metrics,
        clock=clock,
    )
    with TestClient(create_app(launcher=launcher)) as client:
        yield client, fake, clock, metrics


def test_health(setup) -> None:
    client, _, _, _ = setup
    assert client.get("/health").json() == {"status": "ok", "retrain_mode": "k8s"}


def test_evaluation_updates_gauges(setup) -> None:
    client, _, _, metrics = setup
    resp = client.post("/evaluations", json=EVALUATION)
    assert resp.status_code == 200
    reg = metrics.registry
    assert reg.get_sample_value("model_mae_latest") == pytest.approx(4.2)
    assert reg.get_sample_value("baseline_mae_latest") == pytest.approx(5.0)
    assert reg.get_sample_value("model_ref_mae") == pytest.approx(3.0)
    assert reg.get_sample_value("drift_share_latest") == pytest.approx(0.4)
    assert reg.get_sample_value(
        "model_mae", {"replay_month": "2020-03", "model_version": "1"}
    ) == pytest.approx(4.2)
    assert reg.get_sample_value("replay_month_info", {"month": "2020-03"}) == 1.0
    text = client.get("/metrics").text
    assert "model_mae_latest 4.2" in text


def test_firing_alert_creates_one_job(setup) -> None:
    client, fake, _, metrics = setup
    client.post("/evaluations", json=EVALUATION)
    resp = client.post("/alerts", json=_payload(_alert()))
    assert resp.status_code == 200
    assert len(fake.created) == 1
    job = fake.created[0]
    assert job.metadata.name.startswith("nyc-taxi-retrain-202003-")
    assert job.spec.backoff_limit == 0
    assert job.spec.ttl_seconds_after_finished == 3600
    assert job.metadata.labels == {"app": APP_LABEL}
    container = job.spec.template.spec.containers[0]
    assert container.command == ["python", "-m", "nyc_taxi", "retrain", "--month", "2020-03"]
    env = {e.name: e.value for e in container.env}
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert env["DATA_DIR"].startswith("/tmp")
    assert metrics.registry.get_sample_value("retrain_jobs_triggered_total") == 1.0
    assert resp.json()["decisions"][0]["action"] == "created"


def test_second_firing_within_cooldown_creates_no_job(setup) -> None:
    client, fake, clock, _ = setup
    client.post("/evaluations", json=EVALUATION)
    client.post("/alerts", json=_payload(_alert()))
    clock.now += 60
    resp = client.post("/alerts", json=_payload(_alert()))
    assert len(fake.created) == 1
    assert resp.json()["decisions"][0] == {
        "action": "skipped",
        "reason": "cooldown",
        "month": "2020-03",
        "job_name": None,
    }


def test_firing_after_cooldown_creates_another_job(setup) -> None:
    client, fake, clock, _ = setup
    client.post("/evaluations", json=EVALUATION)
    client.post("/alerts", json=_payload(_alert()))
    clock.now += 601
    client.post("/alerts", json=_payload(_alert()))
    assert len(fake.created) == 2


def test_active_job_blocks_creation(setup) -> None:
    client, fake, _, _ = setup
    fake.active = True
    client.post("/evaluations", json=EVALUATION)
    resp = client.post("/alerts", json=_payload(_alert()))
    assert fake.created == []
    assert resp.json()["decisions"][0]["reason"] == "active job"


def test_resolved_alert_is_ignored(setup) -> None:
    client, fake, _, metrics = setup
    client.post("/evaluations", json=EVALUATION)
    resp = client.post("/alerts", json=_payload(_alert(status="resolved")))
    assert fake.created == []
    assert resp.json()["decisions"][0]["action"] == "ignored"
    assert metrics.registry.get_sample_value("alerts_received_total", {"status": "resolved"}) == 1.0


def test_non_matching_alertname_is_ignored(setup) -> None:
    client, fake, _, metrics = setup
    client.post("/evaluations", json=EVALUATION)
    resp = client.post("/alerts", json=_payload(_alert(alertname="ApiHighErrorRate")))
    assert fake.created == []
    assert resp.json()["decisions"][0]["reason"] == "name"
    assert metrics.registry.get_sample_value("alerts_received_total", {"status": "firing"}) == 1.0


def test_alert_without_evaluation_is_ignored(setup) -> None:
    client, fake, _, _ = setup
    resp = client.post("/alerts", json=_payload(_alert()))
    assert fake.created == []
    assert resp.json()["decisions"][0]["reason"] == "no evaluation yet"


def test_dry_run_records_decision_without_api_calls() -> None:
    fake = FakeBatchApi()
    launcher = RetrainLauncher(
        mode="dry-run", namespace="ns", image="img", batch_api=fake, clock=FakeClock()
    )
    decision = launcher.handle_alert("2020-04")
    assert decision.action == "dry-run"
    assert fake.created == []
    assert launcher.decisions == [decision]
    assert launcher.handle_alert("2020-04").reason == "cooldown"


def test_build_retrain_job_without_mlflow_uri_writes_under_tmp() -> None:
    job = build_retrain_job("2020-05", "ns", "img:1", mlflow_uri=None, now=1700000000)
    assert job.metadata.name == "nyc-taxi-retrain-202005-1700000000"
    env = {e.name: e.value for e in job.spec.template.spec.containers[0].env}
    assert "MLFLOW_TRACKING_URI" not in env
    assert all(env[k].startswith("/tmp") for k in ("DATA_DIR", "REPORTS_DIR", "MODELS_DIR"))
    assert job.spec.template.spec.restart_policy == "Never"


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        RetrainLauncher(mode="bogus", namespace="ns", image="img")
