"""Retrain Job construction and the cooldown/dedupe launcher used by the alert webhook."""

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kubernetes import client
from kubernetes import config as kube_config
from kubernetes.config import ConfigException

from nyc_taxi.monitor.metrics import MonitorMetrics, build_monitor_metrics

log = logging.getLogger(__name__)

APP_LABEL = "nyc-taxi-retrain"
ALERTNAME = "ModelMaeDegraded"
TTL_SECONDS_AFTER_FINISHED = 3600
MODE_K8S = "k8s"
MODE_DRY_RUN = "dry-run"
_MAX_DECISIONS = 100


def build_retrain_job(
    month: str,
    namespace: str,
    image: str,
    mlflow_uri: str | None = None,
    now: float | None = None,
) -> client.V1Job:
    """Build the k8s Job that runs `python -m nyc_taxi retrain --month <month>`."""
    stamp = int(now if now is not None else time.time())
    name = f"nyc-taxi-retrain-{month.replace('-', '')}-{stamp}"
    env = [client.V1EnvVar(name="MLFLOW_TRACKING_URI", value=mlflow_uri)] if mlflow_uri else None
    container = client.V1Container(
        name="retrain",
        image=image,
        command=["python", "-m", "nyc_taxi", "retrain", "--month", month],
        env=env,
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": APP_LABEL}),
        spec=client.V1PodSpec(restart_policy="Never", containers=[container]),
    )
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=0,
        ttl_seconds_after_finished=TTL_SECONDS_AFTER_FINISHED,
    )
    return client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=name, namespace=namespace, labels={"app": APP_LABEL}),
        spec=spec,
    )


def _load_kube_config() -> None:
    try:
        kube_config.load_incluster_config()
    except ConfigException:
        kube_config.load_kube_config()


@dataclass(frozen=True)
class RetrainDecision:
    """Outcome of one alert: created, dry-run or skipped (with reason)."""

    action: str
    reason: str
    month: str
    job_name: str | None = None


class RetrainLauncher:
    """Creates retrain Jobs on ModelMaeDegraded alerts, with cooldown and active-job dedupe."""

    def __init__(
        self,
        mode: str,
        namespace: str,
        image: str,
        cooldown_seconds: float = 600,
        mlflow_uri: str | None = None,
        batch_api: Any | None = None,
        metrics: MonitorMetrics | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in (MODE_K8S, MODE_DRY_RUN):
            raise ValueError(f"RETRAIN_MODE must be 'k8s' or 'dry-run', got {mode!r}")
        self.mode = mode
        self.namespace = namespace
        self.image = image
        self.cooldown_seconds = cooldown_seconds
        self.mlflow_uri = mlflow_uri
        self.metrics = metrics or build_monitor_metrics()
        self._batch_api = batch_api
        self._clock = clock
        self._lock = threading.Lock()
        self._last_trigger: float | None = None
        self._decisions: list[RetrainDecision] = []

    @classmethod
    def from_env(cls, metrics: MonitorMetrics | None = None) -> "RetrainLauncher":
        """Build from RETRAIN_* and POD_NAMESPACE env (default mode: k8s if in-cluster)."""
        in_cluster = "KUBERNETES_SERVICE_HOST" in os.environ
        mode = os.getenv("RETRAIN_MODE") or (MODE_K8S if in_cluster else MODE_DRY_RUN)
        return cls(
            mode=mode,
            namespace=os.getenv("POD_NAMESPACE", "nyc-taxi"),
            image=os.getenv("RETRAIN_IMAGE", "nyc-taxi:latest"),
            cooldown_seconds=float(os.getenv("RETRAIN_COOLDOWN_SECONDS", "600")),
            mlflow_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
            metrics=metrics,
        )

    @property
    def decisions(self) -> list[RetrainDecision]:
        """Most recent decisions (bounded)."""
        with self._lock:
            return list(self._decisions)

    def _api(self) -> Any:
        if self._batch_api is None:
            _load_kube_config()
            self._batch_api = client.BatchV1Api()
        return self._batch_api

    def _has_active_job(self) -> bool:
        resp = self._api().list_namespaced_job(
            namespace=self.namespace, label_selector=f"app={APP_LABEL}"
        )
        return any(job.status is not None and bool(job.status.active) for job in resp.items)

    def _record(self, decision: RetrainDecision) -> RetrainDecision:
        self._decisions = [*self._decisions[-(_MAX_DECISIONS - 1) :], decision]
        return decision

    def handle_alert(self, month: str) -> RetrainDecision:
        """Create (or record, in dry-run) a retrain for `month` unless deduped."""
        with self._lock:
            now = self._clock()
            if self._last_trigger is not None and now - self._last_trigger < self.cooldown_seconds:
                return self._record(RetrainDecision("skipped", "cooldown", month))
            if self.mode == MODE_DRY_RUN:
                self._last_trigger = now
                log.info("dry-run: would retrain %s", month)
                return self._record(RetrainDecision("dry-run", "dry-run mode", month))
            if self._has_active_job():
                return self._record(RetrainDecision("skipped", "active job", month))
            job = build_retrain_job(
                month, self.namespace, self.image, mlflow_uri=self.mlflow_uri, now=now
            )
            self._api().create_namespaced_job(namespace=self.namespace, body=job)
            self._last_trigger = now
            self.metrics.retrain_jobs_triggered.inc()
            name = job.metadata.name
            log.info("created retrain job %s", name)
            return self._record(RetrainDecision("created", "alert", month, job_name=name))
