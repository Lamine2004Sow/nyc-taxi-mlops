"""MLflow tracking and model registry helpers (versions + champion/challenger aliases)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# MLflow 3.x marks the file tracking store as deprecated; the project still supports
# file:// URIs for local runs and tests, so opt in explicitly.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from nyc_taxi.config import EXPERIMENT_NAME, MODEL_NAME
from nyc_taxi.model import META_FILE, MODEL_FILE, save_model_dir


def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def log_and_register(
    booster: Any,
    meta: dict[str, Any],
    metrics: dict[str, float],
    params: dict[str, Any],
    tracking_uri: str,
) -> str:
    """Log a run with params/metrics and the model dir, register it; returns the model version."""
    client = _client(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = save_model_dir(booster, Path(tmp) / "model", {**meta, "model_version": None})
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        version = _create_version(client, run_id)
        # Re-log meta.json so the artifact carries the registry version it was given.
        full_meta = {**meta, "model_version": version}
        save_model_dir(booster, model_dir, full_meta)
        for name in (MODEL_FILE, META_FILE):
            client.log_artifact(run_id, str(model_dir / name), artifact_path="model")
    return version


def _create_version(client: MlflowClient, run_id: str) -> str:
    """Register the run's model artifacts as a new version of MODEL_NAME."""
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        client.create_registered_model(MODEL_NAME)
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri.rstrip('/')}/model"
    mv = client.create_model_version(name=MODEL_NAME, source=source, run_id=run_id)
    return str(mv.version)


def set_alias(version: str, alias: str, tracking_uri: str) -> None:
    """Point a registry alias (champion or challenger) at a model version."""
    _client(tracking_uri).set_registered_model_alias(MODEL_NAME, alias, str(version))


def get_alias_version(alias: str, tracking_uri: str) -> str | None:
    """Model version behind an alias, or None if the alias or model does not exist."""
    try:
        mv = _client(tracking_uri).get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None
    return str(mv.version)


def download_alias(alias: str, dst: str | Path, tracking_uri: str) -> Path:
    """Download the artifacts of models:/MODEL_NAME@alias; returns the dir with model.txt."""
    mlflow.set_tracking_uri(tracking_uri)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{alias}", dst_path=str(dst)
    )
    root = Path(local)
    if (root / MODEL_FILE).exists():
        return root
    found = next(root.rglob(MODEL_FILE), None)
    if found is None:
        raise FileNotFoundError(f"no {MODEL_FILE} under alias {alias}")
    return found.parent
