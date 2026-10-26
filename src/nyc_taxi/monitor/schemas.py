"""Request schemas for the monitor service."""

from pydantic import BaseModel, ConfigDict, Field


class Evaluation(BaseModel):
    """One replayed month, posted by the replay pipeline."""

    replay_month: str = Field(pattern=r"^\d{4}-\d{2}$")
    model_version: str
    mae: float
    baseline_mae: float
    ref_mae: float
    drift_share: float
    n: int = Field(ge=0)


class AlertItem(BaseModel):
    """One alert from an Alertmanager v4 webhook payload."""

    model_config = ConfigDict(extra="ignore")

    status: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    fingerprint: str | None = None


class AlertmanagerPayload(BaseModel):
    """Alertmanager webhook body (version 4)."""

    model_config = ConfigDict(extra="ignore")

    version: str | None = None
    status: str | None = None
    alerts: list[AlertItem] = Field(default_factory=list)
