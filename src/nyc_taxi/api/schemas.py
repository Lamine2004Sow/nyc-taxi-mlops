"""Request and response schemas for the prediction API."""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

MAX_TRIPS = 5000
MIN_ZONE = 1
MAX_ZONE = 265


class TripIn(BaseModel):
    """One trip to score. Pickup time is interpreted as wall-clock (timezone dropped)."""

    request_id: str | None = None
    pickup_datetime: datetime
    PULocationID: int = Field(ge=MIN_ZONE, le=MAX_ZONE)
    DOLocationID: int = Field(ge=MIN_ZONE, le=MAX_ZONE)

    @field_validator("pickup_datetime")
    @classmethod
    def _naive_wall_clock(cls, value: datetime) -> datetime:
        return value.replace(tzinfo=None)


class PredictRequest(BaseModel):
    trips: list[TripIn] = Field(min_length=1, max_length=MAX_TRIPS)


class PredictionOut(BaseModel):
    request_id: str
    duration_min: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[PredictionOut]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    model_loaded: bool
