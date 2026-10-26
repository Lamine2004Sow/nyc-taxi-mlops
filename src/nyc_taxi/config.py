"""Runtime configuration, read from environment variables once at import time."""

import os
from dataclasses import dataclass

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
MODEL_NAME = "nyc-taxi-duration"
EXPERIMENT_NAME = "nyc-taxi-duration"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

DEFAULT_REPLAY_MONTHS = "2020-02,2020-03,2020-04,2020-05,2020-06,2020-07"


@dataclass(frozen=True)
class Settings:
    """All tunables for the closed MLOps loop."""

    data_dir: str = os.getenv("DATA_DIR", "data")
    reports_dir: str = os.getenv("REPORTS_DIR", "reports")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    train_month: str = os.getenv("TRAIN_MONTH", "2020-01")
    replay_months: tuple[str, ...] = tuple(
        os.getenv("REPLAY_MONTHS", DEFAULT_REPLAY_MONTHS).split(",")
    )
    sample_rows: int = int(os.getenv("SAMPLE_ROWS", "300000"))
    degradation_factor: float = float(os.getenv("DEGRADATION_FACTOR", "1.15"))
    promotion_margin: float = float(os.getenv("PROMOTION_MARGIN", "0.02"))
    seed: int = 42


def get_settings() -> Settings:
    """Return the settings built from the current environment."""
    return Settings()
