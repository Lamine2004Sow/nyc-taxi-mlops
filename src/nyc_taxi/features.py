"""Feature engineering shared by training, offline replay and the API (single source of truth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nyc_taxi.zones import load_centroids

FEATURES = ["PULocationID", "DOLocationID", "hour", "dow", "is_weekend", "centroid_km", "same_zone"]
CATEGORICAL = ["PULocationID", "DOLocationID"]
TARGET = "duration_min"
FORBIDDEN = [
    "trip_distance",
    "fare_amount",
    "total_amount",
    "tpep_dropoff_datetime",
    "duration_min",
]
ZONE_IDS = list(range(1, 264))
KM_PER_FT = 0.0003048


def build_features(df: pd.DataFrame, centroids: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the model feature matrix (columns exactly FEATURES, same index as df)."""
    if centroids is None:
        centroids = load_centroids()
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    pu = df["PULocationID"].astype("int64").to_numpy()
    do = df["DOLocationID"].astype("int64").to_numpy()

    pu_x = centroids["x_ft"].reindex(pu).to_numpy(dtype=float)
    pu_y = centroids["y_ft"].reindex(pu).to_numpy(dtype=float)
    do_x = centroids["x_ft"].reindex(do).to_numpy(dtype=float)
    do_y = centroids["y_ft"].reindex(do).to_numpy(dtype=float)
    centroid_km = np.hypot(pu_x - do_x, pu_y - do_y) * KM_PER_FT

    dow = pickup.dt.dayofweek.to_numpy()
    out = pd.DataFrame(
        {
            "PULocationID": pd.Categorical(pu, categories=ZONE_IDS),
            "DOLocationID": pd.Categorical(do, categories=ZONE_IDS),
            "hour": pickup.dt.hour.to_numpy(),
            "dow": dow,
            "is_weekend": (dow >= 5).astype(int),
            "centroid_km": centroid_km,
            "same_zone": (pu == do).astype(int),
        },
        index=df.index,
    )
    return out[FEATURES]
