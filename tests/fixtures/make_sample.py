"""Generate synthetic yellow-taxi trips in the raw TLC schema (deterministic).

Duration follows the real zone centroid distance, rush-hour effects and noise, so a model
using centroid_km and hour can beat a (PU, DO, hour) median baseline. Run as a script to
regenerate tests/fixtures/sample.parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nyc_taxi.features import KM_PER_FT, ZONE_IDS
from nyc_taxi.zones import load_centroids

RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
FIXTURE_PATH = Path(__file__).parent / "sample.parquet"


def make_trips(
    n: int,
    month: str,
    seed: int = 42,
    shift: float = 1.0,
    junk_frac: float = 0.0,
) -> pd.DataFrame:
    """Synthetic raw trips for one month. `shift` multiplies durations (simulated drift)."""
    rng = np.random.default_rng(seed)
    centroids = load_centroids()
    start = pd.Timestamp(f"{month}-01")
    days_in_month = (start + pd.DateOffset(months=1) - start).days

    day = rng.integers(0, days_in_month, size=n)
    hour_weights = np.array([1.0 + 2.5 * (h in RUSH_HOURS) for h in range(24)])
    hour = rng.choice(24, size=n, p=hour_weights / hour_weights.sum())
    minute = rng.uniform(0, 60, size=n)
    pickup = start + pd.to_timedelta(day * 24 + hour, unit="h") + pd.to_timedelta(minute, unit="m")

    pu = rng.choice(ZONE_IDS, size=n)
    do = rng.choice(ZONE_IDS, size=n)
    x, y = centroids["x_ft"], centroids["y_ft"]
    dist_km = (
        np.hypot(
            x.reindex(pu).to_numpy() - x.reindex(do).to_numpy(),
            y.reindex(pu).to_numpy() - y.reindex(do).to_numpy(),
        )
        * KM_PER_FT
    )

    rush = np.where(np.isin(hour, RUSH_HOURS), 1.3, 1.0)
    night = np.where(hour <= 5, 0.85, 1.0)
    base = (3.0 + 2.4 * dist_km) * rush * night * shift
    noise = rng.normal(0, 1.0, size=n) + 0.06 * base * rng.normal(size=n)
    duration = np.clip(base + noise, 1.2, 175.0)
    dropoff = pickup + pd.to_timedelta(duration, unit="m")

    trip_distance = np.maximum(dist_km * 0.62 * (1 + rng.normal(0, 0.1, size=n)), 0.1)
    fare = 2.5 + 2.6 * trip_distance + rng.normal(0, 1.0, size=n)
    df = pd.DataFrame(
        {
            "VendorID": rng.integers(1, 3, size=n),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": dropoff,
            "passenger_count": rng.integers(1, 4, size=n),
            "trip_distance": trip_distance.round(2),
            "PULocationID": pu.astype("int64"),
            "DOLocationID": do.astype("int64"),
            "fare_amount": fare.round(2),
            "total_amount": (fare * 1.2).round(2),
        }
    )
    if junk_frac > 0:
        df = _inject_junk(df, rng, junk_frac)
    return df.reset_index(drop=True)


def _inject_junk(df: pd.DataFrame, rng: np.random.Generator, frac: float) -> pd.DataFrame:
    """Corrupt a fraction of rows: zero/over-long durations, unknown zone, missing zone."""
    rows = rng.choice(len(df), size=max(1, int(len(df) * frac)), replace=False)
    kinds = rng.integers(0, 4, size=len(rows))
    for row, kind in zip(rows, kinds, strict=True):
        pickup = pd.Timestamp(df.loc[row, "tpep_pickup_datetime"])
        if kind == 0:
            df.loc[row, "tpep_dropoff_datetime"] = pickup
        elif kind == 1:
            df.loc[row, "tpep_dropoff_datetime"] = pickup + pd.Timedelta(hours=5)
        elif kind == 2:
            df.loc[row, "PULocationID"] = 265
        else:
            df.loc[row, "DOLocationID"] = np.nan
    return df


def main() -> None:
    """Write the deterministic 5000-row January 2020 fixture."""
    df = make_trips(5000, "2020-01", seed=42, junk_frac=0.02)
    df.to_parquet(FIXTURE_PATH, index=False)
    print(f"wrote {FIXTURE_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
