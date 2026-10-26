"""Non-ML reference model: median duration per (pickup zone, dropoff zone, hour)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    return pd.DataFrame(
        {
            "pu": df["PULocationID"].astype("int64").to_numpy(),
            "do": df["DOLocationID"].astype("int64").to_numpy(),
            "hour": pickup.dt.hour.to_numpy(),
        },
        index=df.index,
    )


class MedianBaseline:
    """Median duration with fallbacks: (PU, DO, hour) -> (PU, DO) -> global median."""

    def __init__(self) -> None:
        self._global: float = float("nan")
        self._pair_hour: pd.Series | None = None
        self._pair: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> MedianBaseline:
        """Learn medians from a training frame with a duration_min column."""
        keys = _keys(df)
        frame = keys.assign(y=df["duration_min"].to_numpy())
        self._global = float(frame["y"].median())
        self._pair_hour = frame.groupby(["pu", "do", "hour"])["y"].median()
        self._pair = frame.groupby(["pu", "do"])["y"].median()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict durations (minutes) for each row of df."""
        if self._pair_hour is None or self._pair is None:
            raise RuntimeError("MedianBaseline is not fitted")
        keys = _keys(df)
        idx_hour = pd.MultiIndex.from_arrays([keys["pu"], keys["do"], keys["hour"]])
        idx_pair = pd.MultiIndex.from_arrays([keys["pu"], keys["do"]])
        pred = self._pair_hour.reindex(idx_hour).to_numpy(dtype=float)
        fallback = self._pair.reindex(idx_pair).to_numpy(dtype=float)
        pred = np.where(np.isnan(pred), fallback, pred)
        return np.where(np.isnan(pred), self._global, pred)

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        if self._pair_hour is None or self._pair is None:
            raise RuntimeError("MedianBaseline is not fitted")
        return {
            "global": self._global,
            "pair_hour": [
                [int(a), int(b), int(c), float(v)] for (a, b, c), v in self._pair_hour.items()
            ],
            "pair": [[int(a), int(b), float(v)] for (a, b), v in self._pair.items()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MedianBaseline:
        """Rebuild a fitted baseline from to_dict() output."""
        model = cls()
        model._global = float(data["global"])
        ph = data["pair_hour"]
        p = data["pair"]
        model._pair_hour = pd.Series(
            [row[3] for row in ph],
            index=pd.MultiIndex.from_tuples(
                [tuple(row[:3]) for row in ph], names=["pu", "do", "hour"]
            ),
            dtype=float,
        )
        model._pair = pd.Series(
            [row[2] for row in p],
            index=pd.MultiIndex.from_tuples([tuple(row[:2]) for row in p], names=["pu", "do"]),
            dtype=float,
        )
        return model
