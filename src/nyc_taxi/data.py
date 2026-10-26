"""Download, read and clean NYC yellow taxi trip records (TLC parquet)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from nyc_taxi.config import DATA_URL

RAW_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
MIN_ZONE, MAX_ZONE = 1, 263
MIN_DURATION_MIN, MAX_DURATION_MIN = 1.0, 180.0


def raw_path(month: str, data_dir: str) -> Path:
    """Local path of the raw parquet for a month (YYYY-MM)."""
    return Path(data_dir) / "raw" / f"yellow_tripdata_{month}.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_manifest(manifest: Path, month: str, entry: dict) -> None:
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data[month] = entry
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True))


def download_month(month: str, data_dir: str) -> Path:
    """Download the raw parquet for a month unless it already exists; update data/manifest.json."""
    path = raw_path(month, data_dir)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = DATA_URL.format(month=month)
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                if chunk:
                    fh.write(chunk)
    tmp.replace(path)
    entry = {"url": url, "sha256": _sha256(path), "bytes": path.stat().st_size}
    _update_manifest(Path(data_dir) / "manifest.json", month, entry)
    return path


def _request_ids(df: pd.DataFrame, positions: pd.Series) -> list[str]:
    """Stable 16-char sha1 id per trip from pickup, dropoff, zones and original row position."""
    ids = []
    for pickup, dropoff, pu, do, pos in zip(
        df["tpep_pickup_datetime"],
        df["tpep_dropoff_datetime"],
        df["PULocationID"],
        df["DOLocationID"],
        positions,
        strict=True,
    ):
        raw = f"{pickup.isoformat()}|{dropoff.isoformat()}|{pu}|{do}|{pos}"
        ids.append(hashlib.sha1(raw.encode()).hexdigest()[:16])
    return ids


def clean(df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Add duration_min and request_id; keep only plausible, in-month, known-zone trips."""
    out = df.copy()
    out["tpep_pickup_datetime"] = pd.to_datetime(out["tpep_pickup_datetime"], errors="coerce")
    out["tpep_dropoff_datetime"] = pd.to_datetime(out["tpep_dropoff_datetime"], errors="coerce")
    out["PULocationID"] = pd.to_numeric(out["PULocationID"], errors="coerce")
    out["DOLocationID"] = pd.to_numeric(out["DOLocationID"], errors="coerce")
    out["duration_min"] = (
        out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    out["_pos"] = range(len(out))

    start = pd.Timestamp(f"{month}-01")
    end = start + pd.DateOffset(months=1)
    keep = (
        out["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)
        & out["PULocationID"].between(MIN_ZONE, MAX_ZONE)
        & out["DOLocationID"].between(MIN_ZONE, MAX_ZONE)
        & out["tpep_pickup_datetime"].ge(start)
        & out["tpep_pickup_datetime"].lt(end)
    )
    out = out[keep].copy()
    out = out.astype({"PULocationID": "int32", "DOLocationID": "int32"})
    out["request_id"] = _request_ids(out, out["_pos"])
    return out.drop(columns="_pos").reset_index(drop=True)


def _sample(df: pd.DataFrame, sample_rows: int | None, seed: int) -> pd.DataFrame:
    if sample_rows is None or len(df) <= sample_rows:
        return df
    sampled = df.sample(n=sample_rows, random_state=seed)
    return sampled.sort_values("tpep_pickup_datetime").reset_index(drop=True)


def load_file(
    path: str | Path, month: str, sample_rows: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Read a raw TLC parquet file (only the needed columns), clean it and optionally sample."""
    raw = pd.read_parquet(path, columns=RAW_COLUMNS)
    return _sample(clean(raw, month), sample_rows, seed)


def load_month(
    month: str, data_dir: str, sample_rows: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Download (if needed) and load one month of cleaned, optionally sampled trips."""
    path = download_month(month, data_dir)
    return load_file(path, month, sample_rows=sample_rows, seed=seed)


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()
