"""NYC taxi zone centroids: built once from the TLC shapefile, then shipped as a CSV."""

from __future__ import annotations

import shutil
import zipfile
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapefile

from nyc_taxi.config import ZONES_URL

CENTROID_COLUMNS = ["LocationID", "x_ft", "y_ft"]
CENTROID_CSV = Path(__file__).parent / "resources" / "zone_centroids.csv"


def _download_zip(data_dir: str) -> Path:
    """Download taxi_zones.zip into data/raw/zones (skipped if already present)."""
    zone_dir = Path(data_dir) / "raw" / "zones"
    zone_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zone_dir / "taxi_zones.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".part")
        with requests.get(ZONES_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(zip_path)
    return zip_path


def _ring_moments(ring: np.ndarray) -> tuple[float, float, float]:
    """Signed area and area-weighted centroid numerators of a closed ring."""
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    cx = ((x + x1) * cross).sum() / 6.0
    cy = ((y + y1) * cross).sum() / 6.0
    return area, cx, cy


def polygon_centroid(points: list, parts: list[int]) -> tuple[float, float]:
    """Area-weighted centroid of a (multi-ring) polygon; holes are subtracted by sign."""
    pts = np.asarray(points, dtype=float)
    bounds = list(parts) + [len(pts)]
    total = sx = sy = 0.0
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        area, cx, cy = _ring_moments(pts[start:end])
        if abs(area) < 1e-12:
            continue
        total += area
        sx += cx
        sy += cy
    if abs(total) < 1e-12:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return float(sx / total), float(sy / total)


def build_centroids(data_dir: str = "data") -> Path:
    """Download the TLC zone shapefile, compute centroids and write the packaged CSV."""
    zip_path = _download_zip(data_dir)
    extract_dir = zip_path.parent / "shp"
    shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    shp_path = next(extract_dir.rglob("*.shp"))

    reader = shapefile.Reader(str(shp_path))
    rows = []
    for record, shape in zip(reader.records(), reader.shapes(), strict=True):
        cx, cy = polygon_centroid(shape.points, shape.parts)
        rows.append((int(record["LocationID"]), cx, cy))

    centroids = (
        pd.DataFrame(rows, columns=CENTROID_COLUMNS)
        .groupby("LocationID", as_index=False)
        .mean()
        .sort_values("LocationID")
    )
    centroids.to_csv(CENTROID_CSV, index=False, float_format="%.3f")
    load_centroids.cache_clear()
    return CENTROID_CSV


@lru_cache(maxsize=1)
def load_centroids() -> pd.DataFrame:
    """Zone centroids indexed by LocationID with columns x_ft, y_ft (feet, NY State Plane)."""
    with resources.files("nyc_taxi.resources").joinpath("zone_centroids.csv").open("rb") as fh:
        df = pd.read_csv(fh)
    return df.set_index("LocationID")
