"""Tests for feature construction and the leakage guard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nyc_taxi.features import CATEGORICAL, FEATURES, FORBIDDEN, KM_PER_FT, build_features


def test_feature_columns_exact_and_ordered(sample_df):
    feats = build_features(sample_df)
    assert list(feats.columns) == FEATURES
    assert feats.index.equals(sample_df.index)


def test_no_forbidden_column_is_a_feature():
    assert not set(FEATURES) & set(FORBIDDEN)
    assert "duration_min" not in FEATURES
    assert set(CATEGORICAL) <= set(FEATURES)


def test_centroid_distance_matches_manual(centroids):
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2020-01-05 09:00")],
            "PULocationID": [1],
            "DOLocationID": [263],
        }
    )
    feats = build_features(df, centroids)
    dx = centroids.loc[1, "x_ft"] - centroids.loc[263, "x_ft"]
    dy = centroids.loc[1, "y_ft"] - centroids.loc[263, "y_ft"]
    expected = np.hypot(dx, dy) * KM_PER_FT
    assert feats["centroid_km"].iloc[0] == np.float64(expected)


def test_same_zone_and_calendar_fields():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2020-01-04 17:30")],  # a Saturday
            "PULocationID": [50],
            "DOLocationID": [50],
        }
    )
    feats = build_features(df)
    row = feats.iloc[0]
    assert row["same_zone"] == 1
    assert row["centroid_km"] == 0.0
    assert row["hour"] == 17
    assert row["dow"] == 5
    assert row["is_weekend"] == 1


def test_unknown_zone_gives_nan_centroid():
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [pd.Timestamp("2020-01-04 17:30")],
            "PULocationID": [264],
            "DOLocationID": [10],
        }
    )
    assert np.isnan(build_features(df)["centroid_km"].iloc[0])


def test_zone_columns_are_categorical_with_fixed_categories(sample_df):
    feats = build_features(sample_df)
    for col in CATEGORICAL:
        assert isinstance(feats[col].dtype, pd.CategoricalDtype)
        assert list(feats[col].cat.categories) == list(range(1, 264))
