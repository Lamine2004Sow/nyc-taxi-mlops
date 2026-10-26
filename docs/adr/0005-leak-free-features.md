# ADR 0005: Leak-free features

Status: Accepted. Date: 2026-10-26.

## Context

The target is `duration_min`. Several TLC columns are known only after the ride: `trip_distance` is
measured by the meter, and `fare_amount`, `total_amount` and the drop-off time follow from the trip.
They would inflate offline accuracy and be useless at pickup time, when the API is called.

## Decision

- Features are known at pickup: PU and DO zone IDs, hour, day of week, weekend flag, the zone centroid
  distance (`centroid_km`) and `same_zone`.
- `src/nyc_taxi/features.py` defines `FEATURES` and `FORBIDDEN` (`trip_distance`, `fare_amount`,
  `total_amount`, `tpep_dropoff_datetime`, `duration_min`).
- `tests/test_features.py` asserts the two lists do not overlap, so a leak fails CI.
- Training, replay and the API all call `build_features`.

## Consequences

- Offline MAE is an honest estimate of what the API achieves at pickup time.
- Centroid distance is a proxy for the unknown travelled distance. This costs some accuracy.
- The test protects the list only while it is maintained: check new columns against the pickup rule.
- The committed centroid CSV means the API needs no shapefile at runtime.
