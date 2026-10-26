# ADR 0004: No ONNX export

Status: Accepted. Date: 2026-10-26.

## Context

The model is a LightGBM booster. ONNX would allow ONNX Runtime in the API, but the model uses
categorical features (`PULocationID`, `DOLocationID`) with fixed categories `range(1, 264)`. The
converter must reproduce them exactly for training and serving to match.

## Decision

- Serve the LightGBM booster directly through `nyc_taxi.model.predict`. The replay, the API and the
  tests share that code path.
- Artifacts stay as `model.txt` plus `meta.json` (`save_model_dir` / `load_model_dir`).

## Consequences

- No conversion step, no second categorical implementation, and no train/serve skew from a converter.
- The API image needs the LightGBM runtime and `libgomp1`, a small cost already paid.
- No ONNX Runtime latency gain was measured, and no latency problem has been observed against the
  p95 objective (300 ms, see `docs/slo.md`). The optimization is not justified yet.
- Revisit if the model moves to a runtime without a LightGBM binding, or if measured latency becomes
  the bottleneck.
