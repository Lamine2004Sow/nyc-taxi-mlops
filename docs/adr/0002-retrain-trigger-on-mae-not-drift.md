# ADR 0002: Retrain trigger on MAE, not drift

Status: Accepted. Date: 2026-10-26.

## Context

Retraining is often triggered by data drift because labels arrive late or never. Here the TLC data
includes the true trip duration, so the error is measurable on every replayed month. Evidently's
drift share fires almost every month because of seasonality (weekday, hour, the March/April 2020
shock), even when the model is accurate.

## Decision

- The trigger is the model's own error: `model_mae_latest > model_ref_mae * 1.15`
  (`ModelMaeDegraded`, in `infra/k8s/components/monitoring/monitoring.yaml`).
- The replay joins ground truth by `request_id` and posts MAE to the monitor (`POST /evaluations`).
- Evidently drift is still computed and stored (`reports/drift/{month}.html`, gauge
  `drift_share_latest`). It is a diagnostic for investigating a degradation, not a trigger.

## Consequences

- Retrains happen only when the model is measurably worse, so the alert stays meaningful.
- The design needs labels. With late labels it would need a delayed evaluation job, which is out of scope.
- Drift can warn earlier, but nothing acts on it. This is accepted: drift fires too often to be actionable.
- The 15% factor (`DEGRADATION_FACTOR`) is a fixed threshold, not a statistical test.
