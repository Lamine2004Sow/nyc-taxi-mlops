# Service level objectives

Four objectives cover the serving path and the model. Each has an SLI (a PromQL expression over metrics
the services already export) and the mechanism that enforces it. Where enforcement is weaker than the
objective, the gap is stated.

| # | Objective | Enforced by |
|---|---|---|
| 1 | `/predict` availability >= 99% over the replay window | Dashboard only (gap, see below) |
| 2 | `/predict` p95 latency < 300 ms | Alert `ApiHighLatencyP95` |
| 3 | Model MAE <= 1.15 x reference MAE | Alert `ModelMaeDegraded` (triggers retrain) |
| 4 | Model beats the median baseline | CI gate `evaluate-ci` (job `model-gate`) |

Metric names are those exported by `src/nyc_taxi/api/metrics.py` and `src/nyc_taxi/monitor/metrics.py`.
Prometheus scrapes both services through ServiceMonitors (`job="api"`, `job="monitor"`, every 15 s).

**Label note:** the API's `endpoint` label (the route, such as `/predict`) would collide with the
`endpoint` target label set by prometheus-operator; the ServiceMonitors use `honorLabels: true` so
the route label is kept as `endpoint`.

---

## 1. Availability of /predict: 99%

**Objective:** at least 99% of `/predict` requests over the replay window do not fail with a 5xx.
Client errors (422 for invalid trips) count as available.

**Error budget:** 1% of requests in the window.

**SLI** (replace `$window` with the replay length, for example `2h`; the Prometheus retention is 2d):

```promql
sum(increase(http_requests_total{job="api", endpoint="/predict", status!~"5.."}[$window]))
/
sum(increase(http_requests_total{job="api", endpoint="/predict"}[$window]))
```

**Enforcement:** the 99% target has no dedicated alert. `ApiHighErrorRate` fires when more than 5% of
all `job="api"` requests are 5xx for 2 minutes. That is a 95% threshold, so a sustained 2% error rate
passes silently. Closing this gap means adding a rule with `1 - sli > 0.01` over the window, or
tightening `ApiHighErrorRate` to 1%. Neither is in the repo today.

**Where to look:** Grafana panel "API error rate (5m)" (a 5-minute view, not the SLO window).

---

## 2. Latency of /predict: p95 < 300 ms

**Objective:** the 95th percentile of `/predict` latency stays below 300 ms.

**SLI:**

```promql
histogram_quantile(0.95,
  sum by (le) (rate(http_request_duration_seconds_bucket{job="api", endpoint="/predict"}[5m]))
)
```

**Enforcement:** `ApiHighLatencyP95` (warning) fires when
`histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{job="api"}[5m]))) > 0.3`
for 2 minutes.

**Known differences:**

- The alert covers all API routes, while the SLO covers `/predict` only. `/health` and `/metrics` are
  cheap, so in practice they do not move the p95.
- The histogram uses the default buckets, and 0.3 s falls between the 0.25 s and 0.5 s buckets. The
  p95 is interpolated, so values near the threshold are approximate.

---

## 3. Model accuracy: MAE <= 1.15 x reference MAE

**Objective:** the MAE of the served champion on the latest replayed month stays at or below 1.15 times
its own validation MAE (`ref_mae`, stored in the model's `meta.json`). This is also the retrain trigger.

**SLI:**

```promql
model_mae_latest / on() model_ref_mae
```

The objective holds while the value is at most 1.15.

**Enforcement:** `ModelMaeDegraded` (PrometheusRule `nyc-taxi-alerts`):

```promql
(model_mae_latest > on() model_ref_mae * 1.15) * on() group_left(month) replay_month_info
```

It fires with `for: 0m`, one series per replayed month. Alertmanager routes it to the monitor webhook,
which creates a retrain Job. The gauges come from `POST /evaluations`, which the replay sends after
each month, so the SLI is updated per replayed month rather than continuously.

**Why MAE and not drift:** ground truth exists in the TLC data, so the objective is measured directly.
Evidently drift is reported as a diagnostic only (see ADR 0002).

---

## 4. Model beats the median baseline

**Objective:** the model must have a lower MAE than `MedianBaseline` (median duration per PU/DO/hour,
falling back to PU/DO, then to the global median).

**Enforcement (CI, blocking):** `evaluate-ci` trains on the fixture (`tests/fixtures/sample.parquet`),
holds out the last 20% by time, and exits with status 1 if the model MAE is not below the baseline MAE.
The `model-gate` job in `.github/workflows/ci.yml` runs it, and the `docker` job needs it, so a model
that fails the gate is never built or pushed to GHCR.

**Runtime SLI (observation only):**

```promql
baseline_mae_latest - model_mae_latest
```

A positive value means the model beats the baseline on the latest replayed month. No alert covers
this at runtime.

**Known gap:** retrain promotion compares the challenger with the champion on the holdout
(`PROMOTION_MARGIN`, 2%). It does not check the baseline. A retrained model that loses to the
baseline would still be promoted if it beats the current champion. Closing the gap means adding the
baseline comparison to `retrain` in `src/nyc_taxi/pipeline.py`.

---

## Review

- Check the four SLIs in Grafana (dashboard "NYC Taxi MLOps") after each replay.
- Check the 1% error budget after every replay-live run, since the replay is the only load the stack
  sees.
- Operational responses to a breach are in [runbook.md](runbook.md).
