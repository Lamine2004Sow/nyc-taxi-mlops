# Runbook

Operational procedures for the local nyc-taxi MLOps stack (kind + docker compose). Alert names and
thresholds come from `infra/k8s/components/monitoring/monitoring.yaml`.

## Endpoints

| Component | Address | Notes |
|---|---|---|
| API | http://127.0.0.1:30080 | NodePort 30080 (`infra/kind/cluster.yaml`) |
| Monitor | http://127.0.0.1:30081 | NodePort 30081 |
| Prometheus | http://127.0.0.1:30090 | NodePort 30090, retention 2d |
| Grafana | http://127.0.0.1:30300 | admin / admin (dev only) |
| MLflow UI | http://127.0.0.1:5000 | docker compose, `--allowed-hosts` includes 127.0.0.1 |
| Alertmanager | no host port | `kubectl port-forward` (see below) |

Use `127.0.0.1`, not `localhost` (see [Stack won't start](#stack-wont-start)).

```bash
KCTL="kubectl --context kind-nyc-taxi"

# Quick health pass
curl -fsS http://127.0.0.1:30080/health        # api: model_version, model_loaded
curl -fsS http://127.0.0.1:30081/health        # monitor: retrain_mode (k8s or dry-run)
curl -fsS http://127.0.0.1:5000/health         # MLflow
$KCTL -n nyc-taxi get pods -o wide
$KCTL -n nyc-taxi get jobs -l app=nyc-taxi-retrain

# Alertmanager UI (no NodePort is exposed)
$KCTL -n monitoring get svc | grep alertmanager
$KCTL -n monitoring port-forward svc/<alertmanager-service> 9093:9093   # then http://127.0.0.1:9093
```

---

## ModelMaeDegraded

**Severity:** warning. **Fires when:** `model_mae_latest > on() model_ref_mae * 1.15`, evaluated once
per replayed month (the `month` label comes from `replay_month_info`). `for: 0m`, so it fires as soon
as a replay posts an evaluation above the threshold.

**What it means:** the served champion's MAE on the latest replayed month is more than 15% worse than
its own validation MAE (`model_ref_mae`). Ground truth is available in TLC data, so this is a direct
accuracy signal, not a drift guess (see ADR 0002).

**Automatic path:** Prometheus -> Alertmanager (route `alertname="ModelMaeDegraded"`) -> webhook
`POST /alerts` on the monitor -> monitor creates a Job `nyc-taxi-retrain-<yyyymm>-<timestamp>` that runs
`python -m nyc_taxi retrain --month <month>`. The monitor ignores the alert when the last retrain was
under `RETRAIN_COOLDOWN_SECONDS` ago (120 s in the base manifest) or a retrain Job is still active.

### How to check

```bash
# Current gauges (Prometheus)
curl -sG http://127.0.0.1:30090/api/v1/query --data-urlencode 'query=model_mae_latest / model_ref_mae'
curl -s http://127.0.0.1:30090/api/v1/alerts | python3 -m json.tool | grep -A3 ModelMaeDegraded

# Monitor state
curl -s http://127.0.0.1:30081/metrics | grep -E '^(model_mae_latest|model_ref_mae|baseline_mae_latest|replay_month_info|alerts_received_total|retrain_jobs_triggered_total)'

# Retrain Jobs and their logs
kubectl --context kind-nyc-taxi -n nyc-taxi get jobs -l app=nyc-taxi-retrain
kubectl --context kind-nyc-taxi -n nyc-taxi logs job/<job-name>
```

- **Grafana:** dashboard "NYC Taxi MLOps", panel "Model MAE vs reference and baseline (minutes)"
  (series `model_mae_latest`, `model_ref_mae`, `baseline_mae_latest`, `model_ref_mae * 1.15`).
  Also check "Drift share over time".
- **MLflow UI** (http://127.0.0.1:5000): model `nyc-taxi-duration` -> check which version holds
  `champion` and `challenger`, and compare the validation metrics of the versions.
- **Decide whether the replay month is real degradation:** the replay is a sample
  (`SAMPLE_ROWS`, default 300000). A bad sample is unlikely at that size, but check `n` in the
  evaluation and the per-month table in `reports/replay.md` if you ran the replay on the host.

Interpreting the outcome:

| Observation | Meaning | Action |
|---|---|---|
| Job `Complete`, alias `champion` moved to a new version | Challenger won the holdout by `PROMOTION_MARGIN` (2%) | Verify the API serves it (below). Expected for March/April 2020. |
| Job `Complete`, `challenger` alias moved, `champion` unchanged | Challenger did not beat the champion by 2% | Champion stays. Investigate features and drift, or retrain manually with a different month. |
| Job `Failed` | Usually: CloudFront download blocked, MLflow unreachable (stale endpoint), or a bad month | `kubectl logs job/<name>`. Stale endpoint: see [Stack won't start](#stack-wont-start). |
| No Job created | Cooldown, active Job, or no evaluation stored yet | Re-check the monitor metrics and the Job list. Wait `RETRAIN_COOLDOWN_SECONDS`. |

Confirm the API serves the promoted version (the poll runs every `MODEL_POLL_SECONDS`, 30 s locally):

```bash
curl -s http://127.0.0.1:30080/health
curl -s http://127.0.0.1:30080/metrics | grep -E '^(model_info|model_reloads_total)'
```

### Mitigation

1. If the latest retrain promoted a worse model, or the champion is clearly worse than the previous
   version, follow [Rollback the model](#rollback-the-model).
2. To retrain by hand from the host (writes `reports/promotions.jsonl`):
   `make retrain RETRAIN_MONTH=2020-03` (needs the MLflow container up).
3. During planned maintenance, silence the alert in Alertmanager instead of letting retrains run.

**Note:** the alert clears only after the next replay posts a new evaluation, because the gauges are
set by `POST /evaluations`. A rollback does not change `model_ref_mae` until the next replay.

---

## ApiHighErrorRate

**Severity:** critical. **Fires when:** more than 5% of `job="api"` requests return 5xx over 5 minutes,
for 2 minutes. 4xx (for example 422 on invalid trips) does not count.

### How to check

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30080/health   # 503 means no model loaded
curl -s http://127.0.0.1:30080/metrics | grep 'http_requests_total{' | grep 'status="5'
kubectl --context kind-nyc-taxi -n nyc-taxi get pods -l app.kubernetes.io/name=api -o wide
kubectl --context kind-nyc-taxi -n nyc-taxi logs deploy/api --tail=200
kubectl --context kind-nyc-taxi -n nyc-taxi rollout status deploy/api --timeout=60s
```

- **Grafana:** "API error rate (5m)" and "API request rate by endpoint".
- **Prometheus:** `sum(rate(http_requests_total{job="api",status=~"5.."}[5m])) / sum(rate(http_requests_total{job="api"}[5m]))`.

Typical causes:

- **503 on `/health` and `/predict`:** no model in memory. The alias could not be loaded at startup
  (MLflow unreachable or stale endpoint) and there is no fallback model in the image. Check the
  MLflow endpoint (below), then restart the pods.
- **Crash-looping pods:** `kubectl describe pod` and `logs` show the traceback.
- **5xx right after a promotion:** a model that does not match the feature code. Roll back.

### Mitigation

1. If the errors started after a promotion, [roll back the model](#rollback-the-model).
2. If the MLflow Endpoints point at an old IP, run `make deploy` (see [Stack won't start](#stack-wont-start)).
3. Otherwise restart the API and watch the rollout:
   `kubectl --context kind-nyc-taxi -n nyc-taxi rollout restart deploy/api`.

---

## ApiHighLatencyP95

**Severity:** warning. **Fires when:** the p95 of `http_request_duration_seconds` (all `job="api"`
routes) is above 0.3 s for 2 minutes.

### How to check

```bash
curl -sG http://127.0.0.1:30090/api/v1/query --data-urlencode \
  'query=histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{job="api"}[5m])))'
kubectl --context kind-nyc-taxi -n nyc-taxi get deploy api -o wide
kubectl --context kind-nyc-taxi -n nyc-taxi describe pod -l app.kubernetes.io/name=api | grep -A4 -E 'Limits|Requests'
```

- **Grafana:** "API p95 latency" and "API request rate by endpoint" (a burst of `/predict` is the usual
  cause).
- Check whether a replay is running: a replay sends batches of 1000 trips per request.

Typical causes:

- **Large batches:** a `/predict` with up to 5000 trips is one LightGBM call plus one JSON line per
  trip in the prediction log (written under a lock). This is the first suspect for latency.
- **Too few replicas:** the base manifest runs 2 `api` replicas with no CPU limit (`requests: 100m`).
- **Histogram resolution:** the API uses the default buckets, and the nearest ones around 0.3 s are
  0.25 and 0.5 s, so the p95 is interpolated there. Treat small changes as noise.

### Mitigation

1. Scale out: `kubectl --context kind-nyc-taxi -n nyc-taxi scale deploy/api --replicas=3`.
   Note that `make deploy` resets it to the manifest value.
2. Lower the client batch size, or pause the replay until latency recovers.
3. If p95 stays high at low load, look at the prediction log size under `/tmp/predictions` in the pod.

---

## Rollback the model

Versions are immutable in the MLflow registry. A rollback is only a change of the `champion` alias.
The API polls the alias and hot-reloads (within `MODEL_POLL_SECONDS`, 30 s locally).

1. **Find the previous champion.** Host replays and retrains record it in `reports/promotions.jsonl`
   (`previous_champion`, `new_champion`). Retrains run as Jobs inside the cluster do not write to the
   host, so use the MLflow UI in that case (model `nyc-taxi-duration`, version tags and aliases).

   ```bash
   tail -n 1 reports/promotions.jsonl
   ```

2. **Check the current champion:**

   ```bash
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run python -c "import mlflow; print(mlflow.MlflowClient().get_model_version_by_alias('nyc-taxi-duration', 'champion').version)"
   ```

3. **Point `champion` back at the previous version** (replace `<PREVIOUS_VERSION>`):

   ```bash
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000 MLFLOW_DISABLE_AGENT_HINT=1 uv run python -c "import mlflow; mlflow.MlflowClient().set_registered_model_alias('nyc-taxi-duration', 'champion', '<PREVIOUS_VERSION>')"
   ```

4. **Verify the API serves it** within 30 s, on both replicas:

   ```bash
   curl -s http://127.0.0.1:30080/health
   ```

   `model_version` must equal `<PREVIOUS_VERSION>`. Running requests are not interrupted: the swap is
   atomic in `ModelStore`.

5. **Record the rollback** in the team notes (the MLflow alias history keeps the trail). Rolling back
   does not reset `model_ref_mae`, so `ModelMaeDegraded` may keep firing until the next replay.

To roll forward again, set the alias to the newer version with the same command.

---

## Stack won't start

### inotify limits

`kind` or the kubelet fails with `too many open files` or `failed to create fsnotify watcher` once
several pods run (Prometheus and Grafana are enough to trigger it). Raise the limits:

```bash
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=512
# persist
printf 'fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=512\n' | sudo tee /etc/sysctl.d/99-kind.conf
```

Then `make down && make up`.

### docker group

`permission denied while trying to connect to the Docker daemon socket` means the user is not in the
`docker` group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker        # or log out and back in
docker ps
```

### MLflow endpoint IP is stale

The `mlflow` Service and Endpoints in the `nyc-taxi` namespace carry the container IP on the docker
`kind` network (`scripts/mlflow-endpoints.sh`). When the `nyc-taxi-mlflow` container is recreated, the
IP can change and the API and retrain Jobs fail to reach MLflow.

```bash
docker inspect -f '{{(index .NetworkSettings.Networks "kind").IPAddress}}' nyc-taxi-mlflow
kubectl --context kind-nyc-taxi -n nyc-taxi get endpoints mlflow -o jsonpath='{.subsets[0].addresses[0].ip}'
```

If the two IPs differ, run `make deploy`. It re-applies the Endpoints.

### localhost vs 127.0.0.1

Use `127.0.0.1` in every URL and in `MLFLOW_TRACKING_URI`. The kind host ports are bound to
`127.0.0.1` only (`infra/kind/cluster.yaml`), and `localhost` can resolve to `::1` first, which the
bindings do not serve. The Makefile already uses `127.0.0.1` for `MLFLOW_URI`, `API_URL` and
`MONITOR_URL`.

### Other startup errors

- `docker network 'kind' not found`: run `make cluster-up` first. The network is created by `kind`,
  and the MLflow container joins it.
- MLflow rejects the request with an invalid Host header: the caller uses a host name missing from
  `--allowed-hosts` (`docker-compose.yml`). Use `127.0.0.1:5000` or `mlflow:5000`.
- `make push-local` fails: check that the registry container `kind-registry` is running and
  listening on `localhost:5001`.
