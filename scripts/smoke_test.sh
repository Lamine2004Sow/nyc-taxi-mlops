#!/usr/bin/env bash
# Smoke test for a running api: health, one prediction, prediction metric.
# Usage: scripts/smoke_test.sh [BASE_URL]   (default http://localhost:30080)
set -o errexit -o nounset -o pipefail

readonly BASE_URL="${1:-http://localhost:30080}"
readonly MAX_ATTEMPTS="${SMOKE_ATTEMPTS:-40}"
readonly SLEEP_SECONDS="${SMOKE_SLEEP:-3}"

fail() {
  echo "smoke test FAILED: $*" >&2
  exit 1
}

# 1. Wait for readiness (the model may still be loading).
health=""
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  if health="$(curl -fsS --max-time 5 "${BASE_URL}/health")"; then
    break
  fi
  health=""
  sleep "${SLEEP_SECONDS}"
done
[[ -n "${health}" ]] || fail "/health not reachable at ${BASE_URL} after ${MAX_ATTEMPTS} attempts"
grep -q '"model_loaded":true' <<<"${health// /}" || fail "model not loaded: ${health}"
echo "health ok: ${health}"

# 2. One prediction.
body='{"trips":[{"request_id":"smoke-1","pickup_datetime":"2020-02-03T08:30:00","PULocationID":161,"DOLocationID":236}]}'
response="$(curl -fsS --max-time 10 -H 'Content-Type: application/json' -d "${body}" "${BASE_URL}/predict")" ||
  fail "POST /predict failed"
grep -q '"predictions"' <<<"${response}" || fail "no predictions in response: ${response}"
echo "predict ok: ${response}"

# 3. Prometheus metric incremented by the request above.
metrics="$(curl -fsS --max-time 5 "${BASE_URL}/metrics")" || fail "GET /metrics failed"
grep -q '^predictions_total' <<<"${metrics}" || fail "predictions_total missing from /metrics"
echo "metrics ok"

echo "smoke test PASSED"
