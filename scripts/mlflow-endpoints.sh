#!/usr/bin/env bash
# Print a Service (no selector) + Endpoints pair that points the in-cluster name `mlflow`
# at the MLflow container's IP on the docker `kind` network.
# Usage: scripts/mlflow-endpoints.sh | kubectl apply -f -
# Re-run after the mlflow container is recreated (its IP can change).
set -o errexit -o nounset -o pipefail

readonly NAMESPACE="${NAMESPACE:-nyc-taxi}"
readonly CONTAINER="${MLFLOW_CONTAINER:-nyc-taxi-mlflow}"
readonly MLFLOW_PORT=5000

ip="$(docker inspect -f '{{(index .NetworkSettings.Networks "kind").IPAddress}}' "${CONTAINER}")"
if [[ -z "${ip}" ]]; then
  echo "error: ${CONTAINER} has no IP on the kind network (is it running? run make mlflow-up)" >&2
  exit 1
fi

cat <<EOF
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
spec:
  ports:
    - name: http
      port: ${MLFLOW_PORT}
      targetPort: ${MLFLOW_PORT}
---
apiVersion: v1
kind: Endpoints
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
subsets:
  - addresses:
      - ip: ${ip}
    ports:
      - name: http
        port: ${MLFLOW_PORT}
EOF
