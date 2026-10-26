#!/usr/bin/env bash
# Create the kind cluster `nyc-taxi` together with a local registry on localhost:5001.
# Based on the kind "local registry" recipe (https://kind.sigs.k8s.io/docs/user/local-registry/).
set -o errexit -o nounset -o pipefail

readonly CLUSTER_NAME="nyc-taxi"
readonly KUBE_CONTEXT="kind-${CLUSTER_NAME}"
readonly REG_NAME="kind-registry"
readonly REG_PORT="5001"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for tool in docker kind kubectl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} not found in PATH (run scripts/install-tools.sh)" >&2
    exit 1
  fi
done

# 1. Registry container (bound to 127.0.0.1 only).
if [[ "$(docker inspect -f '{{.State.Running}}' "${REG_NAME}" 2>/dev/null || true)" != "true" ]]; then
  docker run -d --restart=always -p "127.0.0.1:${REG_PORT}:5000" \
    --network bridge --name "${REG_NAME}" registry:2
fi

# 2. Cluster.
if ! kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}" --config "${REPO_ROOT}/infra/kind/cluster.yaml"
fi

# 3. Make the registry reachable from the kind network and teach each node about it.
docker network connect kind "${REG_NAME}" 2>/dev/null || true
for node in $(kind get nodes --name "${CLUSTER_NAME}"); do
  docker exec "${node}" mkdir -p "/etc/containerd/certs.d/localhost:${REG_PORT}"
  printf '[host."http://%s:5000"]\n' "${REG_NAME}" |
    docker exec -i "${node}" cp /dev/stdin "/etc/containerd/certs.d/localhost:${REG_PORT}/hosts.toml"
done

# 4. Document the registry for tooling (kind convention).
kubectl --context "${KUBE_CONTEXT}" apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REG_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

echo "cluster ${CLUSTER_NAME} ready, registry at localhost:${REG_PORT}"
