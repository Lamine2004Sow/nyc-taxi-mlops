#!/usr/bin/env bash
# Install pinned kind, kubectl and helm into ~/.local/bin (Linux amd64).
# Docker is not installed by this script.
set -o errexit -o nounset -o pipefail

readonly KIND_VERSION="v0.27.0"
readonly KUBECTL_VERSION="v1.33.1"
readonly HELM_VERSION="v3.17.3"
readonly BIN_DIR="${HOME}/.local/bin"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: this script only supports Linux x86_64" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
mkdir -p "${BIN_DIR}"

# verify_sha256 FILE EXPECTED_SHA256
verify_sha256() {
  local actual
  actual="$(sha256sum "$1" | awk '{print $1}')"
  if [[ "${actual}" != "$2" ]]; then
    echo "error: checksum mismatch for $1 (expected $2, got ${actual})" >&2
    exit 1
  fi
}

install_kind() {
  echo "installing kind ${KIND_VERSION}"
  curl -fsSL -o "${tmp}/kind" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
  install -m 0755 "${tmp}/kind" "${BIN_DIR}/kind"
}

install_kubectl() {
  echo "installing kubectl ${KUBECTL_VERSION}"
  local base="https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64"
  curl -fsSL -o "${tmp}/kubectl" "${base}/kubectl"
  curl -fsSL -o "${tmp}/kubectl.sha256" "${base}/kubectl.sha256"
  verify_sha256 "${tmp}/kubectl" "$(awk '{print $1}' "${tmp}/kubectl.sha256")"
  install -m 0755 "${tmp}/kubectl" "${BIN_DIR}/kubectl"
}

install_helm() {
  echo "installing helm ${HELM_VERSION}"
  local archive="helm-${HELM_VERSION}-linux-amd64.tar.gz"
  curl -fsSL -o "${tmp}/${archive}" "https://get.helm.sh/${archive}"
  curl -fsSL -o "${tmp}/${archive}.sha256sum" "https://get.helm.sh/${archive}.sha256sum"
  verify_sha256 "${tmp}/${archive}" "$(awk '{print $1}' "${tmp}/${archive}.sha256sum")"
  tar -xzf "${tmp}/${archive}" -C "${tmp}"
  install -m 0755 "${tmp}/linux-amd64/helm" "${BIN_DIR}/helm"
}

install_kind
install_kubectl
install_helm

echo "installed into ${BIN_DIR}:"
"${BIN_DIR}/kind" version
"${BIN_DIR}/kubectl" version --client
"${BIN_DIR}/helm" version --short
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) echo "note: add ${BIN_DIR} to your PATH" ;;
esac
