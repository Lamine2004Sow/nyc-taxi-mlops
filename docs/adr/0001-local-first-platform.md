# ADR 0001: Local-first platform

Status: Accepted. Date: 2026-10-26.

## Context

This is a portfolio MLOps project that must show a closed loop (serve, monitor, retrain, promote)
with Kubernetes, MLflow and Prometheus. It must cost nothing and be reproducible without cloud
accounts.

## Decision

- Everything runs locally: a kind cluster (`nyc-taxi`) for the API, monitor and retrain Jobs, and
  docker compose for MLflow with PostgreSQL and artifacts (`make up`).
- kube-prometheus-stack (Helm) provides Prometheus, Alertmanager and Grafana inside kind.
- GitHub Actions does CI only: lint, tests, the model gate, GHCR image push on `main`, kind smoke test.
- The image reaches the cluster through the local registry `localhost:5001`.

## Consequences

- Zero recurring cost. Anyone can reproduce the loop with `make tools` and `make demo`.
- Single-node failure modes (stale MLflow endpoints, inotify limits) are real and covered in the runbook.
- CI does not run the full replay (network and minutes of compute). The kind smoke test checks that
  the image serves predictions, not that the loop converges.
- A cloud move means replacing NodePorts and MLflow Endpoints with real Services and an ingress.
