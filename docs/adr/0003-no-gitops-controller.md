# ADR 0003: No GitOps controller

Status: Accepted. Date: 2026-10-26.

## Context

ArgoCD or Flux would reconcile `infra/k8s/` from Git. The project has one cluster (kind, on the
developer machine) and one user. Model promotion is not a manifest change: a new version is promoted
by moving the MLflow alias `champion`, which the API polls.

## Decision

- Deploy with `kubectl apply -k` (`make deploy`) and Helm (`make monitoring-up`), driven by Make.
- Promotion uses the MLflow alias. The API reads it at startup and every `MODEL_POLL_SECONDS`, so no
  Git commit is needed to change the served model.
- CI checks the manifests indirectly, through the `kind-smoke` job that applies `overlays/ci` and
  waits for the rollout.

## Consequences

- Fewer moving parts: no controller, no sync loop, no second source of truth.
- Cluster drift from Git is possible. `make deploy` is the reconciliation step.
- The model lifecycle is not in Git history. The MLflow alias history and `reports/promotions.jsonl`
  are the audit trail.
- With several clusters, revisit this decision: ArgoCD would then pay for itself.
