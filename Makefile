# nyc-taxi-mlops entry points. Run `make help` for the list.
#
# Ordering: `make up` needs no prior step. The docker network `kind` is created by
# `kind create cluster`, so cluster-up always runs before mlflow-up (chained in `up`).
# Targets are serial on purpose (.NOTPARALLEL).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:

export PATH := $(HOME)/.local/bin:$(PATH)
export MLFLOW_DISABLE_AGENT_HINT := 1

KIND_CLUSTER := nyc-taxi
KUBE_CONTEXT := kind-$(KIND_CLUSTER)
KUBECTL := kubectl --context $(KUBE_CONTEXT)
HELM := helm --kube-context $(KUBE_CONTEXT)

IMAGE_LOCAL := localhost:5001/nyc-taxi-mlops:dev
MLFLOW_URI := http://127.0.0.1:5000
API_URL := http://127.0.0.1:30080
MONITOR_URL := http://127.0.0.1:30081
# Host-side commands talk to MLflow on localhost; this prefix sets it for one command.
TRACK := MLFLOW_TRACKING_URI=$(MLFLOW_URI)

MONTHS := 2020-01 2020-02 2020-03 2020-04 2020-05 2020-06 2020-07
TRAIN_MONTH ?= 2020-01
RETRAIN_MONTH ?= 2020-03
# Optional pin for the kube-prometheus-stack chart, e.g. make monitoring-up KPS_CHART_VERSION=X.Y.Z
KPS_CHART_VERSION ?=

.PHONY: help install tools lint format test evaluate-ci \
	cluster-up cluster-down mlflow-up mlflow-down monitoring-up \
	build push-local deploy up down \
	data zones train replay replay-live retrain demo smoke clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | awk -F'\t' '{printf "  %-14s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- dev loop
install: ## uv sync (dev dependencies included)
	uv sync

tools: ## Install pinned kind, kubectl, helm into ~/.local/bin
	scripts/install-tools.sh

lint: ## ruff check and format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

test: ## pytest with coverage
	uv run pytest --cov=nyc_taxi --cov-report=xml --cov-report=term

evaluate-ci: ## Train and gate on the fixture (writes models/champion for the Docker build)
	uv run python -m nyc_taxi evaluate-ci --fixture tests/fixtures/sample.parquet

# ---------------------------------------------------------------- infrastructure
cluster-up: ## kind cluster nyc-taxi + local registry on localhost:5001
	scripts/kind-with-registry.sh

cluster-down: ## Delete the kind cluster (registry container is kept)
	kind delete cluster --name $(KIND_CLUSTER)

mlflow-up: ## MLflow + Postgres (docker compose); requires cluster-up for the kind network
	@docker network inspect kind >/dev/null 2>&1 || { echo "docker network 'kind' not found: run make cluster-up first" >&2; exit 1; }
	docker compose up -d --build

mlflow-down: ## Stop MLflow + Postgres (volumes are kept)
	docker compose down

monitoring-up: ## kube-prometheus-stack (Prometheus, Alertmanager, Grafana) + dashboard ConfigMap
	$(HELM) repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
	$(HELM) repo update prometheus-community
	$(HELM) upgrade --install kps prometheus-community/kube-prometheus-stack \
		--namespace monitoring --create-namespace \
		-f infra/helm/kube-prometheus-stack-values.yaml \
		$(if $(KPS_CHART_VERSION),--version $(KPS_CHART_VERSION),) \
		--wait --timeout 10m
	$(KUBECTL) apply -k infra/grafana

build: ## Build the api/monitor/retrain image (creates models/ if missing)
	mkdir -p models
	docker build -t $(IMAGE_LOCAL) .

push-local: ## Push the image to the local registry
	docker push $(IMAGE_LOCAL)

deploy: ## Apply overlays/local and the MLflow endpoints for the kind cluster
	$(KUBECTL) apply -k infra/k8s/overlays/local
	scripts/mlflow-endpoints.sh | $(KUBECTL) apply -f -

up: ## Full local stack: cluster, MLflow, monitoring, image, deploy
up: cluster-up mlflow-up monitoring-up build push-local deploy

down: ## Delete the cluster and stop MLflow
down: cluster-down mlflow-down

# ---------------------------------------------------------------- data and ML
data: ## Download the training and replay months (cached in data/raw)
	uv run python -m nyc_taxi download --months $(MONTHS)

zones: ## Regenerate the committed zone centroids CSV
	uv run python -m nyc_taxi zones

train: data ## Train the champion on TRAIN_MONTH and register it in MLflow
	$(TRACK) uv run python -m nyc_taxi train --month $(TRAIN_MONTH)

replay: ## Offline replay with the local champion (no cluster needed)
	$(TRACK) uv run python -m nyc_taxi replay

replay-live: ## Replay through the cluster api (30080) and monitor (30081)
	$(TRACK) uv run python -m nyc_taxi replay --api-url $(API_URL) --monitor-url $(MONITOR_URL)

retrain: ## Retrain the challenger on RETRAIN_MONTH and gate it against the champion
	$(TRACK) uv run python -m nyc_taxi retrain --month $(RETRAIN_MONTH)

smoke: ## Smoke test the cluster api
	scripts/smoke_test.sh $(API_URL)

demo: ## up, then train, then replay-live through the cluster
demo: up train replay-live

clean: ## Remove local data, models and reports (keeps the cluster and MLflow)
	rm -rf data/raw models predictions reports mlruns mlartifacts
