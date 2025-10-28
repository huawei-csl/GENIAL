REG ?= ghcr.io/ORG/REPO
CACHE_REF ?= $(REG)/build-cache:main
VENV ?= .venv
PYTHON ?= python3
UV ?= 0

PY := $(VENV)/bin/python
PIP := $(PY) -m pip
PRECOMMIT := $(VENV)/bin/pre-commit
PYTEST := $(VENV)/bin/pytest

.PHONY: help bootstrap test smoke pull-prebuilt dev-up dev-up-heavy build-base build-genial build-flowy build-all clean doctor

help: ## Show this help
	grep -E '^[a-zA-Z_-]:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

bootstrap: ## Create venv and install project with dev extras
	if [ "$(UV)" = "1" ] && command -v uv >/dev/null 2>&1; then \
		uv venv $(VENV); \
		uv pip install -p $(PY) -e ".[dev]"; \
	else \
		$(PYTHON) -m venv $(VENV); \
		$(PIP) install -e ".[dev]"; \
	fi
	$(PRECOMMIT) install >/dev/null

test: ## Run pytest in quiet mode
	$(PYTEST) -q

scripts/smoke.sh: ## Create smoke test helper if missing
	mkdir -p scripts
	[ -f $@ ] || { printf '%s\n' '#!/usr/bin/env bash' \
	'set -euo pipefail' \
	'. "$(dirname "$$0")/../.venv/bin/activate"' \
	'python - <<'"'"'PY'"'"'' \
	'import flowy, genial' \
	'PY' \
	'python -m flowy --help >/dev/null' \
	'python -m genial --help >/dev/null' \
	'echo "smoke test passed"' > $@; chmod x $@; }

smoke: scripts/smoke.sh ## Run light smoke tests for CLI modules
	./scripts/smoke.sh

pull-prebuilt: ## Pull prebuilt images if available
	-docker pull $(REG)/oss_eda_base || true
	-docker pull $(REG)/genial || true
	-docker pull $(REG)/flowy || true

dev-up: pull-prebuilt ## Start docker compose using prebuilt images
	docker compose up -d

dev-up-heavy: ## Force local build and start compose (heavy)
	docker compose --profile heavy up -d

build-base: ## Build oss_eda_base image with cache
	docker buildx build --load \
		--cache-from type=registry,ref=$(CACHE_REF) \
		--cache-to type=registry,ref=$(CACHE_REF),mode=max \
		-t $(REG)/oss_eda_base -f ext/oss_eda_base/Dockerfile \
		--target dev ext/oss_eda_base

build-genial: ## Build genial image with cache
	docker buildx build --load \
		--cache-from type=registry,ref=$(CACHE_REF) \
		--cache-to type=registry,ref=$(CACHE_REF),mode=max \
		-t $(REG)/genial -f .devcontainer/docker/Dockerfile \
		--target dev .

build-flowy: ## Build flowy image with cache
	docker buildx build --load \
		--cache-from type=registry,ref=$(CACHE_REF) \
		--cache-to type=registry,ref=$(CACHE_REF),mode=max \
		-t $(REG)/flowy -f ext/flowy/Dockerfile \
		--target dev ext/flowy

build-all: build-base build-genial build-flowy ## Build all images

clean: ## Remove virtualenv and Python caches
	rm -rf $(VENV) dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} 

doctor: ## Show versions of tools
	$(PYTHON) --version
	command -v uv >/dev/null 2>&1 && uv --version || true
	$(PYTHON) -m pip --version
	docker --version
	docker buildx version
	docker compose version
