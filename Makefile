VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

.DEFAULT_GOAL := help

.PHONY: help install install-ci lint test audit run run-multi-report run-integrated doctor clean

help:
	@printf '%s\n' \
		'install          Create $(VENV) and install the package from requirements/dev.lock' \
		'install-ci       Create $(VENV) and install the CI/test environment from requirements/dev.lock' \
		'lint             Run ruff across src, scripts, and CI-owned tests' \
		'test             Run the full pytest suite' \
		'audit            Run secret scan and dependency audit locally' \
		'run              Run the single-asset mission workflow' \
		'run-multi-report Run the multi-asset detection + reporting workflow' \
		'run-integrated   Run the integrated MoE + news orchestrator' \
		'doctor           Print a runtime environment report' \
		'clean            Remove local build and test caches'

$(PYTHON):
	python3 -m venv $(VENV)

install: $(PYTHON)
	PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install --upgrade pip
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install -r requirements/dev.lock
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install -e . --no-deps

install-ci: $(PYTHON)
	PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install --upgrade pip
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install -r requirements/dev.lock
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install -e . --no-deps

lint: install-ci
	PYTHONPATH=src $(RUFF) check src scripts tests .github/tests

test: install-ci
	PYTHONPATH=src MPLBACKEND=Agg $(PYTEST) tests .github/tests

audit: install-ci
	PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install detect-secrets==1.5.0 pip-audit==2.9.0
	git ls-files -z | xargs -0 detect-secrets-hook --exclude-files 'skills/openclaw-moe-orchestrator/templates/auth-profiles.json'
	$(PYTHON) -m pip_audit -r requirements/production.lock

run: install
	$(VENV)/bin/openclaw-moe run-mission

run-multi-report: install
	$(VENV)/bin/openclaw-moe run-multi-report

run-integrated: install
	$(VENV)/bin/openclaw-moe run-integrated

doctor: install
	$(VENV)/bin/openclaw-moe doctor

clean:
	rm -rf .pytest_cache .ruff_cache build dist .eggs *.egg-info
