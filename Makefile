VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

INSTALL_GROUP ?= dev

.DEFAULT_GOAL := help

.PHONY: help install install-ci lint test run run-multi-report run-integrated doctor clean

help:
	@printf '%s\n' \
		'install          Create $(VENV) and install the package with runtime + dev dependencies' \
		'install-ci       Create $(VENV) and install only CI tooling' \
		'lint             Run ruff across src, scripts, and CI-owned tests' \
		'test             Run the full pytest suite' \
		'run              Run the single-asset mission workflow' \
		'run-multi-report Run the multi-asset detection + reporting workflow' \
		'run-integrated   Run the integrated MoE + news orchestrator' \
		'doctor           Print a runtime environment report' \
		'clean            Remove local build and test caches'

$(PYTHON):
	python3 -m venv $(VENV)

install: $(PYTHON)
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install --upgrade pip
	PIP_DISABLE_PIP_VERSION_CHECK=1 DS_BUILD_OPS=0 $(PIP) install -e ".[$(INSTALL_GROUP)]"

install-ci: $(PYTHON)
	PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install --upgrade pip
	PIP_DISABLE_PIP_VERSION_CHECK=1 $(PIP) install pytest==9.0.2 ruff==0.15.5

lint: install-ci
	PYTHONPATH=src $(RUFF) check src scripts tests .github/tests

test: install-ci
	PYTHONPATH=src MPLBACKEND=Agg $(PYTEST) tests .github/tests

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
