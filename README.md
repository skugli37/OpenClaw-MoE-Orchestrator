# OpenClaw MoE Orchestrator

Production-oriented market analysis and anomaly detection workflows built around DeepSpeed Mixture-of-Experts models and zero-cost external news validation.

The repository now also includes a production OpenClaw integration path backed by Ollama Cloud model rotation, with OpenAI and Anthropic kept out of the active provider path.

## What It Does

- Downloads real crypto market data from Yahoo Finance.
- Builds normalized single-asset and multi-asset feature sets.
- Trains DeepSpeed MoE autoencoders for anomaly detection.
- Produces charts, reports, and run metadata for traceability.
- Runs an integrated orchestrator that combines anomaly scoring with live Google News RSS context.

This repository now separates the production runtime from playground code:

- `src/openclaw_moe_orchestrator/`: packaged production code
- `scripts/`: thin compatibility launchers
- `experiments/`: non-production demos, stress tests, and simulations
- `configs/`: checked-in DeepSpeed configuration
- `data/processed/`: generated processed datasets
- `artifacts/`: generated anomaly results, charts, and metadata
- `docs/`: operator-facing reports and architecture notes

## Production Requirements

- Python `3.12+`
- A working local virtual environment
- `torch` and `deepspeed` installed in that environment
- GPU is preferred; CPU execution is supported but slower

The runtime is fail-fast about malformed datasets and missing configs. It does not use synthetic market data in the production path.

## Install

Use the local virtual environment and install through the committed lock file:

```bash
make install
```

If you are already working inside the repository `.venv`, install only the missing tooling:

```bash
source .venv/bin/activate
make install
```

## Commands

Primary packaged CLI:

```bash
openclaw-moe doctor
openclaw-moe run-mission
openclaw-moe run-multi-report
openclaw-moe run-integrated
openclaw-moe detect-single
openclaw-moe detect-multi
openclaw-moe visualize-multi
openclaw-moe install-openclaw-cloud
openclaw-moe doctor-openclaw-cloud
openclaw-moe list-ollama-models
openclaw-moe sync-ollama-models --dry-run
```

Compatibility launchers remain available:

```bash
python scripts/autonomous_mission.py
python scripts/integrated_orchestrator.py
```

## Validation

Environment report:

```bash
openclaw-moe doctor
```

Lint and tests:

```bash
make lint
make test
make audit
```

OpenClaw + Ollama integration report:

```bash
openclaw-moe doctor-openclaw-cloud
```

Seed the local OpenClaw workspace, auth profiles, and overlay config:

```bash
openclaw-moe install-openclaw-cloud
```

The generated OpenClaw auth profile uses the non-secret marker `ollama-cloud`. The actual Ollama Cloud credential should be configured on the Ollama daemon.

Stage Ollama pulls from the committed manifest:

```bash
openclaw-moe sync-ollama-models --role reasoning --max-models 1 --dry-run
openclaw-moe sync-ollama-models --role general --max-models 1 --dry-run
```

Full setup notes live in `docs/openclaw_local_ollama_integration.md`.

## Locked Dependencies

Committed lock files now define the resolved dependency graph used by local installs, CI, and container builds:

- `requirements/production.lock`
- `requirements/dev.lock`

## Container

The repository now includes a production-oriented [Dockerfile](/home/kugli/OpenClaw-MoE-Orchestrator/Dockerfile) for containerized packaging of the CLI runtime:

```bash
docker build -t openclaw-moe-orchestrator .
docker run --rm openclaw-moe-orchestrator doctor
```

## Outputs

Single-asset mission writes a run bundle under `artifacts/runs/<run-id>/` plus the latest operator report in `docs/mission_report.md`:

- `data/processed/market_data_norm.csv`
- `artifacts/runs/<run-id>/inputs/market_data_norm.csv`
- `artifacts/runs/<run-id>/outputs/anomaly_results.csv`
- `artifacts/runs/<run-id>/outputs/anomaly_chart.png`
- `artifacts/runs/<run-id>/outputs/mission_run_metadata.json`
- `artifacts/runs/<run-id>/outputs/mission_report.md`
- `docs/mission_report.md`

Integrated orchestrator writes a run bundle under `artifacts/runs/<run-id>/`:

- `data/processed/multi_asset_returns.csv`
- `artifacts/runs/<run-id>/inputs/multi_asset_returns.csv`
- `artifacts/runs/<run-id>/outputs/integrated_result.json`
- `artifacts/runs/<run-id>/outputs/integrated_run_metadata.json`

Multi-asset correlation detector writes a run bundle under `artifacts/runs/<run-id>/` plus the latest operator report in `docs/multi_asset_report.md`:

- `artifacts/runs/<run-id>/inputs/multi_asset_returns.csv`
- `artifacts/runs/<run-id>/outputs/multi_asset_anomalies.csv`
- `artifacts/runs/<run-id>/outputs/multi_asset_anomaly_chart.png`
- `artifacts/runs/<run-id>/outputs/multi_asset_report_metadata.json`
- `artifacts/runs/<run-id>/outputs/multi_asset_report.md`
- `docs/multi_asset_report.md`

## CI

GitHub Actions runs:

- lint via `ruff`
- pytest suite
- packaging metadata smoke checks
- secret scanning
- dependency auditing
- tag-driven release packaging

Workflow file:

- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `.github/workflows/release.yml`

## Notes

- `experiments/` intentionally contains stress harnesses, simulations, and demo training code. Those files are not part of the production path.
- External context uses real network calls to Google News RSS and returns verifiable headlines instead of invented fallback text.
- DeepSpeed runtime is adapted at execution time to avoid CPU offload settings that were causing unstable local JIT failures.
