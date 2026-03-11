# OpenClaw MoE Orchestrator

Production-oriented market analysis and anomaly detection workflows built around DeepSpeed Mixture-of-Experts models and zero-cost external news validation.

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

Use the local virtual environment and install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

If you are already working inside the repository `.venv`, install only the missing tooling:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
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
```

## Outputs

Single-asset mission writes:

- `data/processed/market_data_norm.csv`
- `artifacts/anomaly_results.csv`
- `artifacts/anomaly_chart.png`
- `artifacts/mission_run_metadata.json`
- `docs/mission_report.md`

Integrated orchestrator writes:

- `data/processed/multi_asset_returns.csv`
- `artifacts/integrated_run_metadata.json`

Multi-asset correlation detector writes:

- `artifacts/multi_asset_anomalies.csv`
- `artifacts/multi_asset_anomaly_chart.png`
- `artifacts/multi_asset_report_metadata.json`
- `docs/multi_asset_report.md`

## CI

GitHub Actions runs:

- lint via `ruff`
- pytest suite
- packaging metadata smoke checks

Workflow file:

- `.github/workflows/ci.yml`

## Notes

- `experiments/` intentionally contains stress harnesses, simulations, and demo training code. Those files are not part of the production path.
- External context uses real network calls to Google News RSS and returns verifiable headlines instead of invented fallback text.
- DeepSpeed runtime is adapted at execution time to avoid CPU offload settings that were causing unstable local JIT failures.
