# OpenClaw Multi-Asset Correlation Report

## Executive Summary
Analyzed 423 multi-asset return observations from 2025-01-02 to 2026-02-28.
The MoE model marked 22 observations as correlation anomalies at the 95th percentile of reconstruction error.

## Technical Specs
- Framework: OpenClaw Orchestration
- Engine: DeepSpeed ZeRO-2
- Assets: BTC-USD, ETH-USD, SOL-USD
- Signal: normalized daily returns

## Findings
- Highest anomaly date: 2026-02-05
- Maximum reconstruction error: 0.706057
- BTC normalized return: -5.889935
- ETH normalized return: -3.727724
- SOL normalized return: -3.307565

## Visual Evidence
The correlation anomaly chart is saved under `/home/kugli/OpenClaw-MoE-Orchestrator/artifacts/runs/multi-asset-report-20260311T110832Z-35e0848b/outputs/multi_asset_anomaly_chart.png`.
