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
- Highest anomaly date: 2025-01-18
- Maximum reconstruction error: 1.292100
- BTC normalized return: 0.000343
- ETH normalized return: -1.195108
- SOL normalized return: 4.278170

## Visual Evidence
The correlation anomaly chart is saved under `artifacts/multi_asset_anomaly_chart.png`.
