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
- Highest anomaly date: 2025-02-17
- Maximum reconstruction error: 0.458391
- BTC normalized return: -0.152773
- ETH normalized return: 0.758569
- SOL normalized return: -1.246642

## Visual Evidence
The correlation anomaly chart is saved under `artifacts/multi_asset_anomaly_chart.png`.
