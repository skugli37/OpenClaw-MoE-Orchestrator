# OpenClaw Intelligence Report: BTC Anomaly Analysis

## Executive Summary
Analyzed 400 BTC market observations from 2025-01-11 to 2026-02-14.
The MoE model marked 20 observations as anomalous at the 95th percentile of reconstruction error.

## Technical Specs
- Framework: OpenClaw Orchestration
- Engine: DeepSpeed ZeRO-2
- Model: Mixture of Experts (4 Experts)
- Input features: normalized close, volume, returns, rolling volatility

## Findings
- Highest anomaly date: 2026-02-06
- BTC close at that point: $70,555.39
- Maximum reconstruction error: 12.061468

## Visual Evidence
The anomaly chart is saved under `artifacts/anomaly_chart.png`.
