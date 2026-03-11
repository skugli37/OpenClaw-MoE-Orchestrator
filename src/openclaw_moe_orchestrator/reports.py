from pathlib import Path

import pandas as pd


def build_single_asset_report(results_path: Path) -> str:
    results_df = pd.read_csv(results_path, parse_dates=["Date"])
    anomalies = results_df[results_df["Is_Anomaly"]]
    top_anomaly = results_df.loc[results_df["Reconstruction_Error"].idxmax()]

    return f"""# OpenClaw Intelligence Report: BTC Anomaly Analysis

## Executive Summary
Analyzed {len(results_df)} BTC market observations from {results_df["Date"].min().date()} to {results_df["Date"].max().date()}.
The MoE model marked {len(anomalies)} observations as anomalous at the 95th percentile of reconstruction error.

## Technical Specs
- Framework: OpenClaw Orchestration
- Engine: DeepSpeed ZeRO-2
- Model: Mixture of Experts (4 Experts)
- Input features: normalized close, volume, returns, rolling volatility

## Findings
- Highest anomaly date: {top_anomaly["Date"].date()}
- BTC close at that point: ${top_anomaly["Close"]:,.2f}
- Maximum reconstruction error: {top_anomaly["Reconstruction_Error"]:.6f}

## Visual Evidence
The anomaly chart is saved under `artifacts/anomaly_chart.png`.
"""


def build_multi_asset_report(results_path: Path) -> str:
    results_df = pd.read_csv(results_path, parse_dates=["Date"])
    anomalies = results_df[results_df["is_anomaly"]]
    top_anomaly = results_df.loc[results_df["reconstruction_error"].idxmax()]

    return f"""# OpenClaw Multi-Asset Correlation Report

## Executive Summary
Analyzed {len(results_df)} multi-asset return observations from {results_df["Date"].min().date()} to {results_df["Date"].max().date()}.
The MoE model marked {len(anomalies)} observations as correlation anomalies at the 95th percentile of reconstruction error.

## Technical Specs
- Framework: OpenClaw Orchestration
- Engine: DeepSpeed ZeRO-2
- Assets: BTC-USD, ETH-USD, SOL-USD
- Signal: normalized daily returns

## Findings
- Highest anomaly date: {top_anomaly["Date"].date()}
- Maximum reconstruction error: {top_anomaly["reconstruction_error"]:.6f}
- BTC normalized return: {top_anomaly["BTC-USD"]:.6f}
- ETH normalized return: {top_anomaly["ETH-USD"]:.6f}
- SOL normalized return: {top_anomaly["SOL-USD"]:.6f}

## Visual Evidence
The correlation anomaly chart is saved under `artifacts/multi_asset_anomaly_chart.png`.
"""
