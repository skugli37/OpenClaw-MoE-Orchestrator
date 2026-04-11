from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def visualize_single_asset(results_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(results_path, parse_dates=["Date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 8))
    plt.plot(df["Date"], df["Close"], label="BTC Price (USD)", color="blue", alpha=0.6)
    anomalies = df[df["Is_Anomaly"]]
    plt.scatter(
        anomalies["Date"],
        anomalies["Close"],
        color="red",
        label="Detected Anomalies",
        s=50,
        edgecolors="black",
    )
    plt.title("Bitcoin Price & Detected Anomalies (MoE + DeepSpeed Analysis)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def visualize_multi_asset(results_path: Path, output_path: Path) -> Path:
    df = pd.read_csv(results_path, parse_dates=["Date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cum_returns = (1 + df[["BTC-USD", "ETH-USD", "SOL-USD"]]).cumprod()
    plt.figure(figsize=(15, 8))
    plt.plot(df["Date"], cum_returns["BTC-USD"], label="BTC", alpha=0.7)
    plt.plot(df["Date"], cum_returns["ETH-USD"], label="ETH", alpha=0.7)
    plt.plot(df["Date"], cum_returns["SOL-USD"], label="SOL", alpha=0.7)
    anomalies = df[df["is_anomaly"]]
    plt.scatter(
        anomalies["Date"],
        cum_returns.loc[anomalies.index, "BTC-USD"],
        color="red",
        label="Correlation Anomaly",
        zorder=5,
    )
    plt.title("Multi-Asset Correlation Anomalies (BTC, ETH, SOL) - MoE Analysis")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
