import yfinance as yf
import pandas as pd
import numpy as np

def prepare_multi_data():
    assets = ["BTC-USD", "ETH-USD", "SOL-USD"]
    print(f"Fetching data for: {assets}...")
    
    data = yf.download(assets, start="2025-01-01", end="2026-03-01")['Close']
    
    # Izračunavanje povraćaja (returns)
    returns = data.pct_change().dropna()
    
    # Normalizacija
    returns_norm = (returns - returns.mean()) / returns.std()
    
    returns_norm.to_csv("multi_asset_returns.csv", index=False)
    print(f"Prepared {len(returns_norm)} data points. Saved to multi_asset_returns.csv")

if __name__ == "__main__":
    prepare_multi_data()
