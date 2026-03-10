import yfinance as yf
import pandas as pd
import numpy as np

def prepare_data():
    print("Fetching BTC-USD data...")
    df = yf.download("BTC-USD", start="2025-01-01", end="2026-02-15")
    df = df[['Close', 'Volume']]
    df['Returns'] = df['Close'].pct_change()
    df['Vol'] = df['Returns'].rolling(window=10).std()
    df = df.dropna()
    df_norm = (df - df.mean()) / df.std()
    df_norm.to_csv("market_data_norm.csv", index=False)
    print(f"Prepared {len(df_norm)} data points. Saved to market_data_norm.csv")

if __name__ == "__main__":
    prepare_data()
