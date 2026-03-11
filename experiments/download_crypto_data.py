import pandas as pd
import yfinance as yf
from datetime import datetime

def download_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if not data.empty:
        filename = f"{ticker.replace('-', '_')}_history.csv"
        data.to_csv(filename)
        print(f"Saved {ticker} data to {filename}")
        return filename
    else:
        print(f"No data found for {ticker}")
        return None

if __name__ == "__main__":
    # Postavljamo opseg podataka od početka 2025. do danas (februar 2026.)
    start = "2025-01-01"
    end = "2026-02-09"
    
    tickers = ["BTC-USD", "ETH-USD"]
    for t in tickers:
        download_data(t, start, end)
