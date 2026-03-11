import pandas as pd
import numpy as np

def inject():
    df = pd.read_csv("multi_asset_returns.csv")
    
    # Ubrizgavamo anomaliju u poslednji red (SOL skok od 500% dok BTC/ETH miruju)
    # To je statistički nemoguće u normalnim uslovima korelacije
    last_idx = len(df) - 1
    df.iloc[last_idx, 0] = 0.0  # BTC miruje
    df.iloc[last_idx, 1] = 0.0  # ETH miruje
    df.iloc[last_idx, 2] = 5.0  # SOL skok (500% u jednom danu)
    
    df.to_csv("stress_test_data.csv", index=False)
    print(f"Injected extreme anomaly at index {last_idx} (SOL +500%). Saved to stress_test_data.csv")

if __name__ == "__main__":
    inject()
