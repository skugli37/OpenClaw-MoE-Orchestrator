import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize():
    df = pd.read_csv("multi_asset_anomalies.csv")
    
    # Kreiranje grafikona sa tri aseta i označenim anomalijama
    plt.figure(figsize=(15, 8))
    
    # Kumulativni povraćaji za vizuelizaciju trenda
    cum_returns = (1 + df[['BTC-USD', 'ETH-USD', 'SOL-USD']]).cumprod()
    
    plt.plot(cum_returns.index, cum_returns['BTC-USD'], label='BTC', alpha=0.7)
    plt.plot(cum_returns.index, cum_returns['ETH-USD'], label='ETH', alpha=0.7)
    plt.plot(cum_returns.index, cum_returns['SOL-USD'], label='SOL', alpha=0.7)
    
    # Označavanje anomalija (reconstruction error > threshold)
    anomalies = df[df['is_anomaly'] == True]
    plt.scatter(anomalies.index, cum_returns.iloc[anomalies.index]['BTC-USD'], 
                color='red', label='Correlation Anomaly', zorder=5)
    
    plt.title("Multi-Asset Correlation Anomalies (BTC, ETH, SOL) - MoE Analysis", fontsize=15)
    plt.xlabel("Days (from 2025-01-01)", fontsize=12)
    plt.ylabel("Cumulative Returns (Normalized)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("multi_asset_anomaly_chart.png")
    print("Multi-asset visualization saved as multi_asset_anomaly_chart.png")

if __name__ == "__main__":
    visualize()
