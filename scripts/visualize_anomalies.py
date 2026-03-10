import pandas as pd
import matplotlib.pyplot as plt

def visualize():
    df = pd.read_csv("anomaly_results.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    
    plt.figure(figsize=(15, 8))
    
    # Glavni grafikon cene
    plt.plot(df.index, df['Close'], label='BTC Price (USD)', color='blue', alpha=0.6)
    
    # Markiranje anomalija
    anomalies = df[df['Is_Anomaly'] == True]
    plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Detected Anomalies', s=50, edgecolors='black')
    
    plt.title('Bitcoin Price & Detected Anomalies (MoE + DeepSpeed Analysis)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("anomaly_chart.png")
    print("Visualization saved as anomaly_chart.png")

if __name__ == "__main__":
    visualize()
