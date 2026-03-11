import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import deepspeed
import time
import subprocess
from deepspeed.moe.layer import MoE

# 1. MoE Arhitektura (Statistički Ekspert)
class IntelligenceMoE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_experts=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.moe = MoE(
            hidden_size=hidden_dim,
            expert=nn.Linear(hidden_dim, hidden_dim),
            num_experts=num_experts,
            k=1,
            use_residual=True,
            min_capacity=0
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x, _, _ = self.moe(x)
        x = self.decoder(x)
        return x

# 2. News-Oracle (OpenClaw / Browser Agent)
from browser_news_oracle import get_live_news

def activate_news_oracle(asset_name):
    print(f"\n[OpenClaw Trigger] MoE detektovao anomaliju na {asset_name}! Aktiviram Browser Agent...")
    news = get_live_news(asset_name)
    print(f"[Analysis] Pronađeni naslovi: {news}")
    return f"Vesti sugerišu sledeće za {asset_name}: {news}"

def run_orchestration():
    print("--- 🦞 Integrated OpenClaw MoE Orchestrator: ACTIVE ---")
    
    # Priprema live podataka
    data_df = pd.read_csv("multi_asset_returns.csv")
    data = torch.tensor(data_df.values.astype(np.float32)).half()
    
    model = IntelligenceMoE().half()
    
    # DeepSpeed inicijalizacija
    model_engine, _, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=model.parameters(),
        config="/home/ubuntu/ds_config_zero2.json"
    )

    # Detekcija anomalija u realnom vremenu
    with torch.no_grad():
        reconstructions = model(data)
        mse = torch.mean((reconstructions.float() - data.float())**2, dim=1)
        threshold = torch.quantile(mse, 0.98)
        
        for i in range(len(mse)):
            if mse[i] > threshold:
                asset_idx = torch.argmax(torch.abs(data[i])).item()
                assets = ["BTC", "ETH", "SOL"]
                trigger_asset = assets[asset_idx]
                
                reason = activate_news_oracle(trigger_asset)
                print(f"\n[FINAL REPORT] Anomalija detektovana: {trigger_asset}")
                print(f"[REASONING] {reason}")
                break

if __name__ == "__main__":
    run_orchestration()
