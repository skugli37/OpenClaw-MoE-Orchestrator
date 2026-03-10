import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import deepspeed
from deepspeed.moe.layer import MoE

class MultiAssetMoE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_experts=4):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.moe = MoE(
            hidden_size=hidden_dim,
            expert=nn.Linear(hidden_dim, hidden_dim),
            num_experts=num_experts,
            k=1,
            use_residual=True
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x, _, _ = self.moe(x)
        x = self.decoder(x)
        return x

def train_and_analyze():
    # Učitavanje podataka (BTC, ETH, SOL)
    data_df = pd.read_csv("multi_asset_returns.csv")
    data = data_df.values.astype(np.float32)
    inputs = torch.tensor(data).half()
    
    model = MultiAssetMoE(input_dim=3).half()
    
    # Inicijalizacija DeepSpeed-a sa ZeRO-2
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=model.parameters(),
        config="/home/ubuntu/ds_config_zero2.json"
    )

    print("Training Multi-Asset MoE...")
    for epoch in range(100):
        outputs = model_engine(inputs)
        loss = nn.MSELoss()(outputs, inputs)
        model_engine.backward(loss)
        model_engine.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # Detekcija anomalija u korelacijama
    with torch.no_grad():
        reconstructions = model(inputs)
        mse = torch.mean((reconstructions.float() - inputs.float())**2, dim=1)
        threshold = torch.quantile(mse, 0.95)
        anomalies = (mse > threshold).numpy()
        
        results_df = data_df.copy()
        results_df['is_anomaly'] = anomalies
        results_df['reconstruction_error'] = mse.numpy()
        results_df.to_csv("multi_asset_anomalies.csv", index=False)
        print(f"Analysis complete. Anomalies detected: {anomalies.sum()}")

if __name__ == "__main__":
    train_and_analyze()
