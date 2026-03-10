import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import deepspeed
from deepspeed.moe.layer import MoE

class MoEAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_experts=4):
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

def train_and_detect():
    data = pd.read_csv("market_data_norm.csv").values.astype(np.float32)
    inputs = torch.tensor(data).half()
    model = MoEAutoencoder().half()
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=model.parameters(),
        config="/home/ubuntu/ds_config_zero2.json"
    )

    for epoch in range(50):
        outputs = model_engine(inputs)
        loss = nn.MSELoss()(outputs, inputs)
        model_engine.backward(loss)
        model_engine.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    with torch.no_grad():
        reconstructions = model(inputs)
        mse = torch.mean((reconstructions.float() - inputs.float())**2, dim=1)
        threshold = torch.quantile(mse, 0.95)
        anomalies = (mse > threshold).numpy()
        pd.DataFrame({"price": data[:, 0], "is_anomaly": anomalies}).to_csv("anomalies.csv", index=False)
        print(f"Anomaly detection complete. Threshold: {threshold.item():.6f}")

if __name__ == "__main__":
    train_and_detect()
