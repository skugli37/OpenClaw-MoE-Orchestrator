import torch
import torch.nn as nn
import deepspeed
from deepspeed.moe.layer import MoE

class MoEModel(nn.Module):
    def __init__(self, hidden_dim=1024, num_experts=8, top_k=2):
        super(MoEModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Implementacija MoE sloja koristeći DeepSpeed MoE
        # expert_module je funkcija koja kreira eksperta
        expert_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.moe_layer = MoE(
            hidden_size=hidden_dim,
            expert=expert_module,
            num_experts=num_experts,
            k=top_k,
            use_residual=True
        )
        
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.backbone(x)
        # MoE sloj vraća output, loss za balansiranje opterećenja i dodatne metapodatke
        output, l_aux, _ = self.moe_layer(x)
        logits = self.head(output)
        return logits, l_aux

def main():
    # Inicijalizacija modela
    model = MoEModel()
    
    # Parametri za DeepSpeed
    ds_config = "ds_config_zero3.json"
    
    # Inicijalizacija DeepSpeed engine-a
    # Napomena: U produkciji bi ovde išli pravi podaci i distributivno okruženje
    print("Initializing DeepSpeed MoE with ZeRO-3...")
    
    # Dummy data za testiranje protoka
    inputs = torch.randn(16, 128, 1024).half()
    targets = torch.randint(0, 10, (16, 128)).cuda() if torch.cuda.is_available() else torch.randint(0, 10, (16, 128))

    # DeepSpeed inicijalizacija (u sandboxu će pokušati CPU offload ako nema GPU)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    print("Starting MoE training step...")
    for step in range(10):
        outputs, l_aux = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, 10), targets.view(-1)) + l_aux
        
        model_engine.backward(loss)
        model_engine.step()
        
        print(f"Step {step} complete. Loss: {loss.item()}")

if __name__ == "__main__":
    main()
