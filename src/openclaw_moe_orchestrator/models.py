import torch
import torch.nn as nn
from deepspeed.moe.layer import MoE


def _dynamo_disable_decorator():
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "disable"):
        return compiler.disable

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "disable"):
        return dynamo.disable

    def passthrough(function):
        return function

    return passthrough


disable_dynamo = _dynamo_disable_decorator()


class MoEAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.moe = MoE(
            hidden_size=hidden_dim,
            expert=nn.Linear(hidden_dim, hidden_dim),
            num_experts=num_experts,
            k=1,
            use_residual=True,
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    @disable_dynamo
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.encoder(x))
        x, _, _ = self.moe(x)
        return self.decoder(x)


class IntelligenceMoE(MoEAutoencoder):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_experts: int = 4) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_experts=num_experts)


class MultiAssetMoE(MoEAutoencoder):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, num_experts: int = 4) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, num_experts=num_experts)
