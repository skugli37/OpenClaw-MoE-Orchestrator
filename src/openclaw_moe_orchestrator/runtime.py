from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


def prepare_distributed_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


def load_runtime_config(config_path: Path | str, batch_size: int) -> dict:
    config = json.loads(Path(config_path).read_text())
    zero_config = config.setdefault("zero_optimization", {})
    zero_config.pop("offload_optimizer", None)
    zero_config.pop("offload_param", None)
    zero_config.pop("cpu_offload", None)

    config["gradient_accumulation_steps"] = 1
    config["train_batch_size"] = batch_size
    config["train_micro_batch_size_per_gpu"] = batch_size

    if not torch.cuda.is_available():
        config.setdefault("fp16", {})["enabled"] = False

    return config


def prepare_model_and_optimizer(model: torch.nn.Module, config: dict):
    if torch.cuda.is_available():
        model = model.cuda().half()
    else:
        model = model.float()

    learning_rate = config.get("optimizer", {}).get("params", {}).get("lr", 1e-3)
    param_groups = split_params_into_different_moe_groups_for_optimizer(
        {"params": list(model.parameters()), "name": "main"}
    )
    optimizer = torch.optim.Adam(param_groups, lr=learning_rate)
    return model, optimizer, param_groups


def tensor_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
