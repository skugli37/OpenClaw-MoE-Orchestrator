from __future__ import annotations

import fcntl
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from .exceptions import ResourceContentionError

GPU_LOCK_TIMEOUT_SECONDS = 900
GPU_LOCK_POLL_INTERVAL_SECONDS = 1.0
MIN_ZERO_BUCKET_SIZE_BYTES = 16 * 1024 * 1024
MAX_ZERO_BUCKET_SIZE_BYTES = 128 * 1024 * 1024


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
    else:
        _cap_zero_bucket_sizes(config)

    return config


def _cap_zero_bucket_sizes(config: dict) -> None:
    zero_config = config.get("zero_optimization", {})
    if not zero_config:
        return

    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_bytes = min(MAX_ZERO_BUCKET_SIZE_BYTES, max(MIN_ZERO_BUCKET_SIZE_BYTES, total_memory // 20))
    dtype_bytes = 2 if config.get("fp16", {}).get("enabled", False) else 4
    capped_elements = max(1, target_bytes // dtype_bytes)

    for key in ("allgather_bucket_size", "reduce_bucket_size"):
        value = zero_config.get(key)
        if isinstance(value, (int, float)):
            zero_config[key] = min(int(value), int(capped_elements))


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def gpu_execution_lock(
    lock_path: Path | str,
    timeout_seconds: int = GPU_LOCK_TIMEOUT_SECONDS,
    poll_interval_seconds: float = GPU_LOCK_POLL_INTERVAL_SECONDS,
):
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("w")
    deadline = time.monotonic() + timeout_seconds
    acquired = False

    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise ResourceContentionError(
                        f"Timed out waiting for exclusive GPU lock at {path}"
                    ) from error
                time.sleep(poll_interval_seconds)

        lock_file.write(str(os.getpid()))
        lock_file.flush()
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
