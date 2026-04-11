import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openclaw_moe_orchestrator.runtime import (
    load_runtime_config,
    prepare_distributed_env,
    prepare_model_and_optimizer,
    shutdown_distributed,
    tensor_dtype,
)

__all__ = [
    "load_runtime_config",
    "prepare_distributed_env",
    "prepare_model_and_optimizer",
    "shutdown_distributed",
    "tensor_dtype",
]
