from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path

import torch

from .llm import OllamaClientError
from .openclaw_local import DEFAULT_OLLAMA_BASE_URL, OpenClawLocalLayout
from .paths import RepoPaths


def collect_environment_report(paths: RepoPaths) -> dict:
    config_files = {
        "zero2": str(paths.config_dir / "ds_config_zero2.json"),
        "zero3": str(paths.config_dir / "ds_config_zero3.json"),
    }
    packages = {}
    for package in ("deepspeed", "matplotlib", "numpy", "pandas", "requests", "torch", "yfinance"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    ollama_models: list[str] | None
    ollama_error: str | None = None
    try:
        from .llm import OllamaClient

        ollama_models = OllamaClient(base_url=DEFAULT_OLLAMA_BASE_URL, timeout_seconds=5).list_models()
    except (OllamaClientError, Exception) as error:
        ollama_models = None
        ollama_error = str(error)

    openclaw_layout = OpenClawLocalLayout.discover()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(paths.repo_root),
        "configs": {name: {"path": path, "exists": Path(path).exists()} for name, path in config_files.items()},
        "artifacts_dir": str(paths.artifacts_dir),
        "processed_data_dir": str(paths.processed_data_dir),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "openclaw_binary": shutil.which("openclaw"),
        "openclaw_state_dir": str(openclaw_layout.state_dir),
        "ollama_binary": shutil.which("ollama"),
        "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
        "ollama_models": ollama_models,
        "ollama_error": ollama_error,
    }


def format_environment_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=True, indent=2)
