from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

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
    }


def format_environment_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=True, indent=2)
