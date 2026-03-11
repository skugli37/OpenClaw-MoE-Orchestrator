from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError
from .llm import ModelRole, load_manifest


@dataclass(frozen=True)
class SyncPlan:
    requested_roles: tuple[ModelRole, ...]
    installed: tuple[str, ...]
    missing: tuple[str, ...]


def _installed_models() -> tuple[str, ...]:
    if not shutil.which("ollama"):
        raise ConfigurationError("`ollama` binary is not installed or not on PATH")
    result = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    installed: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed.append(parts[0])
    return tuple(installed)


def build_sync_plan(manifest_path: Path | str, roles: tuple[ModelRole, ...] | None = None) -> SyncPlan:
    manifest = load_manifest(manifest_path)
    requested_roles = roles or tuple(ModelRole(role) for role in ModelRole)
    desired_models = []
    for role in requested_roles:
        desired_models.extend(spec.model for spec in manifest.models_for_role(role))
    installed = _installed_models()
    missing = tuple(model for model in dict.fromkeys(desired_models) if model not in installed)
    return SyncPlan(requested_roles=requested_roles, installed=installed, missing=missing)


def sync_models(
    manifest_path: Path | str,
    *,
    roles: tuple[ModelRole, ...] | None = None,
    max_models: int | None = None,
    dry_run: bool = False,
) -> dict:
    plan = build_sync_plan(manifest_path, roles=roles)
    to_pull = list(plan.missing)
    if max_models is not None:
        to_pull = to_pull[:max_models]

    pulled: list[str] = []
    for model in to_pull:
        if dry_run:
            continue
        subprocess.run(["ollama", "pull", model], check=True)
        pulled.append(model)

    return {
        "requested_roles": [role.value for role in plan.requested_roles],
        "installed": list(plan.installed),
        "missing": list(plan.missing),
        "selected_for_pull": to_pull,
        "pulled": pulled,
        "dry_run": dry_run,
    }
