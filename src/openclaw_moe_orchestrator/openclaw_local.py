from __future__ import annotations

import json
import os
import secrets
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigurationError
from .llm import ModelRole, OllamaClient, OllamaManifest, OllamaModelEntry, OllamaModelSpec, load_manifest
from .paths import RepoPaths

DEFAULT_OPENCLAW_STATE_DIR = Path.home() / ".openclaw"
DEFAULT_OLLAMA_AUTH_MARKER = "ollama-cloud"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class OpenClawLocalLayout:
    state_dir: Path
    workspace_dir: Path
    skills_dir: Path
    main_agent_dir: Path
    main_session_dir: Path
    auth_profiles_path: Path
    config_path: Path
    overlay_config_path: Path

    @classmethod
    def discover(cls, state_dir: Path | None = None) -> "OpenClawLocalLayout":
        root = (state_dir or DEFAULT_OPENCLAW_STATE_DIR).expanduser().resolve()
        workspace_dir = root / "workspace"
        skills_dir = workspace_dir / "skills"
        main_agent_dir = root / "agents" / "main" / "agent"
        return cls(
            state_dir=root,
            workspace_dir=workspace_dir,
            skills_dir=skills_dir,
            main_agent_dir=main_agent_dir,
            main_session_dir=root / "agents" / "main" / "sessions",
            auth_profiles_path=main_agent_dir / "auth-profiles.json",
            config_path=root / "openclaw.json",
            overlay_config_path=root / "openclaw.moe.local.json",
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.state_dir,
            self.workspace_dir,
            self.skills_dir,
            self.main_agent_dir,
            self.main_session_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def render_local_auth_profiles(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> dict:
    del base_url
    return {
        "profiles": {
            "ollama:default": {
                "provider": "ollama",
                "type": "api_key",
                "key": DEFAULT_OLLAMA_AUTH_MARKER,
            }
        },
        "usageStats": {},
    }


def build_openclaw_local_overlay(
    manifest_path: Path | str,
    workspace_dir: Path | None = None,
    gateway_token: str | None = None,
    role_model_overrides: dict[ModelRole, tuple[str, ...]] | None = None,
) -> dict:
    manifest = load_manifest(manifest_path)
    ordered_manifest = reorder_manifest(manifest, role_model_overrides=role_model_overrides)
    reasoning_models = _ordered_models_for_role(ordered_manifest, ModelRole.REASONING)
    coding_models = _ordered_models_for_role(ordered_manifest, ModelRole.CODING)
    general_models = _ordered_models_for_role(ordered_manifest, ModelRole.GENERAL)
    primary_reasoning = reasoning_models[0]
    fallback_models = _dedupe_models(reasoning_models[1:] + general_models[:1] + coding_models[:1], primary_reasoning)
    workspace_dir = workspace_dir or (DEFAULT_OPENCLAW_STATE_DIR / "workspace")
    gateway_token = gateway_token or secrets.token_hex(32)
    return {
        "gateway": {
            "mode": "local",
            "auth": {"token": gateway_token},
        },
        "auth": {
            "order": {
                "ollama": ["ollama:default"],
            }
        },
        "agents": {
            "defaults": {
                "workspace": str(workspace_dir),
                "sandbox": {"mode": "off"},
                "model": {
                    "primary": f"ollama/{primary_reasoning}",
                    "fallbacks": [f"ollama/{model}" for model in fallback_models],
                },
                "memorySearch": {
                    "enabled": False,
                },
            }
        },
        "models": {
            "mode": "replace",
            "providers": {
                "ollama": {
                    "baseUrl": ordered_manifest.base_url,
                    "apiKey": DEFAULT_OLLAMA_AUTH_MARKER,
                    "api": "ollama",
                    "models": [_render_provider_model(spec) for spec in ordered_manifest.models],
                }
            }
        },
    }


def _render_provider_model(spec: OllamaModelSpec) -> dict:
    context_window = spec.min_context_window or 32768
    return {
        "id": spec.model,
        "name": spec.model,
        "reasoning": spec.role == ModelRole.REASONING,
        "input": ["text"],
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
        },
        "contextWindow": context_window,
        "maxTokens": context_window * 10,
    }


def _ordered_models_for_role(manifest: OllamaManifest, role: ModelRole) -> list[str]:
    models = manifest.models_for_role(role)
    if not models:
        raise ConfigurationError(f"Ollama manifest does not define role {role}")
    ordered = sorted(models, key=lambda item: item.priority, reverse=True)
    return [item.model for item in ordered]


def reorder_manifest(
    manifest: OllamaManifest,
    *,
    role_model_overrides: dict[ModelRole, tuple[str, ...]] | None = None,
) -> OllamaManifest:
    if not role_model_overrides:
        return manifest

    overrides = {role: tuple(models) for role, models in role_model_overrides.items() if models}
    if not overrides:
        return manifest

    available_by_role: dict[ModelRole, dict[str, OllamaModelSpec]] = {
        role: {spec.model: spec for spec in manifest.models_for_role(role)}
        for role in ModelRole
    }
    live_models = _live_model_entries_by_name(manifest.base_url)
    normalized_models: list[OllamaModelSpec] = []
    for role in ModelRole:
        role_specs = list(manifest.models_for_role(role))
        if not role_specs:
            continue
        ordered_for_role: list[OllamaModelSpec] = []
        seen_models: set[str] = set()
        for model in overrides.get(role, ()):
            spec = available_by_role[role].get(model)
            if spec is None:
                spec = _build_live_override_spec(role, model, live_models.get(model))
                if spec is None:
                    raise ConfigurationError(f"Model {model} is not configured for role {role}")
            ordered_for_role.append(spec)
            seen_models.add(model)
        remaining_specs = sorted(role_specs, key=lambda item: item.priority, reverse=True)
        for spec in remaining_specs:
            if spec.model not in seen_models:
                ordered_for_role.append(spec)
        base_priority = max(spec.priority for spec in role_specs) + len(ordered_for_role)
        for index, spec in enumerate(ordered_for_role):
            normalized_models.append(
                OllamaModelSpec(
                    role=spec.role,
                    model=spec.model,
                    priority=base_priority - index,
                    warm=spec.warm,
                    max_concurrency=spec.max_concurrency,
                    min_context_window=spec.min_context_window,
                    capabilities=spec.capabilities,
                )
            )

    return OllamaManifest(
        base_url=manifest.base_url,
        timeout_seconds=manifest.timeout_seconds,
        cooldown_seconds=manifest.cooldown_seconds,
        models=tuple(normalized_models),
    )


def _live_model_entries_by_name(base_url: str) -> dict[str, OllamaModelEntry]:
    try:
        entries = OllamaClient(base_url=base_url).list_model_entries()
    except Exception:
        return {}
    return {entry.name: entry for entry in entries}


def _build_live_override_spec(
    role: ModelRole,
    model: str,
    live_entry: OllamaModelEntry | None,
) -> OllamaModelSpec | None:
    if live_entry is None:
        return None
    return OllamaModelSpec(
        role=role,
        model=model,
        priority=1000,
        warm=False,
        max_concurrency=1,
        min_context_window=_default_context_window_for_role(role),
        capabilities=_capabilities_for_live_entry(role, live_entry),
    )


def _default_context_window_for_role(role: ModelRole) -> int:
    if role is ModelRole.EMBEDDING:
        return 8192
    if role is ModelRole.VISION:
        return 65536
    return 131072


def _capabilities_for_live_entry(role: ModelRole, live_entry: OllamaModelEntry) -> tuple[str, ...]:
    values = ["live"]
    if role is ModelRole.EMBEDDING:
        values.append("embedding")
    elif role is ModelRole.VISION:
        values.append("vision")
    else:
        values.append("text")
    if live_entry.family:
        values.append(live_entry.family)
    if live_entry.remote_host:
        values.append("cloud")
    return tuple(values)


def _dedupe_models(models: list[str], primary_model: str) -> list[str]:
    unique_models = OrderedDict()
    for model in models:
        if model != primary_model:
            unique_models.setdefault(model, None)
    return list(unique_models.keys())


def install_workspace_skill(paths: RepoPaths, layout: OpenClawLocalLayout, mode: str = "copy") -> Path:
    source = paths.repo_root / "skills" / "openclaw-moe-orchestrator"
    if not source.exists():
        raise ConfigurationError(f"Missing workspace skill at {source}")
    target = layout.skills_dir / source.name

    if target.exists() or target.is_symlink():
        if mode == "symlink" and target.is_symlink() and target.resolve() == source.resolve():
            return target
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    if mode == "symlink":
        target.symlink_to(source, target_is_directory=True)
    elif mode == "copy":
        shutil.copytree(source, target)
    else:
        raise ConfigurationError(f"Unsupported skill installation mode: {mode}")
    return target


def write_local_auth_profiles(layout: OpenClawLocalLayout, base_url: str = DEFAULT_OLLAMA_BASE_URL) -> Path:
    layout.ensure_directories()
    layout.auth_profiles_path.write_text(json.dumps(render_local_auth_profiles(base_url), indent=2))
    return layout.auth_profiles_path


def write_openclaw_overlay(
    layout: OpenClawLocalLayout,
    manifest_path: Path | str,
    *,
    role_model_overrides: dict[ModelRole, tuple[str, ...]] | None = None,
) -> Path:
    layout.ensure_directories()
    existing_token = None
    if layout.overlay_config_path.exists():
        try:
            payload = json.loads(layout.overlay_config_path.read_text())
            existing_token = payload.get("gateway", {}).get("auth", {}).get("token")
        except json.JSONDecodeError:
            existing_token = None
    overlay = build_openclaw_local_overlay(
        manifest_path,
        workspace_dir=layout.workspace_dir,
        gateway_token=existing_token,
        role_model_overrides=role_model_overrides,
    )
    layout.overlay_config_path.write_text(json.dumps(overlay, indent=2))
    return layout.overlay_config_path


def install_openclaw_local_bundle(
    paths: RepoPaths,
    *,
    state_dir: Path | None = None,
    skill_mode: str = "copy",
    manifest_path: Path | None = None,
    role_model_overrides: dict[ModelRole, tuple[str, ...]] | None = None,
) -> dict[str, str]:
    layout = OpenClawLocalLayout.discover(state_dir)
    layout.ensure_directories()
    manifest_path = manifest_path or (paths.config_dir / "ollama_model_manifest.json")
    skill_path = install_workspace_skill(paths, layout, mode=skill_mode)
    auth_profiles_path = write_local_auth_profiles(layout)
    overlay_path = write_openclaw_overlay(
        layout,
        manifest_path,
        role_model_overrides=role_model_overrides,
    )
    active_config_path = layout.config_path
    backup_path = None
    if active_config_path.exists():
        current_payload = active_config_path.read_text()
        overlay_payload = overlay_path.read_text()
        if current_payload != overlay_payload:
            backup_path = active_config_path.with_suffix(".json.bak")
            backup_path.write_text(current_payload)
    shutil.copy2(overlay_path, active_config_path)
    _harden_permissions(layout)
    result = {
        "state_dir": str(layout.state_dir),
        "workspace_skill": str(skill_path),
        "auth_profiles": str(auth_profiles_path),
        "overlay_config": str(overlay_path),
        "active_config": str(active_config_path),
        "next_steps": (
            "Install OpenClaw, authenticate Ollama Cloud, then start the gateway with ~/.openclaw/openclaw.json"
        ),
    }
    if backup_path is not None:
        result["active_config_backup"] = str(backup_path)
    if role_model_overrides:
        result["role_model_overrides"] = {
            role.value: list(models)
            for role, models in role_model_overrides.items()
            if models
        }
    return result


def collect_openclaw_local_status(paths: RepoPaths, state_dir: Path | None = None) -> dict:
    layout = OpenClawLocalLayout.discover(state_dir)
    manifest_path = paths.config_dir / "ollama_model_manifest.json"
    return {
        "state_dir": str(layout.state_dir),
        "openclaw_binary": shutil.which("openclaw"),
        "ollama_host": os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_BASE_URL),
        "config_exists": layout.config_path.exists(),
        "overlay_exists": layout.overlay_config_path.exists(),
        "auth_profiles_exists": layout.auth_profiles_path.exists(),
        "workspace_skill_exists": (layout.skills_dir / "openclaw-moe-orchestrator").exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
    }


def _harden_permissions(layout: OpenClawLocalLayout) -> None:
    os.chmod(layout.state_dir, 0o700)
    for path in (layout.auth_profiles_path, layout.overlay_config_path, layout.config_path):
        if path.exists():
            os.chmod(path, 0o600)
