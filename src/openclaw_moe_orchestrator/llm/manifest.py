from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..exceptions import ConfigurationError


class ModelRole(StrEnum):
    REASONING = "reasoning"
    CODING = "coding"
    GENERAL = "general"
    VISION = "vision"
    EMBEDDING = "embedding"
    SAFETY = "safety"


@dataclass(frozen=True)
class OllamaModelSpec:
    role: ModelRole
    model: str
    priority: int
    warm: bool = False
    max_concurrency: int = 1
    min_context_window: int | None = None
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class OllamaManifest:
    base_url: str
    timeout_seconds: float
    cooldown_seconds: float
    models: tuple[OllamaModelSpec, ...] = field(default_factory=tuple)

    def models_for_role(self, role: ModelRole) -> tuple[OllamaModelSpec, ...]:
        return tuple(spec for spec in self.models if spec.role == role)


def load_manifest(path: Path | str) -> OllamaManifest:
    payload = json.loads(Path(path).read_text())
    try:
        models = tuple(
            OllamaModelSpec(
                role=ModelRole(item["role"]),
                model=str(item["model"]),
                priority=int(item.get("priority", 100)),
                warm=bool(item.get("warm", False)),
                max_concurrency=int(item.get("max_concurrency", 1)),
                min_context_window=(
                    int(item["min_context_window"])
                    if item.get("min_context_window") is not None
                    else None
                ),
                capabilities=tuple(str(value) for value in item.get("capabilities", [])),
            )
            for item in payload["models"]
        )
        return OllamaManifest(
            base_url=str(payload.get("base_url", "http://127.0.0.1:11434")),
            timeout_seconds=float(payload.get("timeout_seconds", 60.0)),
            cooldown_seconds=float(payload.get("cooldown_seconds", 30.0)),
            models=models,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Invalid Ollama manifest at {path}: {error}") from error
