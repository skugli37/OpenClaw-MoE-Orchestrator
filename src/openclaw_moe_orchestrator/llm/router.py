from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import ConfigurationError
from .client import OllamaClient, OllamaClientError, OllamaResponse
from .health import ModelHealthTracker
from .manifest import ModelRole, OllamaManifest, OllamaModelSpec


class RoutingError(RuntimeError):
    """Raised when no healthy local model can satisfy a routed request."""


@dataclass(frozen=True)
class RoutingDecision:
    role: ModelRole
    model: str
    attempted_models: tuple[str, ...]


class OllamaRouter:
    def __init__(self, manifest: OllamaManifest, client: OllamaClient | None = None):
        self.manifest = manifest
        self.client = client or OllamaClient(
            base_url=manifest.base_url,
            timeout_seconds=manifest.timeout_seconds,
        )
        self.health = ModelHealthTracker(cooldown_seconds=manifest.cooldown_seconds)
        self._round_robin_index: dict[ModelRole, int] = {role: 0 for role in ModelRole}

    def installed_models(self) -> list[str]:
        return self.client.list_models()

    def select_model(self, role: ModelRole, available_models: set[str] | None = None) -> RoutingDecision:
        specs = list(self.manifest.models_for_role(role))
        if not specs:
            raise ConfigurationError(f"No models configured for role {role}")

        ordered = self._ordered_specs(role, specs)
        attempted: list[str] = []
        for spec in ordered:
            attempted.append(spec.model)
            if available_models is not None and spec.model not in available_models:
                continue
            if self.health.is_available(spec.model):
                return RoutingDecision(role=role, model=spec.model, attempted_models=tuple(attempted))

        raise RoutingError(f"No healthy installed models available for role {role}")

    def chat(
        self,
        role: ModelRole,
        messages: list[dict[str, str]],
        *,
        options: dict | None = None,
    ) -> tuple[RoutingDecision, OllamaResponse]:
        available_models = set(self.installed_models())
        last_error: Exception | None = None
        attempted: list[str] = []

        for _ in self.manifest.models_for_role(role):
            decision = self.select_model(role, available_models=available_models)
            attempted.extend(model for model in decision.attempted_models if model not in attempted)
            try:
                response = self.client.chat(decision.model, messages, options=options)
            except OllamaClientError as error:
                self.health.record_failure(decision.model, str(error))
                last_error = error
                available_models.discard(decision.model)
                continue
            self.health.record_success(decision.model)
            return RoutingDecision(role=role, model=decision.model, attempted_models=tuple(attempted)), response

        raise RoutingError(f"All routed models failed for role {role}: {last_error}")

    def _ordered_specs(self, role: ModelRole, specs: list[OllamaModelSpec]) -> list[OllamaModelSpec]:
        specs = sorted(specs, key=lambda item: item.priority, reverse=True)
        if not specs:
            return specs
        index = self._round_robin_index.get(role, 0) % len(specs)
        ordered = specs[index:] + specs[:index]
        self._round_robin_index[role] = (index + 1) % len(specs)
        return ordered
