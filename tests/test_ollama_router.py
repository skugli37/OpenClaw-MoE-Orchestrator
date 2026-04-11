from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator.llm import (  # noqa: E402
    ModelRole,
    OllamaClientError,
    OllamaManifest,
    OllamaModelSpec,
    OllamaRouter,
)


class FakeClient:
    def __init__(self):
        self.calls: list[str] = []

    def list_models(self) -> list[str]:
        return ["gpt-oss:120b-cloud", "qwen3-next:80b-cloud"]

    def chat(self, model: str, messages: list[dict[str, str]], options: dict | None = None):
        self.calls.append(model)
        if model == "gpt-oss:120b-cloud":
            raise OllamaClientError("simulated failure")
        return type("Response", (), {"model": model, "content": "ok"})()


def make_manifest() -> OllamaManifest:
    return OllamaManifest(
        base_url="http://127.0.0.1:11434",
        timeout_seconds=10,
        cooldown_seconds=30,
        models=(
            OllamaModelSpec(role=ModelRole.REASONING, model="gpt-oss:120b-cloud", priority=120),
            OllamaModelSpec(role=ModelRole.REASONING, model="qwen3-next:80b-cloud", priority=110),
            OllamaModelSpec(role=ModelRole.CODING, model="qwen3-coder-next:cloud", priority=120),
        ),
    )


def test_router_fails_over_to_next_healthy_model() -> None:
    router = OllamaRouter(make_manifest(), client=FakeClient())

    decision, response = router.chat(ModelRole.REASONING, [{"role": "user", "content": "hi"}])

    assert decision.model == "qwen3-next:80b-cloud"
    assert decision.attempted_models == ("gpt-oss:120b-cloud", "qwen3-next:80b-cloud")
    assert response.content == "ok"
    assert router.health.snapshot()["gpt-oss:120b-cloud"]["consecutive_failures"] == 1


def test_router_raises_when_role_missing() -> None:
    manifest = OllamaManifest(
        base_url="http://127.0.0.1:11434",
        timeout_seconds=10,
        cooldown_seconds=30,
        models=(),
    )
    router = OllamaRouter(manifest, client=FakeClient())

    with pytest.raises(Exception, match="No models configured for role"):
        router.select_model(ModelRole.SAFETY)
