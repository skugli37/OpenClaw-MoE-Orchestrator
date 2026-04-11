from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator import cli  # noqa: E402
from openclaw_moe_orchestrator.llm import ModelRole  # noqa: E402


def test_cli_list_ollama_models(monkeypatch, capsys) -> None:
    class FakeClient:
        def list_models(self) -> list[str]:
            return ["gpt-oss:120b-cloud"]

    monkeypatch.setattr("openclaw_moe_orchestrator.llm.OllamaClient", lambda: FakeClient())
    monkeypatch.setattr(sys, "argv", ["openclaw-moe", "list-ollama-models"])

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"models": ["gpt-oss:120b-cloud"]}


def test_role_model_overrides_from_args() -> None:
    args = Namespace(
        reasoning_model=["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"],
        coding_model=["qwen3-coder-next:cloud"],
        general_model=None,
        vision_model=None,
        embedding_model=None,
        safety_model=None,
    )

    result = cli._role_model_overrides_from_args(args)

    assert result == {
        ModelRole.REASONING: ("qwen3-next:80b-cloud", "gpt-oss:120b-cloud"),
        ModelRole.CODING: ("qwen3-coder-next:cloud",),
    }


def test_role_model_overrides_from_args_rejects_duplicates() -> None:
    args = Namespace(
        reasoning_model=["gpt-oss:120b-cloud", "gpt-oss:120b-cloud"],
        coding_model=None,
        general_model=None,
        vision_model=None,
        embedding_model=None,
        safety_model=None,
    )

    with pytest.raises(Exception, match="Duplicate model override"):
        cli._role_model_overrides_from_args(args)
