from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator import cli  # noqa: E402


def test_cli_list_ollama_models(monkeypatch, capsys) -> None:
    class FakeClient:
        def list_models(self) -> list[str]:
            return ["gpt-oss:120b-cloud"]

    monkeypatch.setattr("openclaw_moe_orchestrator.llm.OllamaClient", lambda: FakeClient())
    monkeypatch.setattr(sys, "argv", ["openclaw-moe", "list-ollama-models"])

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"models": ["gpt-oss:120b-cloud"]}
