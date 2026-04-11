from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator.llm import ModelRole  # noqa: E402
from openclaw_moe_orchestrator.ollama_sync import build_sync_plan, sync_models  # noqa: E402


def test_build_sync_plan_reports_missing_models(monkeypatch) -> None:
    def fake_run(cmd, check, capture_output, text):
        assert cmd == ["ollama", "list"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="NAME ID SIZE MODIFIED\ngpt-oss:120b-cloud abc - now\n",
            stderr="",
        )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(subprocess, "run", fake_run)

    plan = build_sync_plan(
        REPO_ROOT / "configs" / "ollama_model_manifest.json",
        roles=(ModelRole.REASONING, ModelRole.GENERAL),
    )

    assert "gpt-oss:120b-cloud" in plan.installed
    assert "qwen3-next:80b-cloud" in plan.missing


def test_sync_models_dry_run_does_not_pull(monkeypatch) -> None:
    def fake_run(cmd, check, capture_output=False, text=False):
        assert cmd == ["ollama", "list"]
        return subprocess.CompletedProcess(cmd, 0, stdout="NAME ID SIZE MODIFIED\n", stderr="")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/ollama")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = sync_models(
        REPO_ROOT / "configs" / "ollama_model_manifest.json",
        roles=(ModelRole.SAFETY,),
        dry_run=True,
    )

    assert result["selected_for_pull"] == ["gpt-oss:120b-cloud", "qwen3-next:80b-cloud"]
    assert result["pulled"] == []
