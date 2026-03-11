from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator.gui import (  # noqa: E402
    _role_model_overrides_from_payload,
    active_openclaw_profile,
    build_model_catalog,
    build_recent_runs,
)
from openclaw_moe_orchestrator.exceptions import ConfigurationError  # noqa: E402
from openclaw_moe_orchestrator.llm import ModelRole, OllamaModelEntry  # noqa: E402
from openclaw_moe_orchestrator.paths import RepoPaths  # noqa: E402


def test_role_model_overrides_from_payload() -> None:
    payload = {
        "reasoning_models": ["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"],
        "coding_models": ["qwen3-coder-next:cloud"],
    }

    result = _role_model_overrides_from_payload(payload)

    assert result == {
        ModelRole.REASONING: ("qwen3-next:80b-cloud", "gpt-oss:120b-cloud"),
        ModelRole.CODING: ("qwen3-coder-next:cloud",),
    }


def test_role_model_overrides_from_payload_rejects_duplicates() -> None:
    payload = {"reasoning_models": ["qwen3-next:80b-cloud", "qwen3-next:80b-cloud"]}

    try:
        _role_model_overrides_from_payload(payload)
    except ConfigurationError as error:
        assert "contains duplicates" in str(error)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ConfigurationError")


def test_build_recent_runs_reads_bundle_metadata(tmp_path: Path) -> None:
    runs_dir = tmp_path / "artifacts" / "runs" / "mission-demo"
    outputs_dir = runs_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "mission_report.md").write_text("# report")
    (outputs_dir / "mission_run_metadata.json").write_text(json.dumps({"workflow": "run-mission"}))

    local_paths = RepoPaths(
        repo_root=tmp_path,
        config_dir=tmp_path / "configs",
        docs_dir=tmp_path / "docs",
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        artifacts_dir=tmp_path / "artifacts",
        logs_dir=tmp_path / "artifacts" / "logs",
    )

    runs = build_recent_runs(local_paths)

    assert runs[0]["workflow"] == "run-mission"
    assert runs[0]["outputs"][0]["name"] == "mission_report.md"


def test_active_openclaw_profile_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    profile = active_openclaw_profile()

    assert profile == {"exists": False}


def test_build_model_catalog_marks_live_and_active_models(monkeypatch) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    monkeypatch.setattr(
        "openclaw_moe_orchestrator.gui.active_openclaw_profile",
        lambda: {
            "exists": True,
            "primary": "ollama/qwen3-next:80b-cloud",
            "fallbacks": ["ollama/gpt-oss:120b-cloud"],
        },
    )
    monkeypatch.setattr(
        "openclaw_moe_orchestrator.gui.OllamaClient.list_models",
        lambda self: ["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"],
    )

    catalog = build_model_catalog(paths)

    assert catalog["defaults"]["reasoning"][0] == "gpt-oss:120b-cloud"
    assert catalog["roles"]["reasoning"][0]["available"] is True
    assert any(item["active"] for item in catalog["roles"]["reasoning"])


def test_build_model_catalog_includes_live_only_models(monkeypatch) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    monkeypatch.setattr(
        "openclaw_moe_orchestrator.gui.active_openclaw_profile",
        lambda: {"exists": True, "primary": "ollama/qwen3-next:80b-cloud", "fallbacks": []},
    )
    monkeypatch.setattr(
        "openclaw_moe_orchestrator.gui.OllamaClient.list_model_entries",
        lambda self: [
            OllamaModelEntry(
                name="kimi-k2.5:cloud",
                family="kimi",
                remote_model="kimi-k2.5",
                remote_host="https://ollama.com",
                size=340,
            )
        ],
    )

    catalog = build_model_catalog(paths)

    assert any(item["model"] == "kimi-k2.5:cloud" for item in catalog["roles"]["reasoning"])
    assert any(item["source"] == "live" for item in catalog["roles"]["reasoning"])
