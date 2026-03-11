from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openclaw_moe_orchestrator.openclaw_local import (  # noqa: E402
    OpenClawLocalLayout,
    collect_openclaw_local_status,
    install_openclaw_local_bundle,
)
from openclaw_moe_orchestrator.llm import ModelRole  # noqa: E402
from openclaw_moe_orchestrator.paths import RepoPaths  # noqa: E402


def test_install_openclaw_local_bundle_writes_expected_files(tmp_path: Path) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    state_dir = tmp_path / ".openclaw"

    result = install_openclaw_local_bundle(paths, state_dir=state_dir)

    layout = OpenClawLocalLayout.discover(state_dir)
    assert Path(result["workspace_skill"]).is_dir()
    assert layout.auth_profiles_path.exists()
    assert layout.overlay_config_path.exists()
    assert layout.config_path.exists()
    assert layout.main_session_dir.exists()
    assert Path(result["active_config"]) == layout.config_path

    auth_profiles = json.loads(layout.auth_profiles_path.read_text())
    assert auth_profiles["profiles"]["ollama:default"]["provider"] == "ollama"
    assert auth_profiles["profiles"]["ollama:default"]["key"] == "ollama-cloud"
    overlay = json.loads(layout.overlay_config_path.read_text())
    active = json.loads(layout.config_path.read_text())
    assert overlay["gateway"]["mode"] == "local"
    assert overlay["gateway"]["auth"]["token"]
    assert active == overlay
    assert overlay["auth"]["order"]["ollama"] == ["ollama:default"]
    assert overlay["agents"]["defaults"]["sandbox"]["mode"] == "off"
    assert overlay["agents"]["defaults"]["memorySearch"]["enabled"] is False
    assert overlay["models"]["providers"]["ollama"]["api"] == "ollama"
    assert overlay["models"]["providers"]["ollama"]["models"]


def test_install_openclaw_local_bundle_updates_active_config_and_keeps_backup(tmp_path: Path) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    state_dir = tmp_path / ".openclaw"
    layout = OpenClawLocalLayout.discover(state_dir)
    layout.ensure_directories()
    layout.config_path.write_text("{\"stale\":true}")

    result = install_openclaw_local_bundle(paths, state_dir=state_dir)

    backup_path = layout.config_path.with_suffix(".json.bak")
    assert result["active_config_backup"] == str(backup_path)
    assert backup_path.read_text() == "{\"stale\":true}"
    assert json.loads(layout.config_path.read_text())["models"]["providers"]["ollama"]["api"] == "ollama"


def test_install_openclaw_local_bundle_applies_role_model_overrides(tmp_path: Path) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    state_dir = tmp_path / ".openclaw"

    result = install_openclaw_local_bundle(
        paths,
        state_dir=state_dir,
        role_model_overrides={
            ModelRole.REASONING: ("qwen3-next:80b-cloud", "gpt-oss:120b-cloud"),
            ModelRole.CODING: ("qwen3-coder-next:cloud",),
            ModelRole.GENERAL: ("gpt-oss:120b-cloud",),
        },
    )

    layout = OpenClawLocalLayout.discover(state_dir)
    overlay = json.loads(layout.overlay_config_path.read_text())
    assert overlay["agents"]["defaults"]["model"]["primary"] == "ollama/qwen3-next:80b-cloud"
    assert overlay["agents"]["defaults"]["model"]["fallbacks"] == [
        "ollama/gpt-oss:120b-cloud",
        "ollama/qwen3-coder-next:cloud",
    ]
    assert result["role_model_overrides"]["reasoning"] == ["qwen3-next:80b-cloud", "gpt-oss:120b-cloud"]


def test_collect_openclaw_local_status_reports_local_files(tmp_path: Path) -> None:
    paths = RepoPaths.discover(REPO_ROOT)
    state_dir = tmp_path / ".openclaw"
    install_openclaw_local_bundle(paths, state_dir=state_dir)

    status = collect_openclaw_local_status(paths, state_dir=state_dir)

    assert status["auth_profiles_exists"] is True
    assert status["overlay_exists"] is True
    assert status["workspace_skill_exists"] is True
    assert status["manifest_exists"] is True
