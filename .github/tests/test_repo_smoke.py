from __future__ import annotations

import compileall
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_python_sources_compile() -> None:
    assert compileall.compile_dir(REPO_ROOT / "src", quiet=1)
    assert compileall.compile_dir(REPO_ROOT / "scripts", quiet=1)


def test_project_metadata_is_complete() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["name"] == "openclaw-moe-orchestrator"
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"]["openclaw-moe"] == "openclaw_moe_orchestrator.cli:main"
    assert "deepspeed==0.18.7" in project["dependencies"]
    assert "pytest==9.0.2" in project["optional-dependencies"]["dev"]


def test_lockfile_exists() -> None:
    assert (REPO_ROOT / "requirements" / "production.lock").exists()
    assert (REPO_ROOT / "requirements" / "dev.lock").exists()
