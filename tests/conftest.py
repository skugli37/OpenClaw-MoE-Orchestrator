from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def load_module(monkeypatch):
    def _load(relative_path: str, *, module_name: str | None = None, injected_modules: dict[str, object] | None = None):
        path = REPO_ROOT / relative_path
        unique_name = module_name or path.stem

        for name, module in (injected_modules or {}).items():
            monkeypatch.setitem(sys.modules, name, module)

        if "." in unique_name:
            package_parts = unique_name.split(".")
            package_path = path.parent
            for index in range(1, len(package_parts)):
                package_name = ".".join(package_parts[:index])
                if package_name not in sys.modules:
                    package_module = types.ModuleType(package_name)
                    package_module.__path__ = [str(package_path)]
                    monkeypatch.setitem(sys.modules, package_name, package_module)
                package_path = package_path.parent

        spec = importlib.util.spec_from_file_location(unique_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        monkeypatch.setitem(sys.modules, unique_name, module)
        spec.loader.exec_module(module)
        return module

    return _load
