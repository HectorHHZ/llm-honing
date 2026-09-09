"""Shared fixtures: the repository root and the scripts that live outside the package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str) -> ModuleType:
    """Import a standalone script by path.

    Repository automation under ``.github/scripts`` is deliberately not part of
    the installed package, so it has no import path. Loading it by location is
    what lets those scripts carry contract tests like any other code.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
