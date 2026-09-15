"""Dependency boundaries, checked mechanically so they cannot erode quietly.

- ``honing.core`` imports only the standard library, so every plane can depend
  on it without pulling in a framework.
- Only ``honing/backends/`` may import a training framework. Slime is the one
  dependency; Reef is a design reference and is never imported.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

FRAMEWORKS = {"slime", "slime_plugins", "megatron", "sglang", "reef", "reef_client"}


def imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.unit
def test_core_imports_only_the_standard_library(repo_root: Path) -> None:
    allowed = set(sys.stdlib_module_names) | {"honing"}
    offenders = {
        str(path.relative_to(repo_root)): sorted(imported_top_levels(path) - allowed)
        for path in sorted((repo_root / "honing" / "core").rglob("*.py"))
        if imported_top_levels(path) - allowed
    }
    assert offenders == {}, f"honing.core must import only the standard library: {offenders}"


@pytest.mark.unit
def test_core_imports_no_other_honing_package(repo_root: Path) -> None:
    offenders = [
        f"{path.relative_to(repo_root)} imports {node.module}"
        for path in sorted((repo_root / "honing" / "core").rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("honing.")
        and not node.module.startswith("honing.core")
    ]
    assert offenders == []


@pytest.mark.unit
def test_only_backend_packages_import_a_training_framework(repo_root: Path) -> None:
    package = repo_root / "honing"
    offenders = {
        str(path.relative_to(repo_root)): sorted(imported_top_levels(path) & FRAMEWORKS)
        for path in sorted(package.rglob("*.py"))
        if path.relative_to(package).parts[0] != "backends" and imported_top_levels(path) & FRAMEWORKS
    }
    assert offenders == {}, f"framework imports outside honing/backends/: {offenders}"
