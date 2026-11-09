"""Check independent game imports, compatibility and replay source coverage."""

import ast
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from honing.backends.griddly import source
from honing.backends.griddly.replay import audit_run
from honing.backends.rea_issac.match import runtime_fingerprint
from integrations.rea_issac import engine as rea_engine


@pytest.mark.unit
@pytest.mark.parametrize(
    "game, engine_name", [("craftax", "CraftaxEngine"), ("griddly", "GriddlyEngine"), ("rea_issac", "Engine")]
)
def test_old_engine_import_is_the_same_class(game: str, engine_name: str) -> None:
    old = importlib.import_module(f"honing.backends.{game}.engine")
    new = importlib.import_module(f"integrations.{game}.engine")
    assert getattr(old, engine_name) is getattr(new, engine_name)


@pytest.mark.integration
def test_games_import_without_honing_or_native_dependencies(repo_root: Path) -> None:
    code = """
import sys
from integrations.craftax.engine import CraftaxEngine
from integrations.griddly.engine import GriddlyEngine
from integrations.rea_issac.engine import Engine
from integrations.rea_issac.ui import Session
assert not any(name.split('.')[0] in {'honing', 'jax', 'craftax', 'griddly'} for name in sys.modules)
assert Engine(seed=7, max_ticks=1).step({})['status'] != 'running'
"""
    subprocess.run([sys.executable, "-c", code], cwd=repo_root, check=True)


@pytest.mark.unit
def test_game_runtime_has_no_honing_dependency(repo_root: Path) -> None:
    for game in ("craftax", "griddly", "rea_issac"):
        for path in (repo_root / "integrations" / game).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("honing") for alias in node.names), path
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("honing"), path


@pytest.mark.unit
def test_rea_fingerprint_reads_game_implementation() -> None:
    game = Path(rea_engine.__file__).parent
    assert runtime_fingerprint() == {
        name: hashlib.sha256((game / name).read_bytes()).hexdigest()
        for name in ("engine.py", "source_facts.json", "policy.py")
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "changed",
    [
        "integration/engine.py",
        "integration/catalog.py",
        "integration/__init__.py",
        "integration_namespace.py",
        "model.py",
    ],
)
def test_griddly_audit_rejects_changed_runtime_snapshot(tmp_path: Path, changed: str) -> None:
    files = source.source_files()
    assert changed in files
    for name, path in files.items():
        target = tmp_path / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps({"adapter_sha256": source.adapter_sha256()}))
    with (tmp_path / "source" / changed).open("a") as stream:
        stream.write("\n# altered runtime\n")
    with pytest.raises(ValueError, match="source snapshot"):
        audit_run(tmp_path)


@pytest.mark.unit
def test_griddly_digest_changes_with_runtime_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = source.source_files()
    copied = {}
    for name, path in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copied[name] = target
    monkeypatch.setattr(source, "source_files", lambda: copied)
    before = source.adapter_sha256()
    with copied["integration/engine.py"].open("a") as stream:
        stream.write("\n# changed implementation\n")
    assert source.adapter_sha256() != before
