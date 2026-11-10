"""Check independent game imports, compatibility and replay source coverage."""

import ast
import hashlib
import importlib
import json
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from honing.backends.griddly import source
from honing.backends.griddly.replay import audit_run
from honing.backends.rea_issac.source import runtime_fingerprint
from honing.backends.rea_issac.source import source_files as rea_source_files
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
    files = rea_source_files()
    assert files["engine.py"] == game / "engine.py"
    assert files["source_facts.json"] == game / "source_facts.json"
    assert files["policy.py"] == game / "policy.py"
    assert "adapter/model.py" in files
    assert runtime_fingerprint() == {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "changed",
    [
        "integration/engine.py",
        "integration/catalog.py",
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


@pytest.mark.unit
@pytest.mark.parametrize("game", ["griddly", "rea_issac"])
def test_source_identity_ignores_compatibility_and_initializers(
    game: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"honing.backends.{game}.source")
    adapter = tmp_path / "adapter"
    runtime = tmp_path / "runtime"
    adapter.mkdir()
    runtime.mkdir()
    for origin, target in ((Path(module.__file__).parent, adapter), (Path(module.engine.__file__).parent, runtime)):
        for path in origin.glob("*"):
            if path.is_file() and path.suffix in (".py", ".json"):
                (target / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(module, "__file__", str(adapter / "source.py"))
    monkeypatch.setattr(module.engine, "__file__", str(runtime / "engine.py"))
    fingerprint = module.adapter_sha256 if game == "griddly" else module.runtime_fingerprint
    before = fingerprint()
    aliases = ["engine.py", "catalog.py"] if game == "griddly" else ["engine.py", "policy.py", "ui.py"]
    for name in aliases:
        (adapter / name).unlink()
    for path in (adapter / "__init__.py", runtime / "__init__.py", runtime.parent / "__init__.py"):
        path.write_text('"""Changed package description."""\n')
    assert fingerprint() == before
    for path in (runtime / "engine.py", adapter / "model.py"):
        original = path.read_bytes()
        path.write_bytes(original + b"\n# changed play code\n")
        assert fingerprint() != before
        path.write_bytes(original)


@pytest.mark.unit
def test_excluded_game_initializers_only_have_docstrings(repo_root: Path) -> None:
    for path in [
        repo_root / "integrations/__init__.py",
        *(repo_root / "integrations" / game / "__init__.py" for game in ("craftax", "griddly", "rea_issac")),
    ]:
        nodes = ast.parse(path.read_text()).body
        assert len(nodes) == 1 and isinstance(nodes[0], ast.Expr)
        assert isinstance(nodes[0].value, ast.Constant) and isinstance(nodes[0].value.value, str)


@pytest.mark.integration
@pytest.mark.parametrize("module_name", ["honing.backends.rea_issac.ui", "integrations.rea_issac.ui"])
def test_ui_root_assignment_controls_artifacts_and_http_assets(
    module_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ui = importlib.import_module(module_name)
    monkeypatch.setattr(ui, "ROOT", tmp_path)
    (tmp_path / "web").mkdir()
    (tmp_path / "web/index.html").write_text("isolated game assets")
    session = ui.Session(7)
    server = ThreadingHTTPServer(("127.0.0.1", 0), ui.handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert session.run.is_relative_to(tmp_path / "runs")
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5) as response:
            assert response.read() == b"isolated game assets"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        session.log.close()


@pytest.mark.unit
@pytest.mark.parametrize("game", ["craftax", "griddly", "rea_issac"])
@pytest.mark.parametrize("explicit_out", [False, True])
def test_match_cli_reads_runtime_resources_and_writes_outside_package(
    game: str, explicit_out: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    match = importlib.import_module(f"honing.backends.{game}.match")
    # Resource discovery must not depend on the backend's directory depth.
    monkeypatch.setattr(match, "__file__", str(tmp_path / "relocated/backend/match.py"))
    monkeypatch.chdir(tmp_path)
    seen = []

    def run(scenario, out, spec, **kwargs):
        assert spec.files
        assert out == tmp_path / "explicit" if explicit_out else out.parent == tmp_path / "runs" / game
        seen.append(out)
        return {}

    monkeypatch.setattr(match, "run_match", run)
    match.main(["--out", str(tmp_path / "explicit")] if explicit_out else [])
    assert len(seen) == 1
