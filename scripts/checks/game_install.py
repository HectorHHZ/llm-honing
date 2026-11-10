"""Run with python -I after installing the wheel in a clean environment."""

import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    files = importlib.metadata.files("llm-honing")
    assert files is not None
    game_files = [path.parts for path in files if path.parts[0] == "integrations"]
    assert game_files
    assert all(parts[1] in {"__init__.py", "__pycache__", "craftax", "griddly", "rea_issac"} for parts in game_files)
    assert any(str(path) == "honing/harness/methods/meta_harness/SKILL.md" for path in files)
    for game in ("craftax", "griddly", "rea_issac"):
        importlib.import_module(f"integrations.{game}.engine")
    assert "honing" not in sys.modules
    assert not {"jax", "craftax", "griddly"} & sys.modules.keys()

    from honing.backends.griddly.source import adapter_sha256, source_files
    from honing.backends.rea_issac.replay import audit_run
    from integrations.rea_issac.ui import ROOT

    assert len(adapter_sha256()) == 64
    assert all(path.is_file() for path in source_files().values())
    assert (ROOT / "web/index.html").is_file()
    for game in ("craftax", "griddly", "rea_issac"):
        backend = importlib.import_module(f"honing.backends.{game}.harness")
        package = importlib.import_module(f"integrations.{game}")
        backend.seed_harness(Path(package.__file__).parent)
    with tempfile.TemporaryDirectory() as directory:
        cwd = Path(directory).resolve()
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "honing.backends.rea_issac.match", "--max-ticks", "6"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(completed.stdout.strip().splitlines()[-1])
        run = Path(report["out"])
        assert run.parent == cwd / "runs/rea_issac"
        _, _, audited = audit_run(run)
        assert audited["status"] == "truncated"
        # Native packages are optional here. Native CI tests exercise these CLIs
        # with the real engines; these stubs check installed resource/output paths.
        for game in ("craftax", "griddly"):
            match = importlib.import_module(f"honing.backends.{game}.match")
            called = []

            def run_stub(scenario, out, spec, *, expected_game=game, calls=called, **kwargs):
                assert spec.files
                assert out.parent == cwd / "runs" / expected_game
                calls.append(out)
                return {}

            previous = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.object(match, "run_match", run_stub), redirect_stdout(io.StringIO()):
                    match.main([])
            finally:
                os.chdir(previous)
            assert len(called) == 1
    assert not (ROOT / "runs").exists()
    print("Installed CLI resources/output paths and native REA-Issac CLI replay: passed")


if __name__ == "__main__":
    main()
