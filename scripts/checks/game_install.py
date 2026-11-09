"""Run with python -I after installing the wheel in a clean environment."""

import importlib
import importlib.metadata
import sys
import tempfile
from pathlib import Path


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
    from honing.backends.rea_issac.harness import seed_harness
    from honing.backends.rea_issac.match import run_match
    from honing.backends.rea_issac.replay import audit_run
    from honing.backends.rea_issac.scenarios import ReaIssacScenario
    from integrations.rea_issac import engine
    from integrations.rea_issac.ui import ROOT

    assert len(adapter_sha256()) == 64
    assert all(path.is_file() for path in source_files().values())
    assert (ROOT / "web/index.html").is_file()
    for game in ("craftax", "griddly", "rea_issac"):
        backend = importlib.import_module(f"honing.backends.{game}.harness")
        package = importlib.import_module(f"integrations.{game}")
        backend.seed_harness(Path(package.__file__).parent)
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "episode"
        result = run_match(ReaIssacScenario(max_ticks=6), run, seed_harness(Path(engine.__file__).parent))
        _, _, audited = audit_run(run)
        assert result == audited
    print("Installed game imports, resources, native REA-Issac episode and replay: passed")


if __name__ == "__main__":
    main()
