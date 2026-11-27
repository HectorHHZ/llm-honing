"""Dependency and vocabulary boundaries, checked mechanically so they cannot erode quietly.

- ``honing.core`` imports only the standard library, so every plane can depend
  on it without pulling in a framework.
- Only ``honing/backends/`` may import a training framework. Slime is the one
  training dependency.
- Reef runs as its own service (#137). Only ``honing/backends/reef/`` imports
  its client, ``reef_client``, and nothing imports ``reef`` itself.
- ``honing/backends/hf_checkpoint.py``, which every backend uses to name a
  checkpoint, imports only the standard library and ``honing.core``.
- Importing the decision-trace loader (``honing.backends.tank.traces``) or
  renderer (``honing.harness.traces``) loads, besides the standard library and
  ``honing.core``, only the modules each names: the loader its package and the
  tank decision-log parser, never the runner, grader, rollouts or
  demonstrations.
- Only ``honing/harness/models.py`` imports litellm, an optional extra whose
  pin is reviewed on its own.
- ``honing/backends/slime/sandbox.py`` imports only the standard library, so it
  also runs inside Slime's image, where Honing's own dependencies may be absent.
- Only ``honing/e2e/tracking.py`` imports wandb, another optional extra, and
  only when a run asks to be logged.
- Retired design terms stay out of code and the specification.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

FRAMEWORKS = {"slime", "slime_plugins", "megatron", "sglang", "reef", "reef_client", "openhands"}
RETIRED_TERMS = re.compile(r"\b(provenance|evidence|ledger|sidecar)\b", re.IGNORECASE)


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


@pytest.mark.unit
def test_only_the_reef_backend_imports_reef_client(repo_root: Path) -> None:
    package = repo_root / "honing"
    offenders = {
        str(path.relative_to(repo_root)): sorted(names & {"reef", "reef_client"})
        for path in sorted(package.rglob("*.py"))
        if (names := imported_top_levels(path)) & {"reef", "reef_client"}
        and ("reef" in names or path.relative_to(package).parts[:2] != ("backends", "reef"))
    }
    assert offenders == {}, f"Reef imports outside honing/backends/reef/, or of reef itself: {offenders}"


def test_the_checkpoint_identity_imports_no_backend(repo_root: Path) -> None:
    """Backends name a Hugging Face checkpoint through hf_checkpoint, so it must not import any of them."""
    path = repo_root / "honing" / "backends" / "hf_checkpoint.py"
    assert imported_top_levels(path) <= set(sys.stdlib_module_names) | {"honing"}
    honing_imports = [
        node.module
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("honing")
    ]
    assert honing_imports and all(name == "honing.core" for name in honing_imports)


# Each trace reader, and the Honing modules outside honing.core that importing it may load.
TRACE_READERS = {
    "honing.backends.tank.traces": {
        "honing",
        "honing.backends",
        "honing.backends.tank",
        "honing.backends.tank.decisions",
        "honing.backends.tank.traces",
    },
    "honing.harness.traces": {"honing", "honing.harness", "honing.harness.traces"},
}


@pytest.mark.integration
@pytest.mark.parametrize("module", sorted(TRACE_READERS))
def test_importing_a_trace_reader_loads_only_core_and_the_modules_it_names(repo_root: Path, module: str) -> None:
    """GEPA, Meta-Harness and System 2 read traces through these, so importing one must not load a plane or the
    rest of a backend. A fresh interpreter imports it, and every module that import loaded is checked."""
    script = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        f"import {module}\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    run = subprocess.run([sys.executable, "-c", script], cwd=repo_root, capture_output=True, text=True, check=True)
    loaded = set(json.loads(run.stdout))

    honing = {name for name in loaded if name.split(".")[0] == "honing"}
    core = {name for name in honing if name == "honing.core" or name.startswith("honing.core.")}
    assert honing - core == TRACE_READERS[module]
    assert {name.split(".")[0] for name in loaded - honing} <= set(sys.stdlib_module_names)


@pytest.mark.unit
def test_every_public_tank_name_is_listed_alike_in_all_three_places_and_resolves(repo_root: Path) -> None:
    """``honing.backends.tank`` names each public name in its ``TYPE_CHECKING`` imports, in the table its lazy
    ``__getattr__`` reads, and in ``__all__``. A merge that adds a name to one place can miss another."""
    tank = importlib.import_module("honing.backends.tank")
    tree = ast.parse((repo_root / "honing/backends/tank/__init__.py").read_text(encoding="utf-8"))
    checked = {
        alias.asname or alias.name: statement.module.rsplit(".", 1)[-1]
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
        for statement in node.body
        if isinstance(statement, ast.ImportFrom) and statement.module
        for alias in statement.names
    }
    table = tank._MODULES

    assert checked == table
    assert sorted(tank.__all__) == sorted(table)
    for name, module in table.items():
        assert getattr(tank, name) is getattr(importlib.import_module(f"honing.backends.tank.{module}"), name)


@pytest.mark.unit
def test_only_the_chat_client_imports_litellm(repo_root: Path) -> None:
    package = repo_root / "honing"
    importers = [
        str(path.relative_to(repo_root))
        for path in sorted(package.rglob("*.py"))
        if "litellm" in imported_top_levels(path)
    ]
    assert importers == ["honing/harness/models.py"]


@pytest.mark.unit
def test_the_slime_sandbox_imports_only_the_standard_library(repo_root: Path) -> None:
    path = repo_root / "honing" / "backends" / "slime" / "sandbox.py"
    assert imported_top_levels(path) - set(sys.stdlib_module_names) == set()


@pytest.mark.unit
def test_only_the_run_tracker_imports_wandb(repo_root: Path) -> None:
    package = repo_root / "honing"
    importers = [
        str(path.relative_to(repo_root))
        for path in sorted(package.rglob("*.py"))
        if "wandb" in imported_top_levels(path)
    ]
    assert importers == ["honing/e2e/tracking.py"]


@pytest.mark.unit
def test_honing_never_imports_openhands(repo_root: Path) -> None:
    """OpenHands runs in its own environment; its backend only starts the worker there."""
    offenders = sorted(
        str(path.relative_to(repo_root))
        for path in (repo_root / "honing").rglob("*.py")
        if "openhands" in imported_top_levels(path)
    )
    assert offenders == []


def resolve_calls(path: Path) -> list[str]:
    """The function around each ``x.resolve(...)`` call with an argument in ``path`` (``Path.resolve()`` takes
    none), or ``<module>``."""
    found: list[str] = []

    def visit(node: ast.AST, function: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and node.args
        ):
            found.append(function)
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(ast.parse(path.read_text(encoding="utf-8")), "<module>")
    return found


@pytest.mark.unit
def test_runners_resolve_a_policy_only_through_resolve_endpoint(repo_root: Path) -> None:
    """A resolve can start a server for minutes, so a runner on an event loop calls it through
    ``resolve_endpoint``, on a thread (#154). The other calls run on no loop."""
    calls = sorted(
        f"{path.relative_to(repo_root)}:{function}"
        for path in (repo_root / "honing").rglob("*.py")
        for function in resolve_calls(path)
    )
    assert calls == [
        "honing/backends/ra2/duel.py:_url",  # in the synchronous evaluate, before its asyncio.run
        "honing/backends/sglang/endpoints.py:<module>",  # the command-line entry point
        "honing/harness/endpoints.py:_resolve_on_thread",  # resolve_endpoint's thread
    ]


@pytest.mark.unit
def test_retired_terms_stay_out_of_code_and_the_specification(repo_root: Path) -> None:
    scanned = [
        *sorted((repo_root / "honing").rglob("*.py")),
        *sorted((repo_root / "system-design").glob("*.md")),
        *sorted((repo_root / "system-design").glob("*.py")),
        repo_root / "README.md",
    ]
    hits = [
        f"{path.relative_to(repo_root)}:{number}: {line.strip()}"
        for path in scanned
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if RETIRED_TERMS.search(line)
    ]
    assert hits == [], "use 'manifest', 'record', 'metadata', or 'importance' instead:\n" + "\n".join(hits)
