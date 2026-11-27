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


@pytest.mark.unit
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
        "honing/campaign/cost.py:resolve",  # MeteredPolicyEndpoints, a map: runs where its own resolve is called
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


# Integration directories and backend packages that serve every environment rather than being one.
# One set for both, so a shared name such as reef stays shared when it gains a backend package.
SHARED = {"_shared", "agent_gateway", "djev", "jev", "llm_gateway", "openhands", "pi", "reef", "sglang", "slime"}


def environment_names(repo_root: Path) -> re.Pattern[str]:
    """Every environment's name, from its integration directory and its backend package, as one pattern."""
    integrations = {
        path.name
        for path in (repo_root / "integrations").iterdir()
        if path.is_dir() and not path.name.startswith(("_", ".")) and path.name not in SHARED
    }
    backends = {
        path.name
        for path in (repo_root / "honing" / "backends").iterdir()
        if (path / "__init__.py").is_file() and path.name not in SHARED
    }
    spelled = {re.sub(r"[-_]", "[-_ ]?", name) for name in integrations | backends}
    return re.compile(r"\b(" + "|".join(sorted(spelled, key=len, reverse=True)) + r")\b", re.IGNORECASE)


@pytest.mark.unit
def test_core_names_no_environment(repo_root: Path) -> None:
    """``honing/core`` serves every game and task, so no environment's name or vocabulary belongs in it."""
    names = environment_names(repo_root)
    assert names.search("from honing.backends.tank import TankGrader")
    assert names.search("a Tank War round") and names.search("rea_issac") and not names.search("pipeline")
    hits = [
        f"{path.relative_to(repo_root)}:{number}: {line.strip()}"
        for path in sorted((repo_root / "honing" / "core").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if names.search(line)
    ]
    assert hits == [], "move environment-specific code to its backend package:\n" + "\n".join(hits)


def lazy_exports(init: Path) -> tuple[dict[str, str], list[str], dict[str, str]] | None:
    """A PEP 562 package's ``_MODULES`` table, its ``__all__``, and its ``TYPE_CHECKING`` imports, or None."""
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    table: dict[str, str] | None = None
    exported: list[str] = []
    typed: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "_MODULES":
                table = ast.literal_eval(node.value)
            elif node.targets[0].id == "__all__":
                exported = list(ast.literal_eval(node.value))
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for statement in node.body:
                if isinstance(statement, ast.ImportFrom) and statement.module:
                    typed.update((alias.asname or alias.name, statement.module) for alias in statement.names)
    return None if table is None else (table, exported, typed)


@pytest.mark.unit
def test_a_lazy_package_lists_each_export_in_its_table_all_and_type_imports(repo_root: Path) -> None:
    """A package that imports its names on first use lists each in three places, and they must agree."""
    checked = []
    for init in sorted((repo_root / "honing").rglob("__init__.py")):
        exports = lazy_exports(init)
        if exports is None:
            continue
        table, exported, typed = exports
        package = ".".join(init.parent.relative_to(repo_root).parts)
        assert sorted(exported) == sorted(set(exported)), f"{package}: a name is in __all__ twice"
        assert sorted(table) == sorted(exported), f"{package}: _MODULES and __all__ differ"
        assert typed == {
            name: f"{package}.{module}" for name, module in table.items()
        }, f"{package}: the TYPE_CHECKING imports differ from _MODULES"
        checked.append(package)
    assert "honing.backends.tank" in checked
