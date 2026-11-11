"""LocalContainerSandbox against slime's ``Sandbox`` protocol at the pinned commit.

slime does not pip-install on macOS, so these tests import ``slime.agent.sandbox``
from a checkout on ``PYTHONPATH`` (``scripts/fetch-slime-agent.sh``) and skip without
one; the ``test-sandbox`` CI job sets ``HONING_REQUIRE_SLIME=1`` so they cannot skip
there. ``isinstance`` on a ``runtime_checkable`` protocol checks only that the names
exist, so the signatures are compared member by member as well.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from honing.backends.slime import LocalContainerSandbox
from tests.conftest import require

MEMBERS = {"sandbox_id", "__aenter__", "__aexit__", "exec", "write_file", "read_file"}
METHODS = sorted(MEMBERS - {"sandbox_id"})


@pytest.fixture(scope="module")
def slime_sandbox() -> ModuleType:
    try:
        import slime.agent.sandbox as module  # type: ignore[import-not-found]
    except ImportError:
        require("slime", "slime.agent.sandbox is not importable; run scripts/fetch-slime-agent.sh and set PYTHONPATH")
    return module


@pytest.mark.unit
def test_the_checkout_is_the_pinned_commit(slime_sandbox: ModuleType, repo_root: Path) -> None:
    # The fetch script checks the commit; this catches a PYTHONPATH that points somewhere else.
    source = Path(slime_sandbox.__file__).read_text(encoding="utf-8")
    assert "async def exec_and_wait(" in source
    assert "async def ensure_agent_user(" in source


@pytest.mark.unit
def test_the_protocol_has_the_members_the_sandbox_implements(slime_sandbox: ModuleType) -> None:
    protocol = slime_sandbox.Sandbox
    declared = set(protocol.__annotations__) | {
        name
        for name, value in vars(protocol).items()
        if inspect.isfunction(value) and value.__qualname__.startswith("Sandbox.")
    }
    assert declared == MEMBERS


@pytest.mark.unit
def test_the_sandbox_is_an_instance_of_the_protocol(slime_sandbox: ModuleType) -> None:
    assert isinstance(LocalContainerSandbox("any-image"), slime_sandbox.Sandbox)


@pytest.mark.unit
@pytest.mark.parametrize("name", METHODS)
def test_each_method_has_the_protocols_signature(slime_sandbox: ModuleType, name: str) -> None:
    ours, theirs = getattr(LocalContainerSandbox, name), getattr(slime_sandbox.Sandbox, name)
    assert inspect.iscoroutinefunction(ours) == inspect.iscoroutinefunction(theirs)

    def shape(function: object) -> list[tuple[str, object, object]]:
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(function).parameters.values()  # type: ignore[arg-type]
        ]

    assert shape(ours) == shape(theirs)


@pytest.mark.integration  # starts a subprocess
def test_the_module_loads_with_only_the_standard_library(repo_root: Path) -> None:
    # Every import outside the standard library is refused, Honing's own included, and the
    # file is loaded by path: a copy of it runs in Slime's image whatever is installed there.
    code = f"""
import importlib.abc, importlib.util, sys

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in sys.stdlib_module_names:
            raise ImportError(f"refused {{name}}")
        return None

sys.meta_path.insert(0, Refuse())
spec = importlib.util.spec_from_file_location("sandbox", {str(repo_root / "honing/backends/slime/sandbox.py")!r})
module = sys.modules["sandbox"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.LocalContainerSandbox("image").create_argv()[1])
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "create"
