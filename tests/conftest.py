"""Shared fixtures: the repository root and the scripts that live outside the package."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

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


def require(name: str, reason: str) -> None:
    """Skip, or fail when ``HONING_REQUIRE_<NAME>=1``: a CI job that exists to run the test cannot skip it silently."""
    if os.environ.get(f"HONING_REQUIRE_{name.upper()}") == "1":
        pytest.fail(f"HONING_REQUIRE_{name.upper()}=1, but {reason}")
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def close_sglang_endpoints(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Close every ``SGLangPolicyEndpoints`` a test leaves open (#154).

    The map's owner thread is a daemon so that a process exits without waiting for it. If it were not, the maps
    tests leave open would hang the run at exit rather than fail the test that checks it. Patches only when a test
    module has imported the map already, so no test gains an import.
    """
    module = sys.modules.get("honing.backends.sglang.endpoints")
    if module is None:
        yield
        return
    opened: list[Any] = []
    init = module.SGLangPolicyEndpoints.__init__

    def recording_init(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(module.SGLangPolicyEndpoints, "__init__", recording_init)
    yield
    for endpoints in opened:
        endpoints.close()


#: The kind every test declares with exactly one marker (CONTRIBUTING.md, Tests).
KINDS = ("unit", "functional", "integration", "gpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuse a test that declares no kind or two, so every test says what it is and where it runs."""
    wrong = [item.nodeid for item in items if sum(item.get_closest_marker(kind) is not None for kind in KINDS) != 1]
    if wrong:
        listed = "\n  ".join(wrong[:20]) + (f"\n  ... and {len(wrong) - 20} more" if len(wrong) > 20 else "")
        raise pytest.UsageError(
            f"{len(wrong)} tests do not declare exactly one kind of {', '.join(KINDS)}:\n  {listed}"
        )


def _refuse(what: str) -> Any:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"a unit test {what}: mark it functional or integration, or use a fake (tests/fakes)")

    return refuse


@pytest.fixture(autouse=True)
def unit_tests_start_no_process_and_open_no_socket(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A test marked ``unit`` that starts a process or connects or binds a socket fails, so the marker stays
    true. A fake the test installs itself takes precedence, since the test's own patches come after this one."""
    if request.node.get_closest_marker("unit") is None:
        return
    import socket
    import subprocess

    monkeypatch.setattr(subprocess.Popen, "_execute_child", _refuse("started a process"))
    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, _refuse("opened a socket"))
