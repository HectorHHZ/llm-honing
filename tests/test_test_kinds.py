"""The test kinds (CONTRIBUTING.md, Tests): every test declares exactly one, and a unit test stays a unit test."""

from __future__ import annotations

import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.conftest import KINDS, pytest_collection_modifyitems


def item(nodeid: str, *kinds: str) -> SimpleNamespace:
    return SimpleNamespace(nodeid=nodeid, get_closest_marker=lambda name: name if name in kinds else None)


@pytest.mark.unit
def test_a_unit_test_cannot_start_a_process() -> None:
    with pytest.raises(AssertionError, match="a unit test started a process: mark it functional or integration"):
        subprocess.run([sys.executable, "-c", "pass"], check=True)


@pytest.mark.unit
@pytest.mark.parametrize("call", ["bind", "connect"])
def test_a_unit_test_cannot_open_a_socket(call: str) -> None:
    with socket.socket() as s, pytest.raises(AssertionError, match="a unit test opened a socket"):
        getattr(s, call)(("127.0.0.1", 0))


@pytest.mark.functional
def test_a_functional_test_may_start_a_process() -> None:
    assert subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True).stdout == "ok\n"


@pytest.mark.unit
def test_a_fake_the_test_installs_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert subprocess.run(["anything"]).returncode == 0


@pytest.mark.unit
def test_every_test_declares_exactly_one_kind() -> None:
    pytest_collection_modifyitems(None, [item(f"t{kind}", kind) for kind in KINDS])  # type: ignore[arg-type]
    for wrong in (item("none"), item("two", "unit", "functional")):
        with pytest.raises(pytest.UsageError, match=f"1 tests do not declare exactly one kind .*\n  {wrong.nodeid}"):
            pytest_collection_modifyitems(None, [item("fine", "unit"), wrong])  # type: ignore[arg-type]
