"""A tank round's raw states, read back through the integration's Node tool.

``integrations/tank-war/scripts/match.mjs`` writes ``states.jsonl`` next to
``decisions.jsonl``. Its first line names the format and version and the
platform that recorded the round, and holds the harness files the round was
played under. Each further line is one agent
decision: the raw game state the controller built the decision's context from,
the state's SHA-256, and the motion the weapon question was asked about. When
the round ends, the run's manifest records the SHA-256 of the whole file with
the other outputs, and :func:`read_states` refuses a file that no longer
matches it.

:class:`TankRawStates` drives ``integrations/tank-war/scripts/raw-states.mjs``,
one Node process per call, so the context is rendered by the same code a match
runs:

- :meth:`~TankRawStates.render` re-renders logged states under any harness.
- :meth:`~TankRawStates.round_trip` checks that the logged harness reproduces
  every logged ``contextParts`` byte for byte.
- :meth:`~TankRawStates.replay` plays a round again from its seed with the
  logged choices and checks each decision's state hash. This is bit exact on
  the platform that recorded the round, and a mismatch on another platform
  names both.
- :meth:`~TankRawStates.label` writes ``labels.jsonl``, the planner teacher's
  answer to every question of every row.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from honing.backends.tank.traces import STATES_FILE, verify_outputs
from honing.core import ContractError, HarnessSpec

LABELS_FILE = "labels.jsonl"
STATES_FORMAT = "tank-raw-states"
STATES_VERSION = 1


class TankRequest(TypedDict):
    """One request a tank decision sends to ``/v1/decide``: ``parts`` is its ``contextParts``."""

    id: str
    parts: dict[str, list[str]]
    questions: dict[str, Any]


def _absolute(path: Path) -> str:
    """``path`` relative to the caller's directory, spelled out for a tool that runs in another one."""
    return str(Path(path).absolute())


def harness_files(spec: HarnessSpec) -> list[dict[str, str]]:
    """A harness as the tool and ``states.jsonl`` hold it: every file's path, kind and text."""
    return [{"path": path, "kind": file.kind, "content": file.content} for path, file in spec.files]


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_states(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the decision lines of a run's ``states.jsonl``.

    Another format or version is refused, and so is a file that no longer matches the SHA-256 its manifest records.
    """
    path = run / STATES_FILE
    if not path.exists():
        raise ContractError(f"{run} has no {STATES_FILE}: it was recorded before raw states were logged")
    header, *states = _lines(path)
    if header.get("format") != STATES_FORMAT or header.get("version") != STATES_VERSION:
        raise ContractError(
            f"{path} is {header.get('format')} version {header.get('version')}, and this reads"
            f" {STATES_FORMAT} version {STATES_VERSION}"
        )
    verify_outputs(run, (STATES_FILE,))
    return header, states


def state_at(run: Path, state_ref: str) -> dict[str, Any]:
    """The line of ``states.jsonl`` that a trace's ``state_ref``, ``states.jsonl:<line>``, names.

    The file must still match the SHA-256 the run's manifest records.
    """
    name, _, number = state_ref.partition(":")
    if name != STATES_FILE or not number.isdigit() or int(number) < 2:
        raise ContractError(f"{state_ref!r} names no decision line of {STATES_FILE}")
    _, states = read_states(run)
    if int(number) - 2 >= len(states):
        raise ContractError(f"{run}/{STATES_FILE} has no line {number}")
    return states[int(number) - 2]


def read_labels(run: Path, out: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The header and the per-row lines of a run's ``labels.jsonl``, in the run or where ``label(out=...)`` put it."""
    path = (run if out is None else out / run.name) / LABELS_FILE
    if not path.exists():
        raise ContractError(f"{path.parent} has no {LABELS_FILE}. Label the run first")
    header, *labels = _lines(path)
    return header, labels


class TankRawStates:
    """Re-render, replay and label tank rounds with ``scripts/raw-states.mjs`` in ``integration_dir``.

    Every call starts one ``node`` process and waits at most ``timeout_seconds``
    for it. A process that fails, times out, or reports a mismatch raises
    :class:`~honing.core.ContractError` with the tool's own message, which
    names the run and the first decision that did not reproduce.
    """

    def __init__(self, integration_dir: Path, *, node: str = "node", timeout_seconds: float = 300.0) -> None:
        if timeout_seconds <= 0:
            raise ContractError("timeout_seconds must be positive")
        # The tool runs in the integration directory, so every path it is given is made absolute first.
        integration_dir = integration_dir.absolute()
        self._tool = integration_dir / "scripts/raw-states.mjs"
        if not self._tool.exists():
            raise ContractError(f"{integration_dir} has no scripts/raw-states.mjs")
        self._integration_dir = integration_dir
        self._node = node
        self._timeout_seconds = timeout_seconds

    def _run(self, arguments: Sequence[str], stdin: str = "") -> list[dict[str, Any]]:
        command = [self._node, str(self._tool), *arguments]
        try:
            process = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                cwd=self._integration_dir,
                timeout=self._timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ContractError(
                f"raw-states.mjs {arguments[0]} did not finish within {self._timeout_seconds:g} s"
            ) from None
        except OSError as error:
            raise ContractError(f"cannot start {self._node}: {error}") from None
        if process.returncode != 0:
            message = process.stderr.strip().splitlines()[-1:] or [f"exit code {process.returncode}"]
            raise ContractError(f"raw-states.mjs {arguments[0]} failed: {message[0]}")
        try:
            return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
        except json.JSONDecodeError as error:
            raise ContractError(f"raw-states.mjs {arguments[0]} printed something other than JSON: {error}") from None

    def render(
        self,
        observed: Sequence[Mapping[str, Any]],
        harness: HarnessSpec | Sequence[Mapping[str, str]],
        *,
        motions: Sequence[str | None] | None = None,
    ) -> list[list[TankRequest]]:
        """Every decision's requests under ``harness``, in one Node process.

        ``observed`` are decision lines of ``states.jsonl``. The weapon request
        of a planned-steering decision is for ``motions[i]`` when given, and
        otherwise for the motion its logged weapon question was asked about.
        """
        if motions is not None and len(motions) != len(observed):
            raise ContractError(f"{len(motions)} motions for {len(observed)} states")
        files = harness_files(harness) if isinstance(harness, HarnessSpec) else [dict(f) for f in harness]
        items = (
            {"observed": state, "harness": files, "motion": None if motions is None else motions[i]}
            for i, state in enumerate(observed)
        )
        replies = self._run(["render"], "".join(json.dumps(item) + "\n" for item in items))
        if len(replies) != len(observed):
            raise ContractError(f"raw-states.mjs rendered {len(replies)} of {len(observed)} states")
        for i, reply in enumerate(replies):
            if "error" in reply:
                raise ContractError(
                    f"state {i} (decision {observed[i].get('decision')}) did not render: {reply['error']}"
                )
        return [reply["requests"] for reply in replies]

    def round_trip(self, runs: Iterable[Path]) -> list[dict[str, Any]]:
        """Re-render every decision of each run under its logged harness, and refuse any row that differs."""
        return self._run(["roundtrip", *map(_absolute, runs)])

    def replay(self, run: Path, *, to: int | None = None) -> dict[str, Any]:
        """Play ``run`` again to decision ``to`` (or to the end, then check its events and result too)."""
        (report,) = self._run(["replay", _absolute(run), *(() if to is None else (f"--to={to}",))])
        return report

    def label(self, runs: Iterable[Path], *, out: Path | None = None) -> list[dict[str, Any]]:
        """Write each run's ``labels.jsonl``. On a teacher's round, labels that differ from its choices are refused.

        With ``out`` each run's labels go to ``out/<run's name>/labels.jsonl`` and the run is left unchanged.
        """
        return self._run(["label", *map(_absolute, runs), *([f"--out={_absolute(out)}"] if out else [])])


__all__ = [
    "LABELS_FILE",
    "STATES_FILE",
    "STATES_FORMAT",
    "STATES_VERSION",
    "TankRawStates",
    "TankRequest",
    "harness_files",
    "read_labels",
    "read_states",
    "state_at",
]
