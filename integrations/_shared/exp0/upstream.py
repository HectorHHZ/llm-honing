"""Load the pinned shared Honing code the Exp 0 tools reuse, without installing anything.

- ``honing.core`` (``DecisionTrace``, ``TraceEvent``, ``DecisionSample``, ``DecisionQuestion``) comes from
  this repository (its root, or ``HONING_UPSTREAM``); these types are unchanged since ``UPSTREAM_SHA``.
- The #168 renderer ``render_traces`` is the exact ``honing/harness/traces.py`` of ``UPSTREAM_SHA``,
  vendored as ``vendor/traces_9d5327f.py`` and checked by its git blob id on load, so that diagnosis
  inputs render exactly as they did for the v0 releases; a later renderer of main is never used.
  Both import only the standard library.
- The #170 identity check ``identity_report`` comes from ``honing/harness/probe.py`` at the shared probe
  revision ``PROBE_REV`` (branch ``feat/counterfactual-probe``), as Griddly v0 used it. That object is not in
  the local clone, so its exact bytes were fetched with a GET-only GitHub API call into
  ``vendor/probe_a151596.py``; the git blob id is checked on load (a pin, as Griddly's ``git show`` did).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

UPSTREAM = Path(os.environ.get("HONING_UPSTREAM", Path(__file__).resolve().parents[3]))
UPSTREAM_SHA = "9d5327f1c5a53b8b134db847952ff6281267c0a6"
PROBE_REV = "a151596a329bc8c8f76e14de470614bd3b28d165"
PROBE_BLOB = "3b523bc1b5515b839f4f395a26694a1d3cdcf7b7"
PROBE_FILE = Path(__file__).resolve().parent / "vendor" / "probe_a151596.py"
TRACES_BLOB = "4140ac647c85669cbe4bca518054b6f50aecf721"
TRACES_FILE = Path(__file__).resolve().parent / "vendor" / "traces_9d5327f.py"


def _ensure_path() -> None:
    if not (UPSTREAM / "honing" / "core" / "__init__.py").is_file():
        raise FileNotFoundError(f"the Honing checkout is missing at {UPSTREAM}")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


def upstream_revision() -> str | None:
    """The checkout's HEAD (read-only ``git rev-parse``), or ``None`` when it is not a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def core() -> ModuleType:
    _ensure_path()
    import honing.core

    return honing.core


def traces() -> ModuleType:
    """The pinned #168 renderer (exact bytes of ``UPSTREAM_SHA:honing/harness/traces.py``), never main's."""
    core()
    source = TRACES_FILE.read_bytes()
    if _blob_id(source) != TRACES_BLOB:
        raise ValueError("vendor/traces_9d5327f.py is not the pinned #168 renderer revision")
    name = "_exp0_traces_9d5327f"
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{UPSTREAM_SHA}:honing/harness/traces.py", "exec"), module.__dict__)
    return module


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def shared_probe() -> ModuleType:
    """The pinned #170 module (exact bytes of ``PROBE_REV:honing/harness/probe.py``)."""
    core()
    source = PROBE_FILE.read_bytes()
    if _blob_id(source) != PROBE_BLOB:
        raise ValueError("vendor/probe_a151596.py is not the pinned #170 probe revision")
    name = "_exp0_shared_probe_a151596"
    if name in sys.modules:
        return sys.modules[name]
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"{PROBE_REV}:honing/harness/probe.py", "exec"), module.__dict__)
    return module


OPTIONS_SECTION = "offered options"


def decision_trace(
    episode_id: str,
    step: int,
    time_value: float,
    row: dict,
    question_key,
    events: tuple = (),
    constraints: str | None = None,
    applied: str | None = None,
    outcome: str = "",
    show_options: bool = False,
):
    """A contract model row as an upstream ``DecisionTrace`` (normalized probabilities, request order).

    ``question_key`` is the plan's question (a string) or, for a decision with several questions (contract 0.2.0),
    the list of the plan's question keys in request order; the trace then carries every question, its choice and
    its normalized probabilities, as ``honing.core.DecisionSample`` allows (Tank War's motion and weapon).
    With ``show_options`` the rendered state gets one more section, ``offered options``, holding each question's
    instructions and every option's text as the model saw them. The #168 renderer otherwise prints only the
    first decision's instructions and the option keys, while harnesses such as HK's put state into option texts.
    """
    c = core()
    request = row["request"]
    keys = [question_key] if isinstance(question_key, str) else list(question_key)
    several = len(keys) > 1
    choices = row["model_choice"] if several else {keys[0]: row["model_choice"]}
    probabilities = row["probabilities"] if several else {keys[0]: row["probabilities"]}
    parts = [(name, tuple(lines)) for name, lines in request["parts"].items()]
    if show_options:
        lines = []
        for key in keys:
            question = request["questions"][key]
            prefix = f"{key} " if several else ""
            lines.append(
                f"{prefix}Instructions: {question['instructions']}"
                if several
                else f"Instructions: {question['instructions']}"
            )
            lines.extend(f"{prefix}Option {option}: {text}" for option, text in question["criteria"].items())
        parts.append((OPTIONS_SECTION, tuple(lines)))
    questions, chosen, behavior = [], [], []
    for key in keys:
        question = request["questions"][key]
        options = list(question["criteria"])
        total = sum(probabilities[key][option] for option in options)
        questions.append(
            (key, c.DecisionQuestion("choice", question["instructions"], tuple(question["criteria"].items())))
        )
        chosen.append((key, choices[key]))
        behavior.append((key, tuple((option, probabilities[key][option] / total) for option in options)))
    sample = c.DecisionSample(
        parts=tuple(parts),
        questions=tuple(questions),
        chosen=tuple(chosen),
        behavior_probs=tuple(behavior),
        group=episode_id,
    )
    return c.DecisionTrace(
        episode_id=episode_id,
        step=step,
        time_seconds=float(time_value),
        decision=sample,
        events=events,
        constraints=constraints,
        applied=applied,
        outcome=outcome,
    )
