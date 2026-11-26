"""Exact save / restore of a raw-key harness's state as JSON, for the probe re-render of Exp 0 v0 (contract section 6).

A harness of the v9_keys family keeps everything that its next request depends on in its `Agent` object: the
latest view (built from the visible observation), the memory (rooms, remembered boxes and hazards, places stood
on, the last decisions, damage), and a few numbers (vertical speed, jump key state, frames played, last strike,
focus run, enemy tracks and speeds). `snapshot(agent)` writes all of it as one JSON object, taken right before
the decision is asked; `restore(state, harness)` builds an `Agent` with no game attached from it, and
`rerender(state, harness)` asks it again and returns the request exactly as the collector built it.

The codec is generic over plain Python state, so a later harness snapshot of the same family can use it
unchanged:
  * None / bool / int / str as they are; a float as it is (JSON keeps every double exactly), a non-finite one
    as {"__float__": "nan" | "inf" | "-inf"};
  * a list as a list; a tuple as {"__tuple__": [...]};
  * a dict with only plain string keys as a JSON object (insertion order kept); any other dict as
    {"__dict__": [[key, value], ...]} (order kept);
  * a set of (int, int) cells as {"__cells__": [[y, [[x0, x1], ...]], ...]} (runs per row, sorted; the harness
    uses these sets only for membership tests); any other set as {"__set__": [...]} sorted by its JSON text;
  * a `see.View` as {"__view__": <its visible observation>} (it is rebuilt from the observation);
  * any other object of the harness package (dataclasses and plain classes such as Box, Room, Memory) as
    {"__obj__": "<module>.<Class>", "fields": {...}}, restored without calling __init__.
Fields left out of the Agent (`EXCLUDE`): the game handle and two logs that the next request does not read
(`forecasts` is rewritten by `ask()`, `check_log` only feeds the run summary). Whether that is true is not
assumed: `probes.py selfcheck` and `selfcheck.py` re-render every logged request and compare the bytes.

Information boundary: everything in a snapshot comes from the visible observations of this run (what the screen and
HUD showed) and the harness's own memory of them; nothing from the debug trace.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
from typing import Any

FORMAT = "hk-v9-harness-state-v1"
EXCLUDE = ("env", "forecasts", "check_log")


def _cells(value) -> bool:
    return all(isinstance(c, tuple) and len(c) == 2 and type(c[0]) is int and type(c[1]) is int for c in value)


def _rle(cells) -> list:
    rows: dict[int, list[int]] = {}
    for x, y in cells:
        rows.setdefault(y, []).append(x)
    out = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        runs, start, last = [], xs[0], xs[0]
        for x in xs[1:]:
            if x == last + 1:
                last = x
                continue
            runs.append([start, last])
            start = last = x
        runs.append([start, last])
        out.append([y, runs])
    return out


def _unrle(rows) -> set:
    out = set()
    for y, runs in rows:
        for x0, x1 in runs:
            out.update((x, y) for x in range(x0, x1 + 1))
    return out


class Codec:
    """Encoder / decoder bound to one loaded harness package (e.g. `harness.v9_keys`)."""

    def __init__(self, package: str):
        self.package = package
        self.see = importlib.import_module(f"{package}.see")

    # ---- encode -----------------------------------------------------------------------------------
    def encode(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"__float__": repr(value)}
        if isinstance(value, list):
            return [self.encode(v) for v in value]
        if isinstance(value, tuple):
            return {"__tuple__": [self.encode(v) for v in value]}
        if isinstance(value, dict):
            if all(type(k) is str and not (k.startswith("__") and k.endswith("__")) for k in value):
                return {k: self.encode(v) for k, v in value.items()}
            return {"__dict__": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (set, frozenset)):
            if value and _cells(value):
                return {"__cells__": _rle(value)}
            items = [self.encode(v) for v in value]
            return {"__set__": sorted(items, key=lambda v: json.dumps(v, sort_keys=True))}
        if isinstance(value, self.see.View):
            return {"__view__": value.obs}
        module = type(value).__module__
        if module.startswith(self.package + "."):
            name = f"{module[len(self.package) + 1:]}.{type(value).__qualname__}"
            if dataclasses.is_dataclass(value):
                fields = {f.name: self.encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
            else:
                fields = {k: self.encode(v) for k, v in vars(value).items()}
            return {"__obj__": name, "fields": fields}
        raise TypeError(f"the harness state holds a {type(value).__name__} ({module}) that the codec cannot save")

    # ---- decode -----------------------------------------------------------------------------------
    def decode(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.decode(v) for v in value]
        if not isinstance(value, dict):
            return value
        if len(value) == 1:
            ((key, item),) = value.items()
            if key == "__float__":
                return float(item)
            if key == "__tuple__":
                return tuple(self.decode(v) for v in item)
            if key == "__dict__":
                return {self.decode(k): self.decode(v) for k, v in item}
            if key == "__cells__":
                return _unrle(item)
            if key == "__set__":
                return {self.decode(v) for v in item}
            if key == "__view__":
                return self.see.View(item)
        if set(value) == {"__obj__", "fields"}:
            module, _, cls = value["__obj__"].rpartition(".")
            klass = getattr(importlib.import_module(f"{self.package}.{module}"), cls)
            obj = object.__new__(klass)
            for k, v in value["fields"].items():
                object.__setattr__(obj, k, self.decode(v))
            return obj
        return {k: self.decode(v) for k, v in value.items()}


def snapshot(agent, package: str, harness_name: str, request_id: int, exclude=EXCLUDE) -> dict:
    """The agent's state right before `ask()`, as JSON-ready data (see the module text)."""
    codec = Codec(package)
    fields = {k: codec.encode(v) for k, v in vars(agent).items() if k not in exclude}
    return {"format": FORMAT, "harness": harness_name, "request_id": int(request_id), "agent": fields}


def restore(state: dict, package: str):
    """An Agent of the loaded harness package with the saved state and no game attached."""
    if state.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} state")
    import runtime  # Laya's tokenizer set in the harness's text module, or refuse

    agent_mod = runtime.harness_agent(package)
    text_mod = importlib.import_module(f"{package}.text")
    codec = Codec(package)
    fields = {k: codec.decode(v) for k, v in state["agent"].items()}
    agent = agent_mod.Agent(None, {k: v for k, v in fields["knobs"].items()})  # noqa: C416
    for k, v in fields.items():
        setattr(agent, k, v)
    if "round" in agent.knobs:  # Agent.__init__ sets this module global from the knobs (text.q / text.small)
        text_mod.STEP = float(agent.knobs["round"])
    agent.env = None
    return agent


def rerender(state: dict, package: str, build_request) -> dict:
    """Ask the restored agent again; the request exactly as the collector built it."""
    agent = restore(state, package)
    ask = agent.ask()
    return build_request(state["request_id"], ask["essential"], ask["questions"])
