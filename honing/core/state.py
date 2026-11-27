"""Rules for the state a campaign writes into the commit record and reads back.

Two components keep such state. A harness evolution method's state is
published and rolled back with a release. A scheduler's state is carried by
every record line written after its last decision, and is read back on a
restart. Both are written as JSON, so both must survive that round trip
unchanged. A value JSON cannot represent exactly, such as a set or a tuple,
would come back as something else.

Every non-empty state carries an integer ``schema_version``. A component's
state changes shape as the component evolves, and a rollback or a restart can
hand old state to newer code; the version is what lets the campaign refuse that
instead of letting the component misread it.

A campaign with one method publishes that method's state as it is. A campaign
with several publishes one state that holds each method's own under its name
(:func:`keyed_method_state`), so a release still restores all of them together.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from honing.core.errors import ContractError

METHOD_STATE_VERSION_KEY = "schema_version"
SCHEDULER_STATE_VERSION_KEY = "schema_version"
METHOD_STATES_KEY = "methods"
KEYED_STATE_SCHEMA_VERSION = 1


def json_native_problem(value: object, path: str = "state") -> str | None:
    """Describe the first value under ``path`` that JSON cannot round-trip, or return ``None``."""
    if value is None or isinstance(value, (bool, str)):
        return None
    if isinstance(value, int):
        return None
    if isinstance(value, float):
        return None if math.isfinite(value) else f"{path} is {value}, which JSON cannot represent"
    if isinstance(value, list):
        for index, item in enumerate(value):
            problem = json_native_problem(item, f"{path}[{index}]")
            if problem is not None:
                return problem
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f"{path} has the key {key!r}; JSON object keys must be strings"
            problem = json_native_problem(item, f"{path}[{key!r}]")
            if problem is not None:
                return problem
        return None
    return f"{path} is a {type(value).__name__}; use a dict, list, str, int, float, bool, or None"


def _state_version(state: Mapping[str, object], owner: str, key: str) -> int | None:
    if not state:
        return None
    version = state.get(key)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ContractError(f"{owner} state must carry an integer {key!r}, got {version!r}")
    return version


def _validate_state(state: Mapping[str, object], owner: str, key: str) -> None:
    problem = json_native_problem(dict(state))
    if problem is not None:
        raise ContractError(f"{owner} state must be JSON-native: {problem}")
    _state_version(state, owner, key)


def method_state_version(state: Mapping[str, object]) -> int | None:
    """The state's ``schema_version``, or ``None`` for an empty state."""
    return _state_version(state, "method", METHOD_STATE_VERSION_KEY)


def validate_method_state(state: Mapping[str, object]) -> None:
    """Refuse state that would not survive the commit record, or that has no schema version."""
    _validate_state(state, "method", METHOD_STATE_VERSION_KEY)


def scheduler_state_version(state: Mapping[str, object]) -> int | None:
    """The scheduler state's ``schema_version``, or ``None`` for an empty state."""
    return _state_version(state, "scheduler", SCHEDULER_STATE_VERSION_KEY)


def validate_scheduler_state(state: Mapping[str, object]) -> None:
    """Refuse scheduler state that would not survive the commit record, or that has no schema version."""
    _validate_state(state, "scheduler", SCHEDULER_STATE_VERSION_KEY)


def keyed_method_state(states: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """One state holding several methods' own states, each under its method's name."""
    for name in states:
        if not isinstance(name, str) or not name:
            raise ContractError(f"a method's state is keyed by its name, got {name!r}")
    return {
        METHOD_STATE_VERSION_KEY: KEYED_STATE_SCHEMA_VERSION,
        METHOD_STATES_KEY: {name: dict(state) for name, state in states.items()},
    }


def method_states(state: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Each method's own state from a :func:`keyed_method_state`; refuses any other shape or version."""
    version = method_state_version(state)
    keyed = state.get(METHOD_STATES_KEY)
    if version != KEYED_STATE_SCHEMA_VERSION or set(state) != {METHOD_STATE_VERSION_KEY, METHOD_STATES_KEY}:
        raise ContractError(
            f"expected method states keyed by method name under schema version {KEYED_STATE_SCHEMA_VERSION},"
            f" got version {version} with keys {sorted(state)}"
        )
    if not isinstance(keyed, dict) or not all(isinstance(own, dict) for own in keyed.values()):
        raise ContractError(f"{METHOD_STATES_KEY!r} must map each method's name to its state")
    return dict(keyed)
