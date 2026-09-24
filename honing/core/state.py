"""Rules for a harness evolution method's state, which is published and rolled back with a release.

The state is written into the commit record as JSON and read back on
rollback, so it must survive that round trip unchanged. A value JSON cannot
represent exactly, such as a set or a tuple, would come back as something else.

Every non-empty state carries an integer ``schema_version``. A method's state
changes shape as the method evolves, and a rollback can hand an old release's
state to newer code; the version is what lets the campaign refuse that instead
of letting the method misread it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from honing.core.errors import ContractError

METHOD_STATE_VERSION_KEY = "schema_version"


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


def method_state_version(state: Mapping[str, object]) -> int | None:
    """The state's ``schema_version``, or ``None`` for an empty state."""
    if not state:
        return None
    version = state.get(METHOD_STATE_VERSION_KEY)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ContractError(f"method state must carry an integer {METHOD_STATE_VERSION_KEY!r}, got {version!r}")
    return version


def validate_method_state(state: Mapping[str, object]) -> None:
    """Refuse state that would not survive the commit record, or that has no schema version."""
    problem = json_native_problem(dict(state))
    if problem is not None:
        raise ContractError(f"method state must be JSON-native: {problem}")
    method_state_version(state)
