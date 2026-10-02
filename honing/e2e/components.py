"""Build a component named in a config by import path, and check what the config needs from the machine.

A component entry is ``{"factory": "package.module:function", "args": {...}}``.
The factory is called as ``function(context, **args)`` and must return an
object satisfying the slot's protocol. ``context`` is whatever the caller
passes: it carries paths, the components built so far, and a ``shared`` map
for factories that belong together (a fake model source and the fake trainer
that must use the same fake storage, for example).
"""

from __future__ import annotations

import importlib
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honing.core import ContractError


@dataclass
class PipelineContext:
    """What a component factory can read: paths, the components built so far, and shared state."""

    root: Path
    work_dir: Path
    components: dict[str, object] = field(default_factory=dict)
    shared: dict[str, object] = field(default_factory=dict)

    def component(self, slot: str) -> Any:
        if slot not in self.components:
            raise ContractError(f"the {slot} component is needed here but was not built earlier")
        return self.components[slot]


def build_component(spec: Mapping[str, object], context: object, slot: str, protocol: type) -> object:
    """Call the entry's factory with ``context`` and its args; refuse a result that is not a ``protocol``."""
    target = str(spec.get("factory", ""))
    module_name, _, function_name = target.partition(":")
    if not module_name or not function_name:
        raise ContractError(f"the {slot} component needs a factory named 'package.module:function', got {target!r}")
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as error:
        raise ContractError(f"cannot load the {slot} factory {target!r}: {error}") from None
    args = spec.get("args", {})
    if not isinstance(args, Mapping):
        raise ContractError(f"the {slot} component's args must be an object")
    component = factory(context, **args)
    if not isinstance(component, protocol):
        raise ContractError(
            f"the {slot} factory {target!r} returned {type(component).__name__}, not a {protocol.__name__}"
        )
    return component


def missing_requirements(requires: Mapping[str, Sequence[str]]) -> list[str]:
    """Every environment variable, command, or path the config needs and this machine lacks."""
    missing = [f"environment variable {name}" for name in requires.get("env", ()) if not os.environ.get(name)]
    missing += [f"command {name}" for name in requires.get("commands", ()) if shutil.which(name) is None]
    missing += [f"path {path}" for path in requires.get("paths", ()) if not Path(os.path.expandvars(path)).exists()]
    return missing
