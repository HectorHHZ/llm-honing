"""Build a component named in a config by import path, and check what the config needs from the machine.

A component entry is ``{"factory": "package.module:function", "args": {...}}``.
Any value in a config may be ``{"ref": "dotted.path"}``, which
:func:`resolve_references` replaces with the value at that path in the same
config, so a setting several entries share (a model pin) is written once.
The factory is called as ``function(context, **args)`` and must return an
object satisfying the slot's protocol. ``context`` is whatever the caller
passes: it carries paths, the components built so far, and a ``shared`` map
for factories that belong together (a fake model source and the fake trainer
that must use the same fake storage, for example).

Every campaign config has the same optional ``scheduler`` slot
(:data:`SCHEDULER_SLOT`). A config that names none runs the moves of its
``schedule`` through a :class:`~honing.campaign.ScriptedScheduler`
(:func:`scheduler_or_script`).

Besides its single slots, a ``components`` block may hold ``trainers``, an
object of entries a train or prune step names with ``"trainer"``, and
``methods``, a list of entries for further harness methods, which a harness
step names by the method's own name with ``"method"``
(:func:`build_named_trainers_and_methods`). A ``direction_edit`` train step can
state its own edit (:func:`direction_edit_spec`).
"""

from __future__ import annotations

import importlib
import os
import shutil
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honing.campaign import ScriptedScheduler
from honing.core import CoAdaptScheduler, ContractError, DirectionEditSpec, EvolveMethod, Move, Trainer

# Built after every other slot, so a scheduler's factory can read all the components.
SCHEDULER_SLOT: tuple[str, type, bool] = ("scheduler", CoAdaptScheduler, False)
#: Sections of a ``components`` block that hold several entries instead of one.
TRAINERS, METHODS = "trainers", "methods"


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
    if not isinstance(spec, Mapping):
        raise ContractError(f"the {slot} component must be an object with a factory, got {type(spec).__name__}")
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


def scheduler_or_script(components: Mapping[str, object], moves: Sequence[Move]) -> CoAdaptScheduler:
    """The ``scheduler`` component a config built, or a :class:`ScriptedScheduler` replaying ``moves``.

    With a scheduler, the config's own moves are not run, so a warning names them.
    """
    scheduler = components.get("scheduler")
    if scheduler is None:
        return ScriptedScheduler(moves)
    assert isinstance(scheduler, CoAdaptScheduler)
    if moves:
        kinds = ", ".join(move.kind for move in moves)
        warnings.warn(
            f"the {scheduler.name!r} scheduler chooses every move, so the schedule's moves ({kinds}) are not run."
            " In a pipeline they only place its rollback steps.",
            UserWarning,
            stacklevel=2,
        )
    return scheduler


def build_named_trainers_and_methods(
    components: Mapping[str, object], context: object
) -> tuple[dict[str, Trainer], tuple[EvolveMethod, ...]]:
    """Build the ``trainers`` and ``methods`` sections of a ``components`` block; both are optional."""
    trainers = components.get(TRAINERS, {})
    if not isinstance(trainers, Mapping) or not all(isinstance(name, str) and name for name in trainers):
        raise ContractError(f"components.{TRAINERS} must be an object of named component entries")
    methods = components.get(METHODS, [])
    if not isinstance(methods, list):
        raise ContractError(f"components.{METHODS} must be a list of component entries")
    built_trainers: dict[str, Trainer] = {}
    for name, spec in trainers.items():
        built_trainers[name] = build_component(spec, context, f"{TRAINERS}.{name}", Trainer)  # type: ignore[assignment]
    built_methods = [
        build_component(spec, context, f"{METHODS}[{index}]", EvolveMethod) for index, spec in enumerate(methods)
    ]
    return built_trainers, tuple(built_methods)  # type: ignore[arg-type]


def direction_edit_spec(entry: object) -> DirectionEditSpec | None:
    """A train step's ``direction_edit`` entry as a spec; ``None`` when the step has none.

    The entry states all three of ``question``, ``target`` and ``strength``,
    because it replaces all three of the trainer's settings. ``question`` is
    ``null`` for every question.
    """
    if entry is None:
        return None
    fields = {"question", "target", "strength"}
    if not isinstance(entry, Mapping) or set(entry) != fields:
        given = sorted(entry) if isinstance(entry, Mapping) else type(entry).__name__
        raise ContractError(
            f"a step's direction_edit states exactly {sorted(fields)} (question null for every question), got {given}"
        )
    return DirectionEditSpec(question=entry["question"], target=entry["target"], strength=entry["strength"])


def missing_requirements(requires: Mapping[str, Sequence[str]]) -> list[str]:
    """Every environment variable, command, or path the config needs and this machine lacks."""
    missing = [f"environment variable {name}" for name in requires.get("env", ()) if not os.environ.get(name)]
    missing += [f"command {name}" for name in requires.get("commands", ()) if shutil.which(name) is None]
    missing += [f"path {path}" for path in requires.get("paths", ()) if not Path(os.path.expandvars(path)).exists()]
    return missing


def resolve_references(config: Mapping[str, Any]) -> dict[str, Any]:
    """The config with each ``{"ref": "a.b"}`` replaced by the value at ``a.b``; refuses an unknown path or a cycle."""

    def lookup(path: str, seen: tuple[str, ...]) -> Any:
        if path in seen:
            raise ContractError(f"config references form a cycle: {' -> '.join((*seen, path))}")
        value: Any = config
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                raise ContractError(f"the config has no {path!r} for a reference to name")
            value = value[key]
        return resolve(value, (*seen, path))

    def resolve(value: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"ref"} and isinstance(value["ref"], str):
                return lookup(value["ref"], seen)
            return {key: resolve(item, seen) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, seen) for item in value]
        return value

    resolved = resolve(config, ())
    assert isinstance(resolved, dict)
    return resolved
