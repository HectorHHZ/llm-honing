"""Tank duels (Tank War) as a Honing task environment.

A task is one round against the planner opponent, described by a scenario in
the task suite. :class:`TankEpisodeRunner` plays it through the tank
integration (``integrations/tank-war/scripts/match.mjs``) with the policy
served by a Jev decision server, and :class:`TankGrader` checks the round and
turns its result into a grade. :class:`FreshSeedEvaluator` gives every gate
validation seeds no earlier gate played. :class:`TankRolloutProvider` plays sampled
rounds for policy gradient, and :class:`TankDemonstrationProvider` labels the
model's questions with the planner teacher's answers for supervised training.
:class:`TankTraceSource` reads a played round back as decision traces.
Nothing here imports the game; it runs as a separate Node process.

Each name below is imported from its module the first time it is read, so
importing one module of this package, such as ``honing.backends.tank.traces``,
loads only that module and what it imports itself.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from honing.backends.tank.decisions import decisions_from_log
    from honing.backends.tank.demonstrations import TankDemonstrationProvider
    from honing.backends.tank.gate import FreshSeedEvaluator, fresh_seed_gate
    from honing.backends.tank.grader import TankGrader
    from honing.backends.tank.harness import AGENT, CONTROLLER_PATH, QUESTIONS_PATH, check_spec, seed_harness
    from honing.backends.tank.rollouts import TankRolloutProvider
    from honing.backends.tank.runner import TankEpisodeRunner
    from honing.backends.tank.scenarios import (
        SEED_POOLS,
        TankScenario,
        TankSuite,
        build_suite,
        load_suite,
        pool_tasks,
    )
    from honing.backends.tank.traces import TankTraceSource

# Each public name, by the module that defines it.
_MODULES = {
    "AGENT": "harness",
    "CONTROLLER_PATH": "harness",
    "QUESTIONS_PATH": "harness",
    "SEED_POOLS": "scenarios",
    "FreshSeedEvaluator": "gate",
    "TankDemonstrationProvider": "demonstrations",
    "TankEpisodeRunner": "runner",
    "TankGrader": "grader",
    "TankRolloutProvider": "rollouts",
    "TankScenario": "scenarios",
    "TankSuite": "scenarios",
    "TankTraceSource": "traces",
    "build_suite": "scenarios",
    "check_spec": "harness",
    "decisions_from_log": "decisions",
    "fresh_seed_gate": "gate",
    "load_suite": "scenarios",
    "pool_tasks": "scenarios",
    "seed_harness": "harness",
}


def __getattr__(name: str) -> Any:
    """The public ``name``, imported from its module on first use (PEP 562)."""
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_MODULES})


__all__ = [
    "AGENT",
    "CONTROLLER_PATH",
    "QUESTIONS_PATH",
    "SEED_POOLS",
    "FreshSeedEvaluator",
    "TankDemonstrationProvider",
    "TankEpisodeRunner",
    "TankGrader",
    "TankRolloutProvider",
    "TankScenario",
    "TankSuite",
    "TankTraceSource",
    "build_suite",
    "check_spec",
    "decisions_from_log",
    "fresh_seed_gate",
    "load_suite",
    "pool_tasks",
    "seed_harness",
]
