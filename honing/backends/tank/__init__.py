"""Tank duels (Tank War) as a Honing task environment.

A task is one round against the planner opponent, described by a scenario in
the task suite. :class:`TankEpisodeRunner` plays it through the tank
integration (``integrations/tank-war/scripts/match.mjs``) with the policy
served by a Jev decision server, and :class:`TankGrader` checks the round and
turns its result into a grade. :class:`TankRolloutProvider` plays sampled
rounds for policy gradient, and :class:`TankDemonstrationProvider` labels the
model's questions with the planner teacher's answers for supervised training.
Nothing here imports the game; it runs as a separate Node process.
"""

from honing.backends.tank.decisions import decisions_from_log
from honing.backends.tank.demonstrations import TankDemonstrationProvider
from honing.backends.tank.grader import TankGrader
from honing.backends.tank.harness import AGENT, CONTROLLER_PATH, QUESTIONS_PATH, check_spec, seed_harness
from honing.backends.tank.rollouts import TankRolloutProvider
from honing.backends.tank.runner import TankEpisodeRunner
from honing.backends.tank.scenarios import TankScenario, TankSuite, build_suite, load_suite

__all__ = [
    "AGENT",
    "CONTROLLER_PATH",
    "QUESTIONS_PATH",
    "TankDemonstrationProvider",
    "TankEpisodeRunner",
    "TankGrader",
    "TankRolloutProvider",
    "TankScenario",
    "TankSuite",
    "build_suite",
    "check_spec",
    "decisions_from_log",
    "load_suite",
    "seed_harness",
]
