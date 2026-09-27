"""Red Alert 2 as a Honing task environment.

A task is one match against the Supalosa bot, described by a scenario in the
task suite. :class:`RA2EpisodeRunner` plays it through the RA2 integration
(``integrations/ra2``) with the policy served by a Jev decision server, and
:class:`RA2Grader` audits the match and turns its result into a grade. Nothing
here imports the engine or Node code; both run as separate processes.
"""

from honing.backends.ra2.demonstrations import RA2DemonstrationProvider
from honing.backends.ra2.duel import DuelEvaluator
from honing.backends.ra2.grader import RA2Grader
from honing.backends.ra2.harness import LayaQuestionCheck, seed_harness
from honing.backends.ra2.rollouts import RA2RolloutProvider, decisions_from_log
from honing.backends.ra2.runner import RA2EpisodeRunner
from honing.backends.ra2.scenarios import RA2Scenario, RA2Suite, build_suite, load_suite

__all__ = [
    "DuelEvaluator",
    "LayaQuestionCheck",
    "RA2DemonstrationProvider",
    "RA2EpisodeRunner",
    "RA2Grader",
    "RA2RolloutProvider",
    "RA2Scenario",
    "RA2Suite",
    "build_suite",
    "decisions_from_log",
    "load_suite",
    "seed_harness",
]
