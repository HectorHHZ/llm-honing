"""REA-Issac: an independent room-combat simulation, UI and model task adapter.

Importing this backend needs only the Python standard library. No original
game installation, reverse-engineering tools, GPU or model weights are needed.
"""

from honing.backends.rea_issac.grader import ReaIssacGrader
from honing.backends.rea_issac.harness import AGENT, parse_harness, seed_harness
from honing.backends.rea_issac.runner import ReaIssacEpisodeRunner
from honing.backends.rea_issac.scenarios import ReaIssacScenario, ReaIssacSuite, build_suite, load_suite
from integrations.rea_issac.engine import Action, Engine

__all__ = [
    "AGENT",
    "Action",
    "Engine",
    "ReaIssacEpisodeRunner",
    "ReaIssacGrader",
    "ReaIssacScenario",
    "ReaIssacSuite",
    "build_suite",
    "load_suite",
    "parse_harness",
    "seed_harness",
]
