"""End-to-end runs and the checks a new component must pass before it joins one.

- :mod:`honing.e2e.pipeline` runs a campaign through every plane from one JSON
  config that names each component by import path.
- :mod:`honing.e2e.conformance` checks a harness runner (OpenHands, Reef, ...)
  against what a campaign needs from it, and :mod:`honing.e2e.trainer_conformance`
  checks a trainer (Slime, ...).
- :mod:`honing.e2e.workspace_runner` is a reference runner at the boundary an
  agent framework sits behind: harness files in a workspace, an agent process,
  the policy's endpoint.
"""

from honing.e2e.components import PipelineContext, build_component, missing_requirements
from honing.e2e.conformance import DeliveryProbe, RunnerCheck, check_runner, probe_harness
from honing.e2e.trainer_conformance import TrainerCheck, check_trainer
from honing.e2e.workspace_runner import WorkspaceDeliveryProbe, WorkspaceEpisodeRunner

__all__ = [
    "DeliveryProbe",
    "PipelineContext",
    "RunnerCheck",
    "TrainerCheck",
    "WorkspaceDeliveryProbe",
    "WorkspaceEpisodeRunner",
    "build_component",
    "check_runner",
    "check_trainer",
    "missing_requirements",
    "probe_harness",
]
