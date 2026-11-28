"""Honing's Reef recipe: tank episodes in, episode trajectories out, spooled for Honing's campaign to train."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.recipe.base import WeightTrainingRecipe
from reef.storage.scenario.samples import SampleKind
from reef.train.pipeline import PipelineSpec

from integrations.reef.processor import DecisionEpisodeSamples, DecisionJobData
from integrations.reef.report import HoningEpisodeReport


@dataclass(frozen=True, kw_only=True)
class HoningDecisionRecipe(WeightTrainingRecipe):
    """Reported episodes become trajectory samples, and the spool backend hands their job data to Honing.

    Follows OpenClaw-RL's ``OpenClawRLRecipe`` (recipes/openclawrl/recipe.py,
    Reef bc8c150): a recipe class that names its two stages. Differs: its jobs
    go to ``integrations.reef.backend``, which trains nothing in Reef. Its
    ``report_type`` is ``HoningEpisodeReport``, so Reef refuses a report with no
    finite score or no task at admission (#225).
    """

    name: str = "honing"

    @property
    def report_type(self) -> type[HoningEpisodeReport]:
        return HoningEpisodeReport

    @classmethod
    def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
        return {"trajectory": {"objective": "honing"}}

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=DecisionEpisodeSamples, sample2job=DecisionJobData)
