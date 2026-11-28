"""Reef recipe pieces for the tests that start a real Reef: a record-only recipe (#225) and a stage of its own (#228).

Reef loads them by import path in its own environment, with the repository
root on ``PYTHONPATH``, so they import Reef and Honing's recipe, never Honing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from reef.recipe.base import Recipe
from reef.samples import ReportContext, SkippedReport
from reef.train.types import Sample

from integrations.reef.processor import DecisionEpisodeSamples
from integrations.reef.report import HoningEpisodeReport


@dataclass(frozen=True)
class EpisodeReportRecipe(Recipe):
    """Records only, and admits Honing's episode reports only (``tests.fakes.reef_recipe:EpisodeReportRecipe``).

    Follows Reef's ``Recipe.report_type`` (reef/recipe/base.py, Reef bc8c150): a
    recipe names the report type that Reef parses at admission. Differs: it
    records only, and trains nothing.
    """

    @property
    def report_type(self) -> type[HoningEpisodeReport]:
        return HoningEpisodeReport


class CenteredEpisodeSamples(DecisionEpisodeSamples):
    """Honing's episode samples, each rewarded with its score minus 0.5, so a draw is worth nothing.

    The tank dry run with recording on (``tests/test_reef_dry_run.py``) names it
    in its deployment's ``pipeline.record2sample.module``, in place of the
    recipe's own stage, and checks that its rewards reach the trainer.

    Follows Reef's ``PipelineSettings`` (reef/train/pipeline.py, Reef bc8c150):
    a deployment's ``pipeline`` section names a stage class in place of the
    recipe's. Differs: test only.
    """

    def make_sample(self, context: ReportContext) -> Sample | SkippedReport:
        if context.score is not None:
            context = replace(context, score=context.score - 0.5)
        return super().make_sample(context)
