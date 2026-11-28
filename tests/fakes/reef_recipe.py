"""A record-only Reef recipe that admits Honing's episode reports only, for the tests that start a real Reef (#225).

Reef loads it by import path (``tests.fakes.reef_recipe:EpisodeReportRecipe``) in
its own environment, with the repository root on ``PYTHONPATH``, so it imports
Reef and never Honing. Honing's own recipe, ``HoningDecisionRecipe``
(``integrations/reef/recipe.py``), declares the same report type and also
spools job data, which these tests do not need.

Follows Reef's ``Recipe.report_type`` (reef/recipe/base.py, Reef bc8c150): a
recipe names the report type that Reef parses at admission. Differs: it records
only, and trains nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.recipe.base import Recipe

from integrations.reef.report import HoningEpisodeReport


@dataclass(frozen=True)
class EpisodeReportRecipe(Recipe):
    @property
    def report_type(self) -> type[HoningEpisodeReport]:
        return HoningEpisodeReport
