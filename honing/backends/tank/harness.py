"""The tank harness: how the decision model is asked, as files an evolution method can rewrite.

A harness for agent ``tank_classic`` is read by ``integrations/tank-war/scripts/match.mjs``:

- ``tank/controller.json`` (kind ``settings``): controller mode (planned steering
  or raw controls), context profile, dodge assist, shot guard, and for planned
  steering the planner's strategy, by published id or as planner parameters.
- ``tank/questions.json`` (kind ``instructions``): the instruction text of every
  question the model is asked, and the two weapon option descriptions.
- Any other file, of either kind, is shown to the model before the state on
  every decision, as an operating note.

:func:`seed_harness` is the integration's ``config/`` defaults, which reproduce
what the browser game asks. :func:`check_spec` refuses, before any match, a
controller or question file the match would refuse; the match process checks
the strategy's planner parameters itself, and the runner turns that refusal
into the same :class:`~honing.core.CampaignRuleError`.
"""

from __future__ import annotations

import json
from pathlib import Path

from honing.core import CampaignRuleError, HarnessFile, HarnessSpec

AGENT = "tank_classic"
CONTROLLER_PATH = "tank/controller.json"
QUESTIONS_PATH = "tank/questions.json"
CONTROLLER_KEYS = frozenset(
    {"controllerMode", "contextVersion", "assistEnabled", "assistProfile", "fireGuardEnabled", "strategy"}
)


def seed_harness(integration_dir: Path) -> HarnessSpec:
    """The harness the match plays with no harness given: the integration's ``config/`` files."""
    config = integration_dir / "config"
    return HarnessSpec(
        agent=AGENT,
        files=(
            (CONTROLLER_PATH, HarnessFile("settings", (config / "controller.json").read_text(encoding="utf-8"))),
            (QUESTIONS_PATH, HarnessFile("instructions", (config / "questions.json").read_text(encoding="utf-8"))),
        ),
    )


def question_keys(integration_dir: Path) -> frozenset[str]:
    """Every key a question file must define: the keys of the default."""
    return frozenset(json.loads((integration_dir / "config" / "questions.json").read_text(encoding="utf-8")))


def _object(spec: HarnessSpec, path: str) -> dict[str, object] | None:
    file = spec.get(path)
    if file is None:
        return None
    try:
        value = json.loads(file.content)
    except json.JSONDecodeError as error:
        raise CampaignRuleError("harness_readable", f"{path} is not JSON: {error}") from None
    if not isinstance(value, dict):
        raise CampaignRuleError("harness_readable", f"{path} must be a JSON object")
    return value


def check_spec(spec: HarnessSpec, required_questions: frozenset[str]) -> None:
    """Refuse a harness whose controller or question file the match cannot read."""
    if spec.agent != AGENT:
        raise CampaignRuleError("harness_readable", f"the tank runner plays agent {AGENT!r}, not {spec.agent!r}")
    controller = _object(spec, CONTROLLER_PATH)
    if controller is not None:
        unknown = sorted(set(controller) - CONTROLLER_KEYS)
        if unknown:
            raise CampaignRuleError("harness_readable", f"{CONTROLLER_PATH} has unknown settings {unknown}")
    questions = _object(spec, QUESTIONS_PATH)
    if questions is not None:
        missing = sorted(required_questions - set(questions))
        unknown = sorted(set(questions) - required_questions)
        empty = sorted(k for k, v in questions.items() if not isinstance(v, str) or not v.strip())
        if missing or unknown or empty:
            raise CampaignRuleError(
                "harness_readable",
                f"{QUESTIONS_PATH} must define exactly {sorted(required_questions)} as text;"
                f" missing {missing}, unknown {unknown}, empty {empty}",
            )
