"""The RA2 harness: how the decision model is asked, as files an evolution method can rewrite.

A harness for agent ``ra2_decision`` has two files, which the match reads as
``--controller`` and ``--questions``:

- ``ra2/controller.json`` (kind ``settings``): decision mode, candidate mode,
  decision interval, and cadence.
- ``ra2/questions.json`` (kind ``instructions``): the question, option, and
  recommendation text, with ``{name}`` placeholders filled during the match.

:func:`seed_harness` is the integration's current behavior. :class:`LayaQuestionCheck`,
:class:`KevQuestionCheck`, :class:`NanoJevQuestionCheck`, and :class:`StartLuxQuestionCheck`
refuse, before any match starts, a harness the model cannot read: every option,
instruction, and state entry must fit the family's limits (:data:`LAYA_LIMITS`,
:data:`KEV_LIMITS`, :data:`NANOJEV_LIMITS`, :data:`STARTLUX_LIMITS`). Placeholders are filled with the
longest values the match produces, so the check is conservative. Text built
from game content can still overrun during a match; the grader then scores that
match 0 and names the question.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from honing.core import CampaignRuleError, HarnessFile, HarnessSpec

AGENT = "ra2_decision"
CONTROLLER_PATH = "ra2/controller.json"
QUESTIONS_PATH = "ra2/questions.json"
CONTROLLER_KEYS = frozenset({"decisionMode", "candidateMode", "decisionInterval", "cadence"})


@dataclass(frozen=True)
class QuestionLimits:
    """Token limits for one entry of question text, by the role the entry plays."""

    option: int
    instruction: int
    state_entry: int


# Laya: 48 tokens is its lossless limit per option; instructions get half the 512-token question budget;
# one state entry gets a quarter of the 2048-token input.
LAYA_LIMITS = QuestionLimits(option=48, instruction=256, state_entry=512)
# Kev has no per-option limit, only a row limit; these keep one question within the 1,024-token branch it was
# trained on (instructions half of it, an option a quarter) and one state entry within a quarter of its
# 8,192-token serving state.
KEV_LIMITS = QuestionLimits(option=256, instruction=512, state_entry=2048)
# NanoJev scores each option on its own path (state, question, candidate) within a 2,048-token input; these keep
# a candidate within an eighth of it, instructions within a quarter, and one state entry within a quarter.
NANOJEV_LIMITS = QuestionLimits(option=256, instruction=512, state_entry=512)
# StartLux-Decision reads every option of a question in one prompt, up to the server's 8,192-token default; these keep
# an option within a thirty-second of it, instructions within a sixteenth, and one state entry within a quarter.
STARTLUX_LIMITS = QuestionLimits(option=256, instruction=512, state_entry=2048)
OPTION_LIMIT, INSTRUCTION_LIMIT, STATE_ENTRY_LIMIT = (
    LAYA_LIMITS.option,
    LAYA_LIMITS.instruction,
    LAYA_LIMITS.state_entry,
)
# The longest value each placeholder takes in a match.
WORST_CASE = {"stage": "8", "tick": "999999", "queued": "99"}


def seed_harness(integration_dir: Path) -> HarnessSpec:
    """The harness the integration runs with no files given: its ``config/`` defaults."""
    config = integration_dir / "config"
    return HarnessSpec(
        agent=AGENT,
        files=(
            (CONTROLLER_PATH, HarnessFile("settings", (config / "controller.json").read_text(encoding="utf-8"))),
            (QUESTIONS_PATH, HarnessFile("instructions", (config / "questions.json").read_text(encoding="utf-8"))),
        ),
    )


def role(key: str) -> str:
    """Whether a question-text entry is shown as an option, as instructions, or as state."""
    if key in ("staged.cancelPending", "staged.noActionForGroup") or key.startswith("intent.group."):
        return "option"
    if key == "staged.selectActionFirst":
        return "instruction"
    return "state"


def worst_case(template: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: WORST_CASE.get(m.group(1), "9" * 6), template)


class QuestionCheck(Protocol):
    """Refuses, before any match, a harness the policy's model cannot read."""

    def check(self, spec: HarnessSpec) -> None: ...


def _required_keys(integration_dir: Path) -> frozenset[str]:
    return frozenset(json.loads((integration_dir / "config/questions.json").read_text(encoding="utf-8")))


class TokenQuestionCheck:
    """Refuses a harness whose question text does not fit a decision model's token limits, before any match.

    ``tokenizer`` is the model's tokenizer, called as Hugging Face tokenizers
    are: ``tokenizer(text, add_special_tokens=False)["input_ids"]``.
    ``required_keys`` are the question-text keys the match reads; a harness
    missing one would fail at the match's start. Subclasses set the limits and
    how an option is counted.
    """

    limits = LAYA_LIMITS

    def __init__(self, tokenizer: Any, required_keys: frozenset[str]) -> None:
        self._tok = tokenizer
        self._required = required_keys

    def _tokens(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def _option_tokens(self, text: str) -> int:
        return self._tokens(text)

    def _controller(self, text: str) -> list[str]:
        problems = []
        settings = json.loads(text)
        if not isinstance(settings, Mapping):
            return ["controller.json must be an object"]
        unknown = set(settings) - CONTROLLER_KEYS
        if unknown:
            problems.append(f"unknown controller settings {sorted(unknown)}")
        if settings.get("decisionMode", "staged") not in ("staged", "flat"):
            problems.append("decisionMode must be staged or flat")
        if settings.get("candidateMode", "intent") not in ("intent", "catalog", "legacy"):
            problems.append("candidateMode must be intent, catalog, or legacy")
        cadence, interval = settings.get("cadence", 3), settings.get("decisionInterval", 3)
        if not (isinstance(cadence, int) and cadence >= 3 and cadence % 3 == 0):
            problems.append("cadence must be a positive multiple of 3")
        elif not (isinstance(interval, int) and interval >= cadence and interval % cadence == 0):
            problems.append("decisionInterval must be a positive multiple of cadence")
        return problems

    def _questions(self, text: str) -> list[str]:
        problems = []
        entries = json.loads(text)
        if not isinstance(entries, Mapping):
            return ["questions.json must be an object"]
        missing = self._required - set(entries)
        if missing:
            problems.append(f"questions.json is missing {sorted(missing)}")
        for key, template in entries.items():
            if not isinstance(template, str):
                problems.append(f"{key} is not text")
                continue
            filled = worst_case(template)
            kind = role(key)
            if kind == "option":
                count, limit = self._option_tokens(filled), self.limits.option
            else:
                count = self._tokens(filled)
                limit = self.limits.instruction if kind == "instruction" else self.limits.state_entry
            if count > limit:
                problems.append(f"{key} ({kind}) is {count} tokens; the limit is {limit}")
        return problems

    def check(self, spec: HarnessSpec) -> None:
        """Raise ``questions_fit_model`` with every problem found, or return."""
        problems = []
        if spec.agent != AGENT:
            problems.append(f"agent is {spec.agent!r}, not {AGENT!r}")
        controller, questions = spec.get(CONTROLLER_PATH), spec.get(QUESTIONS_PATH)
        try:
            if controller is not None:
                problems += self._controller(controller.content)
            if questions is not None:
                problems += self._questions(questions.content)
        except json.JSONDecodeError as error:
            problems.append(f"a harness file is not JSON: {error}")
        if problems:
            raise CampaignRuleError("questions_fit_model", "; ".join(problems))


class LayaQuestionCheck(TokenQuestionCheck):
    """The check for Laya, whose options are encoded after a space with the mask token blanked."""

    limits = LAYA_LIMITS

    @classmethod
    def for_checkpoint(cls, checkpoint: Path, integration_dir: Path) -> LayaQuestionCheck:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        return cls(tokenizer, _required_keys(integration_dir))

    def _option_tokens(self, text: str) -> int:
        mask = getattr(self._tok, "mask_token", None) or ""
        return self._tokens(" " + (text.replace(mask, " ") if mask else text))


class KevQuestionCheck(TokenQuestionCheck):
    """The check for Kev, whose option spans are the option's tokens between two delimiter tokens."""

    limits = KEV_LIMITS

    @classmethod
    def for_checkpoint(cls, checkpoint: Path, integration_dir: Path) -> KevQuestionCheck:
        """``checkpoint`` is a Kev checkpoint directory, which carries its base's tokenizer."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        return cls(tokenizer, _required_keys(integration_dir))

    def _option_tokens(self, text: str) -> int:
        return 2 + self._tokens(text)


class NanoJevQuestionCheck(TokenQuestionCheck):
    """The check for NanoJev, whose option is the candidate line of its own path, followed by an end token."""

    limits = NANOJEV_LIMITS

    @classmethod
    def for_checkpoint(cls, checkpoint: Path, integration_dir: Path) -> NanoJevQuestionCheck:
        """``checkpoint`` is a NanoJev checkpoint directory, with its tokenizer in ``tokenizer/``."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
        return cls(tokenizer, _required_keys(integration_dir))

    def _option_tokens(self, text: str) -> int:
        return 1 + self._tokens(f"Candidate:\nX: {text}\nDecision:")


class StartLuxQuestionCheck(TokenQuestionCheck):
    """The check for StartLux-Decision, whose option is one lettered line of the prompt, such as ``B) text``."""

    limits = STARTLUX_LIMITS

    @classmethod
    def for_checkpoint(cls, checkpoint: Path, integration_dir: Path) -> StartLuxQuestionCheck:
        """``checkpoint`` is a StartLux-Decision checkpoint directory, which carries its tokenizer."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        return cls(tokenizer, _required_keys(integration_dir))

    def _option_tokens(self, text: str) -> int:
        return self._tokens(f"B) {text}\n")
