"""The RA2 harness (#54): the seed harness, the check before any match, and a question that overruns mid-match."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from honing.backends.ra2 import LayaQuestionCheck, RA2EpisodeRunner, RA2Grader, load_suite, seed_harness
from honing.backends.ra2.harness import OPTION_LIMIT, role, worst_case
from honing.core import CampaignRuleError, EpisodeOutcome, HarnessFile, HarnessSpec
from tests.test_ra2_backend import FAKES, REPO, SUITE_PATH, _endpoints, _policy

INTEGRATION = REPO / "integrations/ra2"
DEFAULTS = json.loads((INTEGRATION / "config/questions.json").read_text())


class WordTokenizer:
    """One token per whitespace-separated word, called the way Hugging Face tokenizers are."""

    mask_token = "[MASK]"

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


def check() -> LayaQuestionCheck:
    return LayaQuestionCheck(WordTokenizer(), frozenset(DEFAULTS))


def with_questions(**changes: str) -> HarnessSpec:
    spec = seed_harness(INTEGRATION)
    return spec.with_file("ra2/questions.json", HarnessFile("instructions", json.dumps({**DEFAULTS, **changes})))


@pytest.mark.unit
def test_the_seed_harness_is_the_integration_s_defaults_byte_for_byte() -> None:
    spec = seed_harness(INTEGRATION)
    assert spec.agent == "ra2_decision"
    assert spec.get("ra2/controller.json").content == (INTEGRATION / "config/controller.json").read_text()  # type: ignore[union-attr]
    assert spec.get("ra2/questions.json").content == (INTEGRATION / "config/questions.json").read_text()  # type: ignore[union-attr]


@pytest.mark.unit
def test_the_seed_harness_passes_the_check() -> None:
    check().check(seed_harness(INTEGRATION))


@pytest.mark.unit
def test_every_entry_has_a_role_and_placeholders_take_their_longest_values() -> None:
    assert {role(k) for k in DEFAULTS} == {"option", "instruction", "state"}
    assert role("intent.group.recruit") == "option" and role("staged.frameFull") == "state"
    assert worst_case("Decision stage {stage} at tick {tick}.") == "Decision stage 8 at tick 999999."


@pytest.mark.unit
def test_an_option_too_long_for_the_model_is_refused_before_any_match() -> None:
    long_option = " ".join(["word"] * (OPTION_LIMIT + 1))
    with pytest.raises(CampaignRuleError) as refused:
        check().check(with_questions(**{"intent.group.recruit": long_option}))
    assert refused.value.rule == "questions_fit_model"
    assert "intent.group.recruit (option) is 49 tokens; the limit is 48" in str(refused.value)


@pytest.mark.unit
def test_a_harness_missing_a_question_or_with_bad_settings_is_refused() -> None:
    partial = {k: v for k, v in DEFAULTS.items() if k != "staged.commitments"}
    spec = seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(partial)))
    with pytest.raises(CampaignRuleError, match="missing"):
        check().check(spec)
    bad = seed_harness(INTEGRATION).with_file(
        "ra2/controller.json", HarnessFile("settings", json.dumps({"decisionInterval": 4, "cadence": 3}))
    )
    with pytest.raises(CampaignRuleError, match="multiple of cadence"):
        check().check(bad)


@pytest.mark.unit
def test_the_runner_refuses_an_unreadable_harness_before_starting_a_match(tmp_path: Path) -> None:
    runner = RA2EpisodeRunner(
        suite=load_suite(SUITE_PATH),
        endpoints=_endpoints(a="http://strong:1"),
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(FAKES / "fake_match.py")],
        question_check=check(),
    )
    spec = with_questions(**{"intent.group.recruit": " ".join(["word"] * 60)})
    with pytest.raises(CampaignRuleError, match="questions_fit_model"):
        asyncio.run(runner.run(spec, load_suite(SUITE_PATH).tasks("val")[0], _policy("a")))
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


@pytest.mark.unit
def test_the_runner_refuses_a_harness_that_changes_the_task_s_cadence(tmp_path: Path) -> None:
    runner = RA2EpisodeRunner(
        suite=load_suite(SUITE_PATH),
        endpoints=_endpoints(a="http://strong:1"),
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(FAKES / "fake_match.py")],
    )
    settings = json.loads((INTEGRATION / "config/controller.json").read_text())
    spec = seed_harness(INTEGRATION).with_file(
        "ra2/controller.json", HarnessFile("settings", json.dumps({**settings, "cadence": 6, "decisionInterval": 6}))
    )
    with pytest.raises(CampaignRuleError, match="cadence_fixed_by_task"):
        asyncio.run(runner.run(spec, load_suite(SUITE_PATH).tasks("val")[0], _policy("a")))
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())
    runner.check_spec(seed_harness(INTEGRATION))  # the seed's cadence is the suite's


def _stopped_match(tmp_path: Path, message: str) -> EpisodeOutcome:
    run = tmp_path / "run"
    run.mkdir()
    (run / "error.json").write_text(json.dumps({"error": message, "tick": 2310}))
    task = load_suite(SUITE_PATH).tasks("val")[0]
    return EpisodeOutcome(task.task_id, "h", "c", exit_code=1, workspace_location=str(run))


@pytest.mark.unit
def test_a_question_that_overruns_mid_match_scores_zero_and_says_which(tmp_path: Path) -> None:
    grader = RA2Grader(suite=load_suite(SUITE_PATH), integration_dir=INTEGRATION, audit_command=["true"])
    message = "Question g1 exceeds lossless option/instruction budget (longest option 61/48 tokens; question 140/512)"
    task = load_suite(SUITE_PATH).tasks("val")[0]
    grade = asyncio.run(grader.grade(task, _stopped_match(tmp_path, message)))
    assert grade.score == 0.0
    assert "tick 2310" in (grade.feedback or "") and "longest option 61/48" in (grade.feedback or "")


@pytest.mark.unit
def test_any_other_mid_match_failure_is_still_an_invalid_episode(tmp_path: Path) -> None:
    grader = RA2Grader(suite=load_suite(SUITE_PATH), integration_dir=INTEGRATION, audit_command=["true"])
    task = load_suite(SUITE_PATH).tasks("val")[0]
    from honing.core import ContractError

    with pytest.raises(ContractError, match="exited 1"):
        asyncio.run(grader.grade(task, _stopped_match(tmp_path, "Laya worker exited 139")))
