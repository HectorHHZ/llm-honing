"""Corrected decisions (#177): weights diagnoses turned into a typed-decisions batch, and a tank dry run on them."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from honing.backends.tank.corrected import CorrectedDecisionProvider
from honing.backends.tank.e2e import build
from honing.backends.tank.harness import seed_harness
from honing.backends.tank.raw_states import TankRawStates, read_labels
from honing.campaign import ScriptedScheduler
from honing.core import ContractError, DataContract, DataProvider, TrainingBatch
from honing.harness.diagnosis import Cause, DiagnosedDecision, Diagnosis

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "integrations/tank-war"
DRY_RUN = REPO / "configs/e2e/tank_dry_run.json"
# A rollout round of the stand-in model under the seed harness (#204), every file of its run directory, gzipped.
FIXTURE = INTEGRATION / "tests/fixtures/rollout-s2002-blue.json.gz"
NODE = shutil.which("node")
SFT = DataContract("typed_decisions")

MOTION = {
    "type": "choice",
    "instructions": "Choose a motion.",
    "criteria": {"stop_none": "Stop.", "forward_none": "Go."},
}
WEAPON = {"type": "choice", "instructions": "Fire?", "criteria": {"fire": "Fire now.", "hold": "Hold."}}
# The policy stopped and fired. The teacher would have driven on and held.
TEACHER_LABELS = ({"motion": "forward_none"}, {"weapon": "hold"})


def _round(directory: Path, state: str = "Seed state.") -> Path:
    """A run directory as match.mjs and ``raw-states.mjs label`` leave it: a motion row and a weapon row."""
    rows = [
        {
            "request_id": "blue-1:motion",
            "contextParts": {"essential": [state]},
            "questions": {"motion": MOTION},
            "chosen": {"motion": "stop_none"},
        },
        {
            "request_id": "blue-1:weapon",
            "contextParts": {"essential": [state, "Fire permitted."]},
            "questions": {"weapon": WEAPON},
            "chosen": {"weapon": "fire"},
        },
    ]
    log = "".join(json.dumps(row) + "\n" for row in rows)
    directory.mkdir(parents=True)
    (directory / "decisions.jsonl").write_text(log)
    outputs = {"decisions.jsonl": hashlib.sha256(log.encode()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps({"seed": 2002, "seat": "blue", "outputs": outputs}))
    header = {"format": "tank-teacher-labels", "version": 1, "source": "planner-teacher", "harness": []}
    lines = [
        {"row": i, "request_id": row["request_id"], "labels": label}
        for i, (row, label) in enumerate(zip(rows, TEACHER_LABELS, strict=True))
    ]
    (directory / "labels.jsonl").write_text("".join(json.dumps(line) + "\n" for line in (header, *lines)))
    return directory


def _diagnosis(
    episode: str,
    cause: Cause = "weights",
    release: str | None = "r0003",
    harness: str = "h-seed",
    decisions: tuple[DiagnosedDecision, ...] = (),
) -> Diagnosis:
    return Diagnosis(episode, cause, 0.8, "scripted", release_id=release, harness_id=harness, decisions=decisions)


def _batch(provider: CorrectedDecisionProvider) -> TrainingBatch:
    batch = provider.next_batch(SFT)
    assert batch is not None
    return batch


@pytest.mark.unit
def test_only_weights_diagnoses_produce_decisions(tmp_path: Path) -> None:
    causes: tuple[Cause, ...] = ("weights", "harness", "controller", "uncertain")
    runs = {cause: _round(tmp_path / cause, state=f"{cause} state.") for cause in causes}
    provider = CorrectedDecisionProvider([_diagnosis(cause, cause) for cause in causes], runs)

    assert isinstance(provider, DataProvider)
    batch = _batch(provider)
    assert batch.batch_id == "tank-corrected-decisions"
    assert [dict(d.parts)["essential"][0] for d in batch.decisions] == ["weights state."] * 2
    # The batch is served once, and to typed-decision jobs only.
    assert provider.next_batch(SFT) is None
    tokens = DataContract("token_sequences")
    assert CorrectedDecisionProvider([_diagnosis("weights")], runs).next_batch(tokens) is None
    # A diagnosis of another cause needs no run, and with no weights diagnosis there is nothing to train on.
    assert CorrectedDecisionProvider([_diagnosis("elsewhere", "harness")], {}).next_batch(SFT) is None


@pytest.mark.unit
def test_a_batch_that_mixes_releases_or_harnesses_is_refused(tmp_path: Path) -> None:
    runs = {name: _round(tmp_path / name, state=f"{name} state.") for name in ("a", "b")}

    with pytest.raises(ContractError, match=r"releases \['r0003', 'r0004'\] under harnesses \['h-seed'\]"):
        CorrectedDecisionProvider([_diagnosis("a"), _diagnosis("b", release="r0004")], runs)
    with pytest.raises(ContractError, match=r"releases \['r0003'\] under harnesses \['h-gepa', 'h-seed'\]"):
        CorrectedDecisionProvider([_diagnosis("a"), _diagnosis("b", harness="h-gepa")], runs)
    # A correction that names no release could be for any weights.
    with pytest.raises(ContractError, match="the weights diagnosis of a names no release or no harness"):
        CorrectedDecisionProvider([_diagnosis("a", release=None)], runs)

    # A harness diagnosis of another release adds no decision, so it mixes nothing.
    other = _diagnosis("c", "harness", release="r0004")
    batch = _batch(CorrectedDecisionProvider([_diagnosis("a"), _diagnosis("b"), other], runs))
    assert len(batch.decisions) == 4
    assert batch.source_releases == {"r0003"} and batch.source_harnesses == {"h-seed"}


@pytest.mark.unit
def test_decisions_that_duplicate_a_demonstration_or_each_other_are_dropped(tmp_path: Path) -> None:
    runs = {"first": _round(tmp_path / "first"), "again": _round(tmp_path / "again")}
    motion, weapon = _batch(CorrectedDecisionProvider([_diagnosis("first")], runs)).decisions
    # A teacher's own round taught the weapon decision: the same state, question and choice.
    taught = replace(weapon, group="tank:300000:blue", source_release=None, label_source=None)
    # The same motion state with the policy's choice teaches something else, so it is no duplicate.
    other_choice = replace(motion, chosen=(("motion", "stop_none"),), label_source=None)

    diagnoses = [_diagnosis("first"), _diagnosis("again")]
    provider = CorrectedDecisionProvider(diagnoses, runs, demonstrations=[taught, other_choice])

    # The second round repeats the first, so its decisions go too.
    assert _batch(provider).decisions == (motion,)


@pytest.mark.unit
def test_every_decision_records_its_label_source(tmp_path: Path) -> None:
    runs = {"lost": _round(tmp_path / "lost")}

    teacher = _batch(CorrectedDecisionProvider([_diagnosis("lost")], runs))
    assert [(dict(d.chosen), d.label_source) for d in teacher.decisions] == [
        ({"motion": "forward_none"}, "planner-teacher"),
        ({"weapon": "hold"}, "planner-teacher"),
    ]
    assert all(d.group == "tank:2002:blue" and d.behavior_probs is None for d in teacher.decisions)

    # System 2's own labels: only the decisions its diagnosis names an option for.
    pointed = (DiagnosedDecision(0, "motion"), DiagnosedDecision(1, "weapon", "hold"))
    system2 = CorrectedDecisionProvider([_diagnosis("lost", decisions=pointed)], runs, labels="system2")
    assert [(dict(d.chosen), d.label_source) for d in _batch(system2).decisions] == [({"weapon": "hold"}, "system2")]


@pytest.mark.unit
def test_labels_that_do_not_fit_the_round_are_refused(tmp_path: Path) -> None:
    run = _round(tmp_path / "lost")
    runs = {"lost": run}

    with pytest.raises(ContractError, match="no run directory for gone"):
        CorrectedDecisionProvider([_diagnosis("gone")], runs)
    with pytest.raises(ContractError, match="labels must be one of"):
        CorrectedDecisionProvider([], {}, labels="planner")  # type: ignore[arg-type]
    for step, question, option, reason in (
        (2, "motion", "stop_none", "lost has 2 decisions, and none is step 2"),
        (0, "weapon", "hold", r"step 0 of lost asks \['motion'\], and its system2 labels answer \['weapon'\]"),
        (0, "motion", "reverse_none", "question 'motion' offers no option 'reverse_none'"),
    ):
        diagnosis = _diagnosis("lost", decisions=(DiagnosedDecision(step, question, option),))
        with pytest.raises(ContractError, match=reason):
            CorrectedDecisionProvider([diagnosis], runs, labels="system2")

    # Labels written for another round, or a log changed after its round ended.
    lines = (run / "labels.jsonl").read_text().splitlines()
    (run / "labels.jsonl").write_text("\n".join(lines[:2]) + "\n")
    with pytest.raises(ContractError, match=r"labels\.jsonl does not label the rows of its decisions\.jsonl"):
        CorrectedDecisionProvider([_diagnosis("lost")], runs)
    with (run / "decisions.jsonl").open("a") as log:
        log.write("\n")
    with pytest.raises(ContractError, match="does not match the SHA-256 its manifest records"):
        CorrectedDecisionProvider([_diagnosis("lost")], runs)


@pytest.mark.unit
def test_a_diagnosis_and_a_label_source_are_checked_when_made(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match=r"Diagnosis\.cause must be one of .*got 'chance'"):
        Diagnosis("lost", "chance", 0.5, "a lucky shot")  # type: ignore[arg-type]
    with pytest.raises(ContractError, match=r"confidence is between 0 and 1, got 1\.5"):
        Diagnosis("lost", "weights", 1.5, "sure")
    with pytest.raises(ContractError, match="step must be >= 0"):
        DiagnosedDecision(-1, "motion")
    runs = {"lost": _round(tmp_path / "lost")}
    sample, _ = _batch(CorrectedDecisionProvider([_diagnosis("lost")], runs)).decisions
    with pytest.raises(ContractError, match="label source must not be empty"):
        replace(sample, label_source="")


@pytest.mark.functional
@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_scripted_weights_diagnosis_feeds_an_sft_move_that_publishes(tmp_path: Path) -> None:
    """The tank dry run's campaign with one sft move, fed by a scripted diagnosis instead of teacher rounds.

    The diagnosed round is #204's recorded rollout, labelled here by the
    planner teacher through ``raw-states.mjs``. The gate plays real rounds with
    the stand-in model, which fires once training has raised its quality.
    """
    lost = tmp_path / "episodes/lost"
    lost.mkdir(parents=True)
    for name, content in json.loads(gzip.decompress(FIXTURE.read_bytes())).items():
        (lost / name).write_text(content, encoding="utf-8")
    TankRawStates(INTEGRATION).label([lost])
    config = json.loads(DRY_RUN.read_text())
    config["schedule"] = [{"kind": "train", "algorithm": "sft", "steps": 1, "contract": "typed_decisions"}]
    components = config["components"]
    del components["rollout"]
    components["data"] = {
        "factory": "tests.fakes.tank:scripted_corrections",
        "args": {"runs": {"lost": str(lost)}, "causes": {"lost": "weights", "lost-to-harness": "harness"}},
    }

    run = build(config, REPO, tmp_path / "out")
    [outcome] = run.campaign.run(ScriptedScheduler(run.moves))

    assert outcome.release is not None and outcome.release.release_id == "r0001"
    [batch] = run.context.components["trainer"].batches  # type: ignore[attr-defined]
    _, labels = read_labels(lost)
    rows = [json.loads(line) for line in (lost / "decisions.jsonl").read_text().splitlines()]
    assert batch.batch_id == "tank-corrected-decisions"
    assert [dict(d.chosen) for d in batch.decisions] == [line["labels"] for line in labels]
    # The teacher would have chosen otherwise at 36 of the round's 44 decisions.
    assert sum(dict(d.chosen) != row["chosen"] for d, row in zip(batch.decisions, rows, strict=True)) == 36
    assert {d.label_source for d in batch.decisions} == {"planner-teacher"}
    assert batch.source_releases == {"r0000"}
    assert batch.source_harnesses == {seed_harness(INTEGRATION).spec_id}
    records = [json.loads(line) for line in run.campaign.chain.record_path.read_text().splitlines()]
    assert [(record["op"], record["candidate"]["kind"]) for record in records] == [
        ("publish", "init"),
        ("publish", "train"),
    ]
