"""The NanoJev family: packing and trimming as #49's adapter did, the answer checks, pins and published checksums,
serving, the harness check, and the trainer's refusals. NanoJev's predictor needs a GPU and its own environment, so
its prepare_examples and DecisionPredictor are stood in for; the GPU checks against #49 are in
integrations/ra2/parity/ (nanojev_serving.py, nanojev_training.py)."""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev import nanojev
from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.jev.nanojev import (
    NanoJevDecisionModel,
    NanoJevSettings,
    checked_answer,
    measure,
    nanojev_snapshot,
    pack_request,
    validate_questions,
    verify_published,
    weights_checksum,
)
from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings
from honing.backends.jev.server import main as server_main
from honing.backends.ra2 import NanoJevQuestionCheck, seed_harness
from honing.backends.ra2.grader import OVER_BUDGET
from honing.backends.ra2.harness import NANOJEV_LIMITS
from honing.core import CampaignRuleError, ContractError, HarnessFile, TrainJobSpec, UpdatePolicy
from tests.fakes.data import token_batch
from tests.test_ra2_backend import REPO

INTEGRATION = REPO / "integrations/ra2"
QUESTIONS = {
    "g1": {"type": "choice", "instructions": "Which action now?", "criteria": {"A": "wait", "B": "build power"}}
}
PARTS = {
    "essential": ["tick 90 paused", "credits 1200"],
    "details": ["enemy north east"],
    "memory": ["built refinery", "built barracks"],
    "logs": ["tick 3 moved", "tick 6 moved"],
}


class WordTokenizer:
    """One token per whitespace-separated word; token ids are word lengths."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}


def prepare_examples(request: dict[str, Any], tokenizer: WordTokenizer, max_length: int) -> list[dict[str, Any]]:
    """NanoJev's encoding, reduced: a path is the state, the question, and one candidate, then an end token."""
    (row,) = request["states"]
    examples = []
    for qid, q in row["questions"].items():
        prefix = tokenizer.encode(f"State: {row['state']} Question: {q['instructions']}")
        leaves = [prefix + tokenizer.encode(f"Candidate: {k}: {v} Decision:") + [0] for k, v in q["criteria"].items()]
        examples.append(
            {
                "qid": qid,
                "state_id": row["id"],
                "type": "choice",
                "candidate_ids": list(q["criteria"]),
                "leaf_tokens": leaves,
            }
        )
    return examples


def count(state: str) -> dict[str, dict[str, Any]]:
    return measure(prepare_examples, WordTokenizer(), state, QUESTIONS)


# ---------------------------------------------------------------- packing, measuring, answers


@pytest.mark.unit
def test_the_state_packs_the_four_sections_and_each_question_is_measured_by_its_longest_path() -> None:
    state, audit, trimmed = pack_request(PARTS, QUESTIONS, count, max_length=100)

    assert state == "\n".join(PARTS["essential"] + PARTS["details"] + PARTS["memory"] + PARTS["logs"])
    assert trimmed == []
    entry = audit["g1"]
    # path: "State:" + 18 state words + "Question:" + 3 words, then "Candidate: B: build power Decision:" + end token
    assert (entry["stateTokens"], entry["inputTokens"]) == (18, 1 + 18 + 1 + 3 + 5 + 1)
    assert entry["stateRoom"] == 100 - (29 - 18) and entry["maxInputTokens"] == 100
    assert entry["options"] == 2 and entry["stateTruncated"] is False
    assert len(entry["candidatePathSha256"]) == 2


@pytest.mark.unit
def test_trimming_drops_whole_entries_oldest_logs_first_then_newest_memory_and_details() -> None:
    state, _, trimmed = pack_request(PARTS, QUESTIONS, count, max_length=19)

    assert [t["text"] for t in trimmed] == ["tick 3 moved", "tick 6 moved", "built barracks", "built refinery"]
    assert state == "tick 90 paused\ncredits 1200\nenemy north east"


@pytest.mark.unit
def test_an_overrun_is_refused_in_words_the_grader_scores_as_zero() -> None:
    with pytest.raises(ValueError) as refused:
        pack_request(PARTS, QUESTIONS, count, max_length=10)
    assert str(refused.value) == STATE_OVERRUN and any(m in STATE_OVERRUN for m in OVER_BUDGET)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ({"g1": {**QUESTIONS["g1"], "extra": 1}}, "Unexpected question fields"),
        ({"g1": {**QUESTIONS["g1"], "type": "score"}}, "Only explicit choice questions"),
        ({"g1": {**QUESTIONS["g1"], "criteria": {"A": "only one"}}}, "2-255 ordered choices"),
    ],
)
def test_only_complete_choice_questions_are_accepted(questions: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_questions(questions)


@pytest.mark.unit
def test_a_measure_that_reorders_or_drops_options_is_refused() -> None:
    def reordered(request: dict[str, Any], tokenizer: WordTokenizer, limit: int) -> list[dict[str, Any]]:
        (example,) = prepare_examples(request, tokenizer, limit)
        return [{**example, "candidate_ids": example["candidate_ids"][::-1]}]

    with pytest.raises(ValueError, match="reordered offered choices"):
        measure(reordered, WordTokenizer(), "state", QUESTIONS)


@pytest.mark.unit
def test_an_answer_must_cover_the_options_sum_to_one_and_choose_its_argmax() -> None:
    criteria = {"A": "wait", "B": "build"}
    assert checked_answer(criteria, {"B": 0.75, "A": 0.25}, "B") == {
        "choice": "B",
        "probabilities": {"A": 0.25, "B": 0.75},
    }
    with pytest.raises(ValueError, match="cover exactly"):
        checked_answer(criteria, {"A": 1.0})
    with pytest.raises(ValueError, match="do not sum to one"):
        checked_answer(criteria, {"A": 0.5, "B": 0.4})
    with pytest.raises(ValueError, match="differs from deterministic argmax"):
        checked_answer(criteria, {"A": 0.25, "B": 0.75}, "A")


class FakePredictor:
    """NanoJev's DecisionPredictor surface: a tokenizer, a length limit, and predict."""

    tokenizer = WordTokenizer()
    limit = 100

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    def predict(self, request: dict[str, Any], batch_questions: int, temperature: float) -> dict[str, Any]:
        assert (batch_questions, temperature) == (1, 1.0)
        return {"states": [{"id": "decision", "answers": self.answers}], "execution": {}}


@pytest.mark.unit
def test_predict_returns_the_fields_laya_s_callers_read_and_refuses_a_partial_answer() -> None:
    good = {"g1": {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}}
    model = NanoJevDecisionModel(FakePredictor(good), prepare_examples, {"family": "nanojev"})

    response = model.predict(PARTS, QUESTIONS)

    assert set(response) == {"answers", "modelResult", "stateText", "tokenAudit", "trimmed", "latencyMs"}
    assert response["answers"] == {"g1": {"choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}}
    with pytest.raises(ValueError, match="partial question batch"):
        NanoJevDecisionModel(FakePredictor({}), prepare_examples, {}).predict(PARTS, QUESTIONS)


# ---------------------------------------------------------------- pins, checksums, snapshots


@pytest.mark.unit
def test_pins_name_the_nanojev_source_release_and_base() -> None:
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    # the same source and model revisions #49 pinned in league/model-sources.json
    assert pins["nanojevSource"]["revision"] == "76fdfc9ecdca45a9bcef17991a07d3041a87685a"
    assert pins["nanojevModels"] == {
        "nanojev": {
            "repo": "C-Tianyu/NanoJev",
            "revision": "047b927b30882a1138fc504821b82ac145a4b81a",
            "base": "Qwen/Qwen3-0.6B",
            "baseRevision": "c1899de289a04d12100db370d81485cdf75e47ca",
        }
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("revision", "dirty"), [("0000000", ""), ("76fdfc9ecdca45a9bcef17991a07d3041a87685a", " M x.py")]
)
def test_the_nanojev_source_must_be_the_pinned_unmodified_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, revision: str, dirty: str
) -> None:
    monkeypatch.setattr(
        nanojev.subprocess, "check_output", lambda command, text: dirty if "status" in command else revision
    )
    with pytest.raises(ValueError, match="pinned, unmodified source checkout"):
        nanojev.import_nanojev(tmp_path, json.loads((INTEGRATION / "pins.json").read_text()))


def _checkpoint(directory: Path, weights: bytes = b"weights") -> Path:
    header = json.dumps({"backbone.w": {"dtype": "F32", "shape": [4, 8], "data_offsets": [0, 0]}}).encode()
    files = {
        "best.safetensors": len(header).to_bytes(8, "little") + header + weights,
        "config.json": json.dumps(
            {"model": "Qwen/Qwen3-0.6B", "resolved_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
        ).encode(),
        "backbone_config/config.json": json.dumps(
            {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "intermediate_size": 3072}
        ).encode(),
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    for name, content in files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(content)
    published = [{"path": n, "sha256": hashlib.sha256(c).hexdigest()} for n, c in files.items()]
    (directory / "SHA256_MANIFEST.json").write_text(json.dumps({"files": published}))
    return directory


@pytest.mark.unit
def test_the_files_nanojev_loads_must_match_the_published_checksums(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "ck")
    verify_published(checkpoint)
    (checkpoint / "tokenizer/tokenizer.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match=r"hash mismatch: tokenizer/tokenizer\.json"):
        verify_published(checkpoint)


@pytest.mark.unit
def test_a_checkpoint_becomes_a_snapshot_with_the_backbone_s_shape(tmp_path: Path) -> None:
    snapshot = nanojev_snapshot(_checkpoint(tmp_path / "ck"))
    assert snapshot.architecture.hidden_size == 1024 and len(snapshot.architecture.layers) == 28
    assert snapshot.parameters == 32 and snapshot.checkpoint.checksum == weights_checksum(tmp_path / "ck")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"kind": "trained-kev"}, "Unrecognized trained checkpoint manifest"),
        (
            {"kind": "trained-nanojev", "sourceRevision": "rev", "parent": {"revision": "other"}},
            "parent does not match",
        ),
    ],
)
def test_a_trained_manifest_must_name_this_source_and_release(
    tmp_path: Path, manifest: dict[str, Any], message: str
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(manifest))
    settings = NanoJevSettings(pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, checkpoint_manifest=path)
    with pytest.raises(ValueError, match=message):
        nanojev.resolve_checkpoint(settings, json.loads(settings.pins_path.read_text()), "rev")


@pytest.mark.unit
def test_a_trained_checkpoint_must_match_its_recorded_weights(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "ck")
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    manifest = {
        "kind": "trained-nanojev", "sourceRevision": "rev", "repo": "local", "revision": "r", "path": str(checkpoint),
        "parent": {"revision": pins["nanojevModels"]["nanojev"]["revision"]}, "weightsSha256": "0" * 64,
    }  # fmt: skip
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(manifest))
    settings = NanoJevSettings(pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, checkpoint_manifest=path)
    with pytest.raises(ValueError, match="weight hash mismatch"):
        nanojev.resolve_checkpoint(settings, pins, "rev")
    path.write_text(json.dumps({**manifest, "weightsSha256": weights_checksum(checkpoint)}))
    assert nanojev.resolve_checkpoint(settings, pins, "rev")["base"] == "Qwen/Qwen3-0.6B"


# ---------------------------------------------------------------- serving, harness check, trainer


@pytest.mark.unit
def test_the_server_names_the_setting_nanojev_needs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        server_main(["--family", "nanojev", "--pins", "pins.json"])
    assert "--family nanojev needs --nanojev-source" in capsys.readouterr().err


@pytest.mark.unit
def test_the_nanojev_check_accepts_the_seed_harness_and_refuses_an_option_over_its_limit() -> None:
    defaults = json.loads((INTEGRATION / "config/questions.json").read_text())
    check = NanoJevQuestionCheck(WordTokenizer(), frozenset(defaults))
    check.check(seed_harness(INTEGRATION))
    long = {**defaults, "intent.group.recruit": " ".join(["word"] * NANOJEV_LIMITS.option)}
    spec = seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(long)))
    with pytest.raises(CampaignRuleError, match=r"intent.group.recruit \(option\) is 260 tokens; the limit is 256"):
        check.check(spec)


def _trainer(tmp_path: Path) -> NanoJevTrainer:
    return NanoJevTrainer(
        NanoJevTrainerSettings(pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, work_dir=tmp_path)
    )


@pytest.mark.unit
def test_the_trainer_refuses_jobs_and_batches_it_cannot_train(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="sft or grpo on typed_decisions"):
        _trainer(tmp_path).train(TrainJobSpec("sft"), None, None)  # type: ignore[arg-type]
    lora = TrainJobSpec("sft", contract="typed_decisions", update_policy=UpdatePolicy(kind="lora", lora_rank=8))
    with pytest.raises(ContractError, match="cannot apply a lora update"):
        _trainer(tmp_path).train(lora, None, None)  # type: ignore[arg-type]
    job = TrainJobSpec("sft", contract="typed_decisions")
    with pytest.raises(ContractError, match="needs a typed_decisions batch"):
        _trainer(tmp_path).train(job, None, token_batch())  # type: ignore[arg-type]
    assert NanoJevTrainer.support.contracts == frozenset({"typed_decisions"})
    assert types.SimpleNamespace(**{"algorithms": dict(NanoJevTrainer.support.algorithms)}).algorithms == {
        "sft": "native",
        "grpo": "native",
    }
