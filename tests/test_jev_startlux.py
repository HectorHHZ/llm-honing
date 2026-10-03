"""The StartLux-Decision family: packing and trimming, the answer checks, pins and published hashes, serving, and the
harness check. StartLux needs a GPU and its own environment, so its prompt module and StartLuxDecision are stood in
for with what Honing calls; the GPU check against StartLux's own server is integrations/ra2/parity/startlux_serving.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev import startlux
from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.jev.model import DecisionModel
from honing.backends.jev.server import main as server_main
from honing.backends.jev.startlux import (
    StartLuxDecisionModel,
    StartLuxSettings,
    checked_answer,
    pack_request,
    rendered_rows,
    startlux_snapshot,
    validate_questions,
    verify_published,
    weights_checksum,
)
from honing.backends.ra2 import StartLuxQuestionCheck, seed_harness
from honing.backends.ra2.grader import OVER_BUDGET
from honing.backends.ra2.harness import STARTLUX_LIMITS
from honing.core import CampaignRuleError, HarnessFile
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
    """One token per whitespace-separated word."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": self.encode(text)}


def _from_systemone(state: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {"state": state, "instructions": spec["instructions"], "options": list(spec.get("criteria") or ["y", "n"])}


def _render_ids(row: dict[str, Any], tokenizer: WordTokenizer, max_length: int) -> tuple[list[str], list[str]]:
    """StartLux's prompt, reduced: a fixed system line, the evidence, the question, and one line per option."""
    text = f"system line Evidence: {row['state']} Question: {row['instructions']} Options: {' '.join(row['options'])}"
    return tokenizer.encode(text), row["options"]


PROMPTS = types.SimpleNamespace(MAX_OPTIONS=26, from_systemone=_from_systemone, render_ids=_render_ids)
# The fixed words of a rendered question: "system line Evidence: Question: Which action now? Options: A B".
FIXED_TOKENS = 10


class FakeEngine:
    """StartLuxDecision reduced to what Honing calls: its tokenizer, group size, and decide."""

    tok = WordTokenizer()
    group = 25

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def decide(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        self.calls.append(state)
        return self.answers, {"input_tokens": 42, "output_tokens": 0}


def _measure(state: str, questions: dict[str, Any] = QUESTIONS) -> dict[str, int]:
    model = StartLuxDecisionModel(FakeEngine({}), PROMPTS, {}, 8192)
    return model._measure(state, questions)


def _count(state: str) -> int:
    return len(state.split())


# ---------------------------------------------------------------- packing, measuring, answers


@pytest.mark.unit
def test_the_state_packs_the_four_sections_and_each_question_is_measured_by_its_prompt() -> None:
    state, audit, trimmed = pack_request(PARTS, QUESTIONS, _measure, _count, max_length=100)

    assert state == "\n".join(PARTS["essential"] + PARTS["details"] + PARTS["memory"] + PARTS["logs"])
    assert trimmed == []
    assert audit["g1"] == {
        "stateTokens": 18,
        "stateRoom": 100 - FIXED_TOKENS,
        "inputTokens": 18 + FIXED_TOKENS,
        "maxInputTokens": 100,
        "options": 2,
        "encoding": "startlux-lettered-prompt",
        "stateTruncated": False,
    }


@pytest.mark.unit
def test_trimming_drops_whole_entries_oldest_logs_first_then_newest_memory_and_details() -> None:
    # All of it is 28 tokens; 21 leaves room for the fixed words, the essential entries, and five more words.
    _, _, trimmed = pack_request(PARTS, QUESTIONS, _measure, _count, max_length=21)

    assert trimmed == [
        {"section": "logs", "text": "tick 3 moved"},
        {"section": "logs", "text": "tick 6 moved"},
        {"section": "memory", "text": "built barracks"},
    ]


@pytest.mark.unit
def test_an_overrun_is_refused_in_words_the_grader_scores_as_zero() -> None:
    with pytest.raises(ValueError) as refused:
        pack_request(PARTS, QUESTIONS, _measure, _count, max_length=FIXED_TOKENS + 3)
    assert str(refused.value) == STATE_OVERRUN and any(m in STATE_OVERRUN for m in OVER_BUDGET)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("questions", "message"),
    [
        ({}, "at least one question"),
        ({"g": {"type": "rank", "instructions": "x", "criteria": {"a": "1", "b": "2"}}}, "one of choice, noul, score"),
        ({"g": {"type": "choice", "instructions": " ", "criteria": {"a": "1", "b": "2"}}}, "needs instructions"),
        ({"g": {"type": "choice", "instructions": "x", "criteria": {"a": "1"}}}, "at least two choices"),
        ({"g": {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2"}, "x": 1}}, "Unexpected"),
    ],
)
def test_only_complete_questions_of_the_three_types_are_accepted(questions: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_questions(questions)


@pytest.mark.unit
def test_unknown_sections_and_entries_that_are_not_text_are_refused() -> None:
    with pytest.raises(ValueError, match="Unknown context sections"):
        pack_request({**PARTS, "observer": ["x"]}, QUESTIONS, _measure, _count, max_length=100)
    with pytest.raises(ValueError, match="complete text records"):
        pack_request({"essential": [1]}, QUESTIONS, _measure, _count, max_length=100)


@pytest.mark.unit
def test_a_choice_over_26_options_is_measured_by_the_groups_startlux_renders() -> None:
    wide = {"type": "choice", "instructions": "Which unit?", "criteria": {f"u{i}": f"unit {i}" for i in range(30)}}

    rows = rendered_rows(PROMPTS, "state", wide, group=25)

    assert [len(row["options"]) for row in rows] == [15, 15]
    assert rows[0]["options"][0] == "u0" and rows[1]["options"][-1] == "u29"
    assert len(rendered_rows(PROMPTS, "state", QUESTIONS["g1"], group=25)) == 1


@pytest.mark.unit
def test_an_answer_must_cover_the_options_sum_to_one_and_choose_its_argmax() -> None:
    question = QUESTIONS["g1"]
    good = {"type": "choice", "choice": "B", "confidence": 0.5, "probabilities": {"B": 0.75, "A": 0.25}}
    assert checked_answer("g1", question, good) == {
        "type": "choice",
        "choice": "B",
        "confidence": 0.5,
        "probabilities": {"A": 0.25, "B": 0.75},
    }
    for answer, message in [
        ({**good, "probabilities": {"A": 1.0}}, "cover exactly the offered choices"),
        ({**good, "probabilities": {"A": 0.5, "B": 0.6}}, "do not sum to one"),
        ({**good, "probabilities": {"A": float("nan"), "B": 1.0}}, "Invalid model probability"),
        ({**good, "choice": "A"}, "differs from deterministic argmax"),
        ({**good, "type": "noul"}, "another type"),
    ]:
        with pytest.raises(ValueError, match=message):
            checked_answer("g1", question, answer)


@pytest.mark.unit
def test_yes_no_and_score_answers_keep_startlux_s_own_fields() -> None:
    noul = {"type": "noul", "instructions": "Attack now?"}
    score = {"type": "score", "instructions": "How strong?", "criteria": ["weak", "even", "strong"]}
    scored = {"type": "score", "score": 1.2, "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}}

    assert checked_answer("n", noul, {"type": "noul", "noul": 0.7}) == {"type": "noul", "noul": 0.7}
    assert checked_answer("s", score, scored) == scored
    with pytest.raises(ValueError, match="Invalid model probability"):
        checked_answer("n", noul, {"type": "noul", "noul": 1.5})


@pytest.mark.unit
def test_predict_returns_the_fields_laya_s_callers_read_and_refuses_a_partial_answer() -> None:
    raw = {"g1": {"type": "choice", "choice": "B", "confidence": 0.5, "probabilities": {"A": 0.25, "B": 0.75}}}
    engine = FakeEngine(raw)
    model = StartLuxDecisionModel(engine, PROMPTS, {"family": "startlux", "temperatureByType": {"choice": 1.0}}, 100)

    response = model.predict(PARTS, QUESTIONS)

    assert isinstance(model, DecisionModel)
    assert set(response) == {"answers", "modelResult", "stateText", "tokenAudit", "trimmed", "latencyMs"}
    assert response["answers"] == raw
    assert engine.calls == [response["stateText"]]
    assert response["modelResult"]["usage"] == {"input_tokens": 42, "output_tokens": 0}
    with pytest.raises(ValueError, match="partial question batch"):
        StartLuxDecisionModel(FakeEngine({}), PROMPTS, {"temperatureByType": {}}, 100).predict(PARTS, QUESTIONS)


@pytest.mark.unit
def test_loading_turns_off_the_shared_prefix_path_and_drops_graphs_that_disagree() -> None:
    class Graphed(FakeEngine):
        def __init__(self, difference: float) -> None:
            super().__init__({})
            self.graphs = {"128": object()}
            self.share = True
            self.difference = difference

        def self_test(self, state: str, questions: dict[str, Any]) -> float:
            return self.difference

    kept, dropped = Graphed(0.001), Graphed(0.05)
    assert StartLuxDecisionModel._prepare(kept) == 0.001 and kept.graphs and not kept.share
    assert StartLuxDecisionModel._prepare(dropped) == 0.05 and not dropped.graphs and not dropped.share


@pytest.mark.unit
def test_a_prompt_limit_beyond_one_eager_batch_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 65536"):
        StartLuxSettings(pins_path=tmp_path, source_dir=tmp_path, max_length=65537)


# ---------------------------------------------------------------- pins, hashes, snapshots


@pytest.mark.unit
def test_pins_name_the_startlux_source_and_the_three_sizes() -> None:
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    assert pins["startluxSource"] == {
        "url": "https://github.com/StartLuxLabs/Startlux-Decision.git",
        "revision": "c61bfab3bb4fa3ae9d155d457247a7fd0a299a24",
    }
    assert pins["startluxModels"] == {
        "startlux-0.8b": {
            "repo": "startlux-models/StartLux-Decision-0.8B",
            "revision": "bd4f76a600e23227547fee7bfc1825e12a32764c",
        },
        "startlux-4b": {
            "repo": "startlux-models/StartLux-Decision-4B",
            "revision": "9302aedb7f7bd889994336f2ae919dffbf5dbee1",
        },
        "startlux-27b": {
            "repo": "startlux-models/StartLux-Decision-27B",
            "revision": "95e31cda814fec2a7dc4ac4f8b9b832aa1c9c2e6",
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("revision", "dirty"), [("0000000", ""), ("c61bfab3bb4fa3ae9d155d457247a7fd0a299a24", " M x.py")]
)
def test_the_startlux_source_must_be_the_pinned_unmodified_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, revision: str, dirty: str
) -> None:
    monkeypatch.setattr(
        startlux.subprocess, "check_output", lambda command, text: dirty if "status" in command else revision
    )
    with pytest.raises(ValueError, match="pinned, unmodified source checkout"):
        startlux.import_startlux(tmp_path, json.loads((INTEGRATION / "pins.json").read_text()))


@pytest.mark.unit
def test_an_unknown_model_is_refused_before_any_download(tmp_path: Path) -> None:
    settings = StartLuxSettings(pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, model="startlux-9b")
    with pytest.raises(ValueError, match="Unknown StartLux model 'startlux-9b'"):
        startlux.resolve_checkpoint(settings, json.loads(settings.pins_path.read_text()))


def _shard(name: str, weights: bytes) -> bytes:
    header = json.dumps({name: {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, len(weights)]}}).encode()
    return len(header).to_bytes(8, "little") + header + weights


def _checkpoint(root: Path) -> Path:
    """A Hugging Face cache snapshot: each shard links to a blob named by its SHA-256."""
    snapshot, blobs = root / "snapshots" / "rev", root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": _shard("a", b"1" * 64),
        "model-00002-of-00002.safetensors": _shard("b", b"2" * 64),
    }
    for name, content in shards.items():
        blob = blobs / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        os.symlink(os.path.relpath(blob, snapshot), snapshot / name)
    index = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    text = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 8, "intermediate_size": 3584}
    config = {"text_config": text}
    (snapshot / "config.json").write_text(json.dumps(config))
    return snapshot


@pytest.mark.unit
def test_every_shard_must_match_the_object_id_the_hub_published(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    assert verify_published(checkpoint) == weights_checksum(checkpoint)
    # A blob may link on into a content store named by another hash; the snapshot's own link names the file.
    shard = checkpoint / "model-00001-of-00002.safetensors"
    blob = shard.resolve()
    stored = tmp_path / "store" / "other-hash"
    stored.parent.mkdir()
    blob.rename(stored)
    blob.symlink_to(stored)
    assert verify_published(checkpoint) == weights_checksum(checkpoint)
    shard = (checkpoint / "model-00002-of-00002.safetensors").resolve()
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=r"hash mismatch: model-00002-of-00002\.safetensors"):
        verify_published(checkpoint)


@pytest.mark.unit
def test_a_checkpoint_becomes_a_snapshot_with_the_text_decoder_s_shape(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    snapshot = startlux_snapshot(checkpoint)
    assert snapshot.architecture.hidden_size == 1024 and len(snapshot.architecture.layers) == 24
    assert snapshot.parameters == 64 and snapshot.checkpoint.checksum == weights_checksum(checkpoint)


# ---------------------------------------------------------------- serving, harness check


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "--family startlux needs --startlux-source"),
        (["--startlux-source", ".", "--checkpoint-manifest", "m.json"], "no trained checkpoints yet"),
    ],
)
def test_the_server_names_what_startlux_needs(
    capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        server_main(["--family", "startlux", "--pins", "pins.json", *extra])
    assert message in capsys.readouterr().err


@pytest.mark.unit
def test_the_startlux_check_accepts_the_seed_harness_and_refuses_an_option_over_its_limit() -> None:
    defaults = json.loads((INTEGRATION / "config/questions.json").read_text())
    check = StartLuxQuestionCheck(WordTokenizer(), frozenset(defaults))
    check.check(seed_harness(INTEGRATION))
    long = {**defaults, "intent.group.recruit": " ".join(["word"] * STARTLUX_LIMITS.option)}
    spec = seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(long)))
    with pytest.raises(CampaignRuleError, match=r"intent.group.recruit \(option\) is 257 tokens; the limit is 256"):
        check.check(spec)
