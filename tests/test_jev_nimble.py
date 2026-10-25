"""The Nimble-v4-max family: the question rules, the answer conversion, packing and refusals, pins and hashes,
serving, and the harness check. Nimble needs a GPU and its own environment, so Bespoke's engine module is stood in
for with what Honing calls; the GPU check against the engine itself is integrations/ra2/parity/nimble_serving.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev import nimble
from honing.backends.jev.kev import QUESTION_OVERRUN, STATE_OVERRUN
from honing.backends.jev.model import DecisionModel
from honing.backends.jev.nimble import (
    NimbleDecisionModel,
    NimbleSettings,
    checked_answer,
    engine_state,
    nimble_snapshot,
    validate_questions,
    verify_published,
    verify_source,
)
from honing.backends.jev.server import main as server_main
from honing.backends.jev.startlux import weights_checksum
from honing.backends.ra2 import NimbleQuestionCheck, seed_harness
from honing.backends.ra2.harness import NIMBLE_LIMITS
from honing.core import CampaignRuleError, HarnessFile
from tests.test_jev_startlux import WordTokenizer, _checkpoint
from tests.test_ra2_backend import REPO

INTEGRATION = REPO / "integrations/ra2"
PINS = json.loads((INTEGRATION / "pins.json").read_text())
LEVELS = ["calm", "tense", "critical"]
QUESTIONS = {
    "g1": {"type": "choice", "instructions": "Which action now?", "criteria": {"A": "wait", "B": "build power"}},
    "attack": {"type": "noul", "instructions": "Attack now?", "criteria": None},
    "threat": {"type": "score", "instructions": "How tense is it?", "criteria": LEVELS},
}
PARTS = {
    "essential": ["tick 90 paused", "credits 1200"],
    "details": ["enemy north east"],
    "memory": ["built refinery", "built barracks"],
    "logs": ["tick 3 moved", "tick 6 moved"],
}
RAW = {
    "g1": {"type": "choice", "choice": "B", "probabilities": {"B": 0.75, "A": 0.25}},
    "attack": {"type": "noul", "noul": 0.2},
    "threat": {"type": "choice", "choice": "1", "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25}},
}
# The fixed words of every prompt the stand-in builds: "system Requested field:" and the question's name.
FIXED_TOKENS = 4


class UnsupportedError(ValueError):
    """A stand-in for the Decision Index kit's refusal type, ``Unsupported``."""


def _native_schema(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return body["state"], dict(body["questions"])


def _prepare(tokenizer: WordTokenizer, context: str, schema: dict[str, Any], limit: int) -> types.SimpleNamespace:
    """Nimble's prompt builder, reduced: every question's prompt holds the whole state, then its own name."""
    names = list(schema)
    full_ids = [tokenizer.encode(f"system {context} Requested field: {name}") for name in names]
    return types.SimpleNamespace(names=names, full_ids=full_ids)


MODULE = types.SimpleNamespace(native_question=lambda question: question, native_schema=_native_schema)


class FakeEngine:
    """NimbleEngine reduced to what Honing calls: its tokenizer, its prompt builder, and a call per request."""

    tokenizer = WordTokenizer()
    prepare = staticmethod(_prepare)

    def __init__(self, answers: dict[str, Any], refusal: str | None = None) -> None:
        self.answers = answers
        self.refusal = refusal
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], None]:
        self.calls.append((state, questions))
        if self.refusal:
            raise UnsupportedError(self.refusal)
        return {"model": "nimble", "answers": self.answers, "usage": {"input_tokens": 42}}, None


def _model(engine: FakeEngine, max_length: int = 100) -> NimbleDecisionModel:
    return NimbleDecisionModel(engine, MODULE, UnsupportedError, {"family": "nimble"}, max_length)


# ---------------------------------------------------------------- questions and answers


@pytest.mark.unit
def test_the_three_question_types_are_accepted() -> None:
    validate_questions(QUESTIONS)
    validate_questions({"y": {"type": "noul", "instructions": "Yes?", "criteria": {"yes": "it is", "no": "not"}}})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "message"),
    [
        ({"type": "choice", "instructions": "Pick", "criteria": {"A": "only"}}, "choice criteria"),
        ({"type": "score", "instructions": "Rate", "criteria": {"0": "low", "1": "high"}}, "score criteria"),
        ({"type": "noul", "instructions": "Yes?", "criteria": {"yes": "it is"}}, "noul criteria"),
        ({"type": "span", "instructions": "Find"}, "must be one of choice, noul, score"),
        ({"type": "choice", "instructions": " ", "criteria": {"A": "a", "B": "b"}}, "needs instructions"),
        ({"type": "choice", "instructions": "Pick", "criteria": {"A": "a", "B": "b"}, "when": "x"}, "Unexpected"),
    ],
)
def test_a_question_nimble_would_read_differently_is_refused(question: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_questions({"q": question})


@pytest.mark.unit
def test_answers_take_honing_s_form_and_a_score_becomes_its_expected_level() -> None:
    assert checked_answer("g1", QUESTIONS["g1"], RAW["g1"]) == {
        "type": "choice",
        "choice": "B",
        "confidence": 0.75,
        "probabilities": {"A": 0.25, "B": 0.75},
    }
    assert list(checked_answer("g1", QUESTIONS["g1"], RAW["g1"])["probabilities"]) == ["A", "B"]
    assert checked_answer("attack", QUESTIONS["attack"], RAW["attack"]) == {"type": "noul", "noul": 0.2}
    assert checked_answer("threat", QUESTIONS["threat"], RAW["threat"]) == {
        "type": "score",
        "score": 1.0,
        "confidence": 0.5,
        "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
        "legend": {"0": "calm", "1": "tense", "2": "critical"},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("key", "answer", "message"),
    [
        ("g1", {"type": "choice", "choice": "A", "probabilities": {"A": 0.25, "B": 0.75}}, "highest probability"),
        ("g1", {"type": "choice", "choice": "B", "probabilities": {"B": 1.0}}, "cover exactly"),
        ("g1", {"type": "choice", "choice": "B", "probabilities": {"A": 0.5, "B": 0.6}}, "Invalid model"),
        ("g1", {"type": "noul", "noul": 0.5}, "another type"),
        ("attack", {"type": "noul", "noul": 1.5}, "yes/no answer"),
        ("threat", {"type": "choice", "choice": "1", "probabilities": {"0": 0.5, "1": 0.5}}, "cover exactly"),
    ],
)
def test_a_partial_or_invalid_answer_is_refused(key: str, answer: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        checked_answer(key, QUESTIONS[key], answer)


@pytest.mark.unit
def test_an_empty_state_is_measured_as_the_engine_reads_it() -> None:
    assert engine_state("") == '""' and engine_state(" \n") == '" \\n"' and engine_state("tick 3") == "tick 3"


# ---------------------------------------------------------------- predict, packing, refusals


@pytest.mark.unit
def test_predict_returns_the_fields_laya_s_callers_read_and_sends_the_packed_state() -> None:
    engine = FakeEngine(RAW)
    model = _model(engine)

    response = model.predict(PARTS, QUESTIONS)

    assert isinstance(model, DecisionModel)
    assert set(response) == {"answers", "modelResult", "stateText", "tokenAudit", "trimmed", "latencyMs"}
    assert response["stateText"] == "\n".join(PARTS["essential"] + PARTS["details"] + PARTS["memory"] + PARTS["logs"])
    assert engine.calls == [(response["stateText"], QUESTIONS)]
    assert response["answers"]["threat"]["type"] == "score" and response["answers"]["g1"]["choice"] == "B"
    # The permutation client and the audit require modelResult.answers to be the answers themselves.
    assert response["modelResult"] == {
        "answers": response["answers"],
        "engineAnswers": RAW,
        "usage": {"input_tokens": 42},
    }
    assert response["tokenAudit"]["g1"] == {
        "stateTokens": 18,
        "stateRoom": 100 - FIXED_TOKENS,
        "inputTokens": 18 + FIXED_TOKENS,
        "maxInputTokens": 100,
        "options": 2,
        "encoding": "nimble-compact-prompt",
        "stateTruncated": False,
    }
    assert response["tokenAudit"]["threat"]["options"] == 3


@pytest.mark.unit
def test_trimming_drops_whole_entries_oldest_logs_first_then_newest_memory() -> None:
    # The whole state is 18 words; 16 leaves room for the fixed words and 12 words of state.
    response = _model(FakeEngine(RAW), max_length=16).predict(PARTS, QUESTIONS)

    assert response["trimmed"] == [
        {"section": "logs", "text": "tick 3 moved"},
        {"section": "logs", "text": "tick 6 moved"},
    ]


@pytest.mark.unit
def test_overruns_are_refused_in_words_the_grader_scores_as_zero() -> None:
    with pytest.raises(ValueError) as refused:
        _model(FakeEngine(RAW), max_length=FIXED_TOKENS + 3).predict(PARTS, QUESTIONS)
    assert str(refused.value) == STATE_OVERRUN

    engine = FakeEngine(RAW, refusal="maximum context length: Longest prompt has 9 tokens; limit is 8.")
    with pytest.raises(ValueError) as refused:
        _model(engine).predict(PARTS, QUESTIONS)
    assert str(refused.value) == QUESTION_OVERRUN.format(
        key="g1, attack, threat", detail="maximum context length: Longest prompt has 9 tokens; limit is 8."
    )

    with pytest.raises(ValueError, match="Nimble refused the request: out of GPU memory"):
        _model(FakeEngine(RAW, refusal="out of GPU memory for one 9-token prompt")).predict(PARTS, QUESTIONS)


@pytest.mark.unit
def test_a_partial_question_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="partial question batch"):
        _model(FakeEngine({"g1": RAW["g1"]})).predict(PARTS, QUESTIONS)


@pytest.mark.unit
def test_a_prompt_limit_beyond_the_model_s_context_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 32768"):
        NimbleSettings(pins_path=tmp_path, source_dir=tmp_path, kit_dir=tmp_path, max_length=32769)


# ---------------------------------------------------------------- pins, source, weights


@pytest.mark.unit
def test_pins_name_the_source_snapshot_the_kit_and_the_model() -> None:
    source = PINS["nimbleSource"]
    assert (source["repo"], source["repoType"], source["revision"]) == (
        "bespokelabs/nimble-v4-max-decision-index",
        "dataset",
        "adea14289afa29dca3ff38b1fcb76e58f6add230",
    )
    assert len(source["files"]) == 10 and all(name.startswith("code/nimble/") for name in source["files"])
    assert PINS["decisionIndexKit"] == {
        "url": "https://github.com/apolinario/decision-index.git",
        "revision": "25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d",
    }
    assert PINS["nimbleModels"] == {
        "nimble-v4-max": {
            "repo": "bespokelabs/Nimble-v4-max",
            "revision": "655aa7051c2ab39660193ebb70c8926d3d2cb7aa",
            "base": "Qwen/Qwen3.8-27B",
            "baseRevision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        }
    }


def _source(root: Path) -> tuple[Path, dict[str, Any]]:
    """A source snapshot whose files match a pins entry made for it."""
    files = {"code/nimble/__init__.py": b"", "code/nimble/evaluation/decision_index_engine.py": b"ENGINE = 1\n"}
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    pins = {
        "nimbleSource": {
            "revision": "rev",
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        },
        "decisionIndexKit": {"revision": "kit"},
    }
    return root, pins


def _git(monkeypatch: pytest.MonkeyPatch, revision: str = "kit", dirty: str = "") -> None:
    monkeypatch.setattr(
        nimble.subprocess, "check_output", lambda command, text: dirty if "status" in command else revision
    )


@pytest.mark.unit
def test_the_pinned_source_and_kit_are_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, pins = _source(tmp_path)
    _git(monkeypatch)
    assert verify_source(source, tmp_path, pins) == ("rev", "kit")


@pytest.mark.unit
def test_a_changed_or_added_source_file_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, pins = _source(tmp_path)
    _git(monkeypatch)
    (source / "code/nimble/evaluation/decision_index_engine.py").write_text("ENGINE = 2\n")
    with pytest.raises(ValueError, match=r"decision_index_engine\.py differs from its pinned SHA-256"):
        verify_source(source, tmp_path, pins)
    source, pins = _source(tmp_path)
    (source / "code/nimble/extra.py").write_text("")
    with pytest.raises(ValueError, match="differ from the pinned file list"):
        verify_source(source, tmp_path, pins)


@pytest.mark.unit
@pytest.mark.parametrize(("revision", "dirty"), [("other", ""), ("kit", " M decision_index/engines/base.py")])
def test_the_kit_must_be_the_pinned_unmodified_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, revision: str, dirty: str
) -> None:
    source, pins = _source(tmp_path)
    _git(monkeypatch, revision, dirty)
    with pytest.raises(ValueError, match="pinned, unmodified Decision Index kit"):
        verify_source(source, tmp_path, pins)


def _nimble_checkpoint(root: Path) -> Path:
    """A cache snapshot with the shard hashes the model publishes beside its weights."""
    checkpoint = _checkpoint(root)
    listed = {path.name: path.resolve().name for path in checkpoint.glob("*.safetensors")}
    (checkpoint / "weights_sha256.json").write_text(json.dumps(listed))
    return checkpoint


@pytest.mark.unit
def test_every_shard_must_match_the_hub_s_object_id_and_the_model_s_own_list(tmp_path: Path) -> None:
    checkpoint = _nimble_checkpoint(tmp_path)
    assert verify_published(checkpoint) == weights_checksum(checkpoint)
    listed = json.loads((checkpoint / "weights_sha256.json").read_text())
    listed["model-00001-of-00002.safetensors"] = "0" * 64
    (checkpoint / "weights_sha256.json").write_text(json.dumps(listed))
    with pytest.raises(ValueError, match=r"hash mismatch: model-00001-of-00002\.safetensors"):
        verify_published(checkpoint)


@pytest.mark.unit
def test_a_changed_shard_is_refused(tmp_path: Path) -> None:
    checkpoint = _nimble_checkpoint(tmp_path)
    shard = (checkpoint / "model-00002-of-00002.safetensors").resolve()
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=r"hash mismatch: model-00002-of-00002\.safetensors"):
        verify_published(checkpoint)


@pytest.mark.unit
def test_an_unknown_model_is_refused_before_any_download(tmp_path: Path) -> None:
    settings = NimbleSettings(pins_path=tmp_path, source_dir=tmp_path, kit_dir=tmp_path, model="nimble-v5")
    with pytest.raises(ValueError, match="Unknown Nimble model 'nimble-v5'"):
        nimble.resolve_checkpoint(settings, PINS)


@pytest.mark.unit
def test_a_checkpoint_whose_contract_names_another_tokenizer_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = _nimble_checkpoint(tmp_path)
    (checkpoint / "schema_config.json").write_text(json.dumps({"model": "Qwen/Qwen3.8-27B", "revision": "other"}))
    hub = types.SimpleNamespace(snapshot_download=lambda *args, **kwargs: str(checkpoint))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    settings = NimbleSettings(pins_path=tmp_path, source_dir=tmp_path, kit_dir=tmp_path)
    with pytest.raises(ValueError, match="tokenizer contract differs"):
        nimble.resolve_checkpoint(settings, PINS)


@pytest.mark.unit
def test_a_checkpoint_becomes_a_snapshot_with_the_text_decoder_s_shape(tmp_path: Path) -> None:
    checkpoint = _nimble_checkpoint(tmp_path)
    snapshot = nimble_snapshot(checkpoint)
    assert snapshot.architecture.hidden_size == 1024 and len(snapshot.architecture.layers) == 24
    assert snapshot.checkpoint.checksum == weights_checksum(checkpoint)


# ---------------------------------------------------------------- serving, harness check


@pytest.mark.unit
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "--family nimble needs --nimble-source and --nimble-kit"),
        (["--nimble-source", "."], "--family nimble needs --nimble-source and --nimble-kit"),
        (["--nimble-source", ".", "--nimble-kit", ".", "--checkpoint-manifest", "m.json"], "no trained checkpoints"),
    ],
)
def test_the_server_names_what_nimble_needs(
    capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        server_main(["--family", "nimble", "--pins", "pins.json", *extra])
    assert message in capsys.readouterr().err


@pytest.mark.unit
def test_the_nimble_check_accepts_the_seed_harness_and_refuses_an_option_over_its_limit() -> None:
    defaults = json.loads((INTEGRATION / "config/questions.json").read_text())
    check = NimbleQuestionCheck(WordTokenizer(), frozenset(defaults))
    check.check(seed_harness(INTEGRATION))
    long = {**defaults, "intent.group.recruit": " ".join(["word"] * NIMBLE_LIMITS.option)}
    spec = seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(long)))
    with pytest.raises(CampaignRuleError, match=r"intent.group.recruit \(option\) is 257 tokens; the limit is 256"):
        check.check(spec)


@pytest.mark.unit
@pytest.mark.parametrize("limit", [256, 8192])
def test_warmup_covers_single_and_shared_prefix_paths_within_the_context(limit: int) -> None:
    class WarmEngine(FakeEngine):
        def __call__(self, state: str, questions: dict[str, Any]) -> tuple[dict[str, Any], None]:
            self.calls.append((state, questions))
            answers = {}
            for name, question in questions.items():
                keys = list(question["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {key: float(key == keys[0]) for key in keys},
                }
            return {"answers": answers}, None

    engine = WarmEngine({})
    model = _model(engine, max_length=limit)
    model.warmup()
    assert {len(questions) for _, questions in engine.calls} == {1, 2, 3}
    lengths = [(len(questions), max(model._measure(state, questions).values())) for state, questions in engine.calls]
    assert all(length <= limit for _, length in lengths)
    if limit > 4096:
        assert any(count == 1 and length > 4096 for count, length in lengths)
        assert any(count > 1 and length > 256 for count, length in lengths)


@pytest.mark.unit
def test_a_warmup_failure_propagates_before_load_yields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from contextlib import nullcontext

    engine = types.SimpleNamespace(warmup=lambda: None, shared_prefix=True)
    module = types.SimpleNamespace(NimbleEngine=lambda **kwargs: engine)
    monkeypatch.setattr(nimble, "import_nimble", lambda *args: (module, UnsupportedError, "source", "kit"))
    monkeypatch.setattr(nimble, "gpu_lock", lambda *args: nullcontext("gpu-uuid"))
    monkeypatch.setattr(nimble, "resolve_checkpoint", lambda *args: {"path": str(tmp_path)})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="test"))

    def fail(self: NimbleDecisionModel) -> None:
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(NimbleDecisionModel, "warmup", fail)
    pins = tmp_path / "pins.json"
    pins.write_text("{}")
    with (
        pytest.raises(RuntimeError, match="warmup failed"),
        NimbleDecisionModel.load(NimbleSettings(pins_path=pins, source_dir=tmp_path, kit_dir=tmp_path)),
    ):
        pytest.fail("The caller must not see a ready model after failed warmup")
