"""The Kev family: request packing and trimming, the response Laya's callers read, pins, serving, the harness check,
and the trainer's refusals. Kev itself (Python 3.12, torch) is not installed here, so its api and model modules are
stood in for with the few functions Honing calls; the GPU checks against the real Kev are in
integrations/ra2/parity/ (kev_serving.py, kev_training.py)."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from honing.backends.jev import kev as kev_backend
from honing.backends.jev.kev import (
    QUESTION_OVERRUN,
    STATE_OVERRUN,
    KevDecisionModel,
    kev_snapshot,
    prepare_request,
    weights_checksum,
)
from honing.backends.jev.kev_trainer import KevTrainer, KevTrainerSettings
from honing.backends.jev.server import main as server_main
from honing.backends.ra2 import KevQuestionCheck, RA2Grader, load_suite, seed_harness
from honing.backends.ra2.grader import OVER_BUDGET
from honing.backends.ra2.harness import KEV_LIMITS
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    HarnessFile,
    TrainJobSpec,
    UpdatePolicy,
)
from tests.fakes.data import token_batch
from tests.test_ra2_backend import REPO, SUITE_PATH

INTEGRATION = REPO / "integrations/ra2"


class WordTokenizer:
    """One token per whitespace-separated word."""

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


@pytest.fixture()
def fake_kev(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kev's api and model modules, reduced to what Honing calls, with Kev's choice rendering ('name: desc')."""

    class SystemOneRequest:
        def __init__(self, state: str, questions: dict[str, dict[str, object]]) -> None:
            self.state, self.questions = state, questions

    def to_record(request: SystemOneRequest) -> tuple[dict[str, object], list[dict[str, object]]]:
        questions, meta = [], []
        for key, q in request.questions.items():
            criteria = q["criteria"]
            assert isinstance(criteria, dict)
            questions.append(
                {"instr": q["instructions"], "options": [f"{k}: {v}" for k, v in criteria.items()], "label": 0}
            )
            meta.append({"id": key, "type": "choice", "keys": list(criteria)})
        return {"state": request.state, "questions": questions}, meta

    def to_answers(rows: list[list[float]], meta: list[dict[str, object]]) -> dict[str, object]:
        out = {}
        for p, m in zip(rows, meta, strict=True):
            keys = m["keys"]
            assert isinstance(keys, list)
            best = max(range(len(p)), key=lambda i: p[i])
            out[m["id"]] = {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, p, strict=True))}
        return out

    api = types.ModuleType("kev.api")
    api.SystemOneRequest, api.to_record, api.to_answers = SystemOneRequest, to_record, to_answers  # type: ignore[attr-defined]
    model = types.ModuleType("kev.model")
    model.user_tokens = lambda tok, text: tok(text)["input_ids"]  # type: ignore[attr-defined]
    package = types.ModuleType("kev")
    monkeypatch.setitem(sys.modules, "kev", package)
    monkeypatch.setitem(sys.modules, "kev.api", api)
    monkeypatch.setitem(sys.modules, "kev.model", model)


QUESTIONS = {
    "g1": {"type": "choice", "instructions": "Which action now?", "criteria": {"A": "wait here", "B": "build power"}}
}
PARTS = {
    "essential": ["tick 90 paused", "credits 1200"],
    "details": ["enemy north east"],
    "memory": ["built refinery", "built barracks"],
    "logs": ["tick 3 moved", "tick 6 moved"],
}


# ---------------------------------------------------------------- packing and trimming


@pytest.mark.unit
def test_the_state_packs_the_four_sections_in_order_and_is_audited(fake_kev: None) -> None:
    state, audit, trimmed = prepare_request(WordTokenizer(), PARTS, QUESTIONS, max_state=100, max_branch=100)

    assert state == "\n".join(PARTS["essential"] + PARTS["details"] + PARTS["memory"] + PARTS["logs"])
    assert trimmed == []
    # state: <state> + 18 words; branch: <q> + 3 instruction words + 2 options of (<opt> + 3 words + </opt>) + <decide>
    assert audit["g1"] == {
        "stateTokens": 1 + 18,
        "stateRoom": 100 - 15,
        "inputTokens": 19 + 15,
        "options": 2,
        "stateTruncated": False,
    }


@pytest.mark.unit
def test_trimming_drops_whole_entries_oldest_logs_first_then_newest_memory_and_details(fake_kev: None) -> None:
    # essential and details are 1 + 8 tokens; every other entry has to go to fit 10
    state, _, trimmed = prepare_request(WordTokenizer(), PARTS, QUESTIONS, max_state=10, max_branch=100)

    assert [t["text"] for t in trimmed] == ["tick 3 moved", "tick 6 moved", "built barracks", "built refinery"]
    assert state == "tick 90 paused\ncredits 1200\nenemy north east"


@pytest.mark.unit
def test_an_overrun_is_refused_in_words_the_grader_scores_as_zero(fake_kev: None) -> None:
    with pytest.raises(ValueError) as question:
        prepare_request(WordTokenizer(), PARTS, QUESTIONS, max_state=100, max_branch=15)
    with pytest.raises(ValueError) as state:
        prepare_request(WordTokenizer(), PARTS, QUESTIONS, max_state=4, max_branch=100)

    assert str(state.value) == STATE_OVERRUN
    assert str(question.value) == QUESTION_OVERRUN.format(key="g1", detail="question 15/14 tokens")
    assert any(marker in str(question.value) for marker in OVER_BUDGET)
    assert any(marker in str(state.value) for marker in OVER_BUDGET)


# ---------------------------------------------------------------- the response


class FakeKevModel:
    """The loaded model's surface KevDecisionModel.predict uses."""

    class head:  # noqa: N801
        temperature = 2.35

    def encode(self, tok: object, record: dict[str, object], **limits: object) -> dict[str, object]:
        assert limits == {"max_state": 100, "max_branch": 100, "strict": True}
        return {"record": record}

    def probs_and_prefix(self, encoded: dict[str, object]) -> tuple[list[object], object]:
        import array

        return [types.SimpleNamespace(tolist=lambda: list(array.array("d", [0.25, 0.75])))], None


@pytest.mark.unit
def test_predict_returns_the_fields_laya_s_callers_read(fake_kev: None, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: __import__("contextlib").nullcontext()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    model = KevDecisionModel(WordTokenizer(), FakeKevModel(), {"family": "kev"}, max_state=100, max_branch=100)

    response = model.predict(PARTS, QUESTIONS)

    assert set(response) == {"answers", "modelResult", "stateText", "tokenAudit", "trimmed", "latencyMs"}
    assert response["answers"] == {"g1": {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}}
    assert response["modelResult"]["temperature"] == 2.35  # type: ignore[index]
    assert response["tokenAudit"]["g1"]["options"] == 2  # type: ignore[index]
    assert model.metadata == {"family": "kev"}


# ---------------------------------------------------------------- checkpoints and pins


def _checkpoint(directory: Path, adapter: bytes = b"adapter", head: bytes = b"head") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"lora_A.weight": {"dtype": "F32", "shape": [16, 1024], "data_offsets": [0, 0]}}).encode()
    (directory / "adapter_model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + adapter)
    (directory / "head.pt").write_bytes(head)
    return directory


@pytest.mark.unit
def test_the_checksum_covers_the_adapter_and_the_head(tmp_path: Path) -> None:
    a = weights_checksum(_checkpoint(tmp_path / "a"))
    assert a == weights_checksum(_checkpoint(tmp_path / "b"))
    assert a != weights_checksum(_checkpoint(tmp_path / "c", head=b"other head"))
    assert a != weights_checksum(_checkpoint(tmp_path / "d", adapter=b"other adapter"))


@pytest.mark.unit
def test_a_checkpoint_becomes_a_snapshot_with_the_base_s_shape(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 1024,
                    "num_hidden_layers": 24,
                    "num_attention_heads": 8,
                    "intermediate_size": 3584,
                }
            }
        )
    )
    snapshot = kev_snapshot(_checkpoint(tmp_path / "ck"), config)

    assert snapshot.architecture.hidden_size == 1024 and len(snapshot.architecture.layers) == 24
    assert snapshot.parameters == 16 * 1024
    assert snapshot.checkpoint.checksum == weights_checksum(tmp_path / "ck")


@pytest.mark.unit
def test_the_kev_source_must_be_at_its_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(kev_backend.subprocess, "check_output", lambda *a, **k: "0000000\n")
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    with pytest.raises(ValueError, match=r"revision in pins\.json"):
        kev_backend.import_kev(tmp_path, pins)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"kind": "trained-laya"}, "Unrecognized trained checkpoint manifest"),
        ({"kind": "trained-kev", "sourceRevision": "rev", "parent": {"revision": "other"}}, "parent does not match"),
    ],
)
def test_a_trained_manifest_must_name_this_source_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict[str, object], message: str
) -> None:
    checkpoint = types.ModuleType("kev.checkpoint")
    checkpoint.read_meta = lambda path: None  # type: ignore[attr-defined]
    checkpoint.resolve_run = lambda run: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kev", types.ModuleType("kev"))
    monkeypatch.setitem(sys.modules, "kev.checkpoint", checkpoint)
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(manifest))
    settings = kev_backend.KevSettings(
        pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, checkpoint_manifest=path
    )
    with pytest.raises(ValueError, match=message):
        kev_backend.resolve_checkpoint(settings, json.loads(settings.pins_path.read_text()), "rev")


@pytest.mark.unit
def test_pins_name_the_kev_source_release_and_base() -> None:
    pins = json.loads((INTEGRATION / "pins.json").read_text())
    assert pins["kevSource"]["revision"] == "45923b7a3460b6d36358e2e143455902c1eb856b"
    assert pins["kevModels"]["kev-0.8b"] == {
        "repo": "jaredpalmer/kev-0.8b",
        "revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        "base": "Qwen/Qwen3.5-0.8B-Base",
        "baseRevision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    }


# ---------------------------------------------------------------- serving and grading


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--family", "kev", "--pins", "pins.json"], "--family kev needs --kev-source"),
        (["--pins", "pins.json", "--model", "typed-decisions"], "--family laya needs"),
    ],
)
def test_the_server_names_the_settings_a_family_needs(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        server_main(argv)
    assert message in capsys.readouterr().err


@pytest.mark.unit
def test_a_kev_overrun_mid_match_scores_zero(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    message = QUESTION_OVERRUN.format(key="g1", detail="question 9000/8191 tokens")
    (run / "error.json").write_text(json.dumps({"error": message, "tick": 300}))
    suite = load_suite(SUITE_PATH)
    task = suite.tasks("val")[0]
    outcome = EpisodeOutcome(task.task_id, "spec", "sum", exit_code=1, final_output="", workspace_location=str(run))

    grade = asyncio.run(RA2Grader(suite=suite, integration_dir=INTEGRATION).grade(task, outcome))

    assert grade.score == 0.0 and "g1" in (grade.feedback or "")


# ---------------------------------------------------------------- the harness check


@pytest.mark.unit
def test_the_kev_check_accepts_the_seed_harness_and_refuses_what_exceeds_kev_s_limits() -> None:
    defaults = json.loads((INTEGRATION / "config/questions.json").read_text())
    check = KevQuestionCheck(WordTokenizer(), frozenset(defaults))
    check.check(seed_harness(INTEGRATION))
    # 60 words is over Laya's 48-token option limit but well within Kev's
    roomy = {**defaults, "intent.group.recruit": " ".join(["word"] * 60)}
    check.check(
        seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(roomy)))
    )
    too_long = {**defaults, "intent.group.recruit": " ".join(["word"] * KEV_LIMITS.option)}
    with pytest.raises(CampaignRuleError, match=r"intent.group.recruit \(option\) is 258 tokens; the limit is 256"):
        check.check(
            seed_harness(INTEGRATION).with_file(
                "ra2/questions.json", HarnessFile("instructions", json.dumps(too_long))
            )
        )


# ---------------------------------------------------------------- the trainer


def _trainer(tmp_path: Path) -> KevTrainer:
    return KevTrainer(KevTrainerSettings(pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, work_dir=tmp_path))


@pytest.mark.unit
def test_the_trainer_refuses_jobs_it_cannot_run(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="sft or grpo on typed_decisions"):
        _trainer(tmp_path).train(TrainJobSpec("sft"), None, None)  # type: ignore[arg-type]
    # A LoRA update is what a Kev checkpoint trains, so it passes the policy check and stops at the missing batch.
    lora = TrainJobSpec("sft", contract="typed_decisions", update_policy=UpdatePolicy(kind="lora", lora_rank=16))
    with pytest.raises(ContractError, match="needs a typed_decisions batch"):
        _trainer(tmp_path).train(lora, None, None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_trainer_refuses_a_missing_or_language_model_batch(tmp_path: Path) -> None:
    job = TrainJobSpec("sft", contract="typed_decisions")
    with pytest.raises(ContractError, match="needs a typed_decisions batch"):
        _trainer(tmp_path).train(job, None, None)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="needs a typed_decisions batch"):
        _trainer(tmp_path).train(job, None, token_batch())  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_trainer_declares_what_it_can_train() -> None:
    support = KevTrainer.support
    assert support.contracts == frozenset({"typed_decisions"})
    assert support.support_for("sft") == "native" and support.support_for("grpo") == "native"
