"""DiffusionGemma through djev, in the Serve plane: the environment check, request packing and both trimming paths,
the answer checks, the client's error mapping, serving, and the harness check. djev and vLLM need their own GPU
environment, so the deployment is stood in for; the GPU check is integrations/ra2/parity/djev_serving.py."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from honing.backends.djev import decision as djev_decision
from honing.backends.djev import runtime as djev_runtime_module
from honing.backends.djev.decision import (
    DjevClient,
    DjevDecisionModel,
    DjevRefusedError,
    DjevSettings,
    checked_answer,
    validate_questions,
)
from honing.backends.djev.runtime import DjevRuntimeSettings, verify_environment
from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.jev.server import main as server_main
from honing.backends.ra2 import DjevQuestionCheck, seed_harness
from honing.backends.ra2.grader import OVER_BUDGET
from honing.core import CampaignRuleError, HarnessFile
from tests.test_ra2_backend import REPO

INTEGRATION = REPO / "integrations/ra2"
PINS = json.loads((INTEGRATION / "pins.json").read_text())
LIMITS = {"state_characters": 60, "instructions_characters": 40, "criterion_characters": 20}
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
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


class FakeDeployment:
    """The djev API's /v1/request: refuses states over ``context`` characters as djev refuses an over-long prompt."""

    def __init__(self, context: int = 10_000, refusal: str | None = None) -> None:
        self.context = context
        self.refusal = refusal
        self.states: list[str] = []

    def request(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        self.states.append(state)
        if self.refusal:
            raise DjevRefusedError(self.refusal)
        if len(state) > self.context:
            raise DjevRefusedError(
                "the prompt and answer canvas exceed the model context limit; shorten the state or questions"
            )
        answers = {
            key: {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}, "confidence": 0.2}
            for key in questions
        }
        return {
            "model": "djev-0.1",
            "answers": answers,
            "usage": {"input_tokens": 300},
            "diagnostics": {"canvas_tokens": 16},
        }


def model(deployment: FakeDeployment) -> DjevDecisionModel:
    return DjevDecisionModel(deployment, WordTokenizer(), LIMITS, {"maxModelLen": 32768})  # type: ignore[arg-type]


# ---------------------------------------------------------------- packing, trimming, answers


@pytest.mark.unit
def test_the_answer_and_the_audit_place_the_state_in_the_context() -> None:
    deployment = FakeDeployment()
    response = model(deployment).predict(PARTS, QUESTIONS)

    assert response["answers"] == {"g1": {"choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}}
    assert set(response) == {"answers", "stateText", "tokenAudit", "trimmed", "modelResult", "latencyMs"}
    audit = response["tokenAudit"]["g1"]  # type: ignore[index]
    tokens = len(response["stateText"].split())  # type: ignore[union-attr]
    assert audit["stateTokens"] == tokens and audit["inputTokens"] == 300 and audit["canvasTokens"] == 16
    assert audit["stateRoom"] == 32768 - 16 - (300 - tokens)
    assert audit["stateCharacters"] == len(response["stateText"]) and audit["stateTruncated"] is False  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_state_is_trimmed_by_whole_entries_to_djev_s_character_limit() -> None:
    deployment = FakeDeployment()
    response = model(deployment).predict(PARTS, QUESTIONS)

    # 60 characters: the logs go first (oldest first), then the newest memory entry
    assert [t["text"] for t in response["trimmed"]] == ["tick 3 moved", "tick 6 moved", "built barracks"]  # type: ignore[index]
    assert response["stateText"] == "tick 90 paused\ncredits 1200\nenemy north east\nbuilt refinery"
    assert deployment.states == [response["stateText"]]  # nothing over the limit was sent


@pytest.mark.unit
def test_a_context_refusal_trims_one_entry_at_a_time_until_djev_accepts() -> None:
    deployment = FakeDeployment(context=45)
    response = model(deployment).predict(PARTS, QUESTIONS)

    assert response["stateText"] == "tick 90 paused\ncredits 1200\nenemy north east"
    assert len(deployment.states) == 2  # 59 characters, refused as over the context; 44 accepted
    assert [t["section"] for t in response["trimmed"]] == ["logs", "logs", "memory", "memory"]  # type: ignore[index]


@pytest.mark.unit
def test_an_essential_state_that_never_fits_is_the_overrun_the_grader_scores_as_zero() -> None:
    with pytest.raises(ValueError) as refused:
        model(FakeDeployment(context=5)).predict(PARTS, QUESTIONS)
    assert str(refused.value) == STATE_OVERRUN and any(m in STATE_OVERRUN for m in OVER_BUDGET)


@pytest.mark.unit
def test_any_other_refusal_is_an_error_not_a_reason_to_trim() -> None:
    deployment = FakeDeployment(refusal="Invalid decision request")
    with pytest.raises(ValueError, match="djev refused the request: Invalid decision request"):
        model(deployment).predict(PARTS, QUESTIONS)
    assert len(deployment.states) == 1


@pytest.mark.unit
def test_a_question_over_djev_s_limits_is_the_harness_s_overrun() -> None:
    long = {"g1": {**QUESTIONS["g1"], "criteria": {"A": "wait", "B": "x" * 21}}}
    with pytest.raises(ValueError) as refused:
        validate_questions(long, LIMITS)
    assert "Question g1 exceeds lossless option/instruction budget" in str(refused.value)
    assert any(marker in str(refused.value) for marker in OVER_BUDGET)
    with pytest.raises(ValueError, match="Only choice questions"):
        validate_questions({"g1": {**QUESTIONS["g1"], "type": "score"}}, LIMITS)


@pytest.mark.unit
def test_an_answer_must_cover_the_options_sum_to_one_and_choose_the_most_probable() -> None:
    criteria = {"A": "wait", "B": "build"}
    good = {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}
    assert checked_answer(criteria, good) == {"choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}
    with pytest.raises(ValueError, match="cover exactly"):
        checked_answer(criteria, {**good, "probabilities": {"A": 1.0}})
    with pytest.raises(ValueError, match="do not sum to one"):
        checked_answer(criteria, {**good, "probabilities": {"A": 0.5, "B": 0.4}})
    with pytest.raises(ValueError, match="not the most probable"):
        checked_answer(criteria, {**good, "choice": "A"})


# ---------------------------------------------------------------- the client


class _FakeApi(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"limits": LIMITS} if self.path == "/config" else {"ready": True})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["state"] == "too long":
            self._send(422, {"error": {"message": "the prompt and answer canvas exceed the model context limit"}})
        elif body["state"] == "down":
            self._send(503, {"error": {"message": "Inference capacity is full; retry later"}})
        else:
            assert body["options"] == {"seed": 0, "samples": 1, "isolation": "joint", "diagnostics": True}
            self._send(200, {"answers": {}, "usage": {"input_tokens": 1}})


@pytest.mark.unit
def test_the_client_maps_djev_s_refusals_and_failures() -> None:
    server = HTTPServer(("127.0.0.1", 0), _FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = DjevClient(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        assert client.ready() and client.config()["limits"] == LIMITS
        assert client.request("fine", QUESTIONS)["usage"] == {"input_tokens": 1}
        with pytest.raises(DjevRefusedError, match="exceed the model context limit"):
            client.request("too long", QUESTIONS)
        with pytest.raises(RuntimeError, match="returned 503: Inference capacity is full"):
            client.request("down", QUESTIONS)
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- the environment check


def _environment(**changes: Any) -> dict[str, Any]:
    packages = {
        "torch": "2.13.0+cu130",
        "vllm": "0.29.1rc1.dev347+gdee37d891",
        "transformers": "5.17.0",
        "Pillow": "12.3.0",
    }
    found = {
        "model": "google/diffusiongemma-26B-A4B-it",
        "modelRevision": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        "pinnedPackages": packages,
        "installedPackages": dict(packages),
        "baseCommit": "dee37d89115db4c94a820a79a78a7828e141c910",
        "structuredReadCommit": "0f4678d44159b42531a9e398ae05df869ec67c5d",
        "installed": {
            "base_commit": "dee37d89115db4c94a820a79a78a7828e141c910",
            "structured_read_commit": "0f4678d44159b42531a9e398ae05df869ec67c5d",
            "packages": packages,
        },
    }
    return {**found, **changes}


def _verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, found: dict[str, Any], revision: str | None = None
) -> Any:
    pinned = PINS["djevSource"]["revision"]

    def check_output(command: list[str], **kwargs: Any) -> str:
        if command[0] == "git":
            return "" if "status" in command else (revision or pinned) + "\n"
        return json.dumps(found) + "\n"

    monkeypatch.setattr(djev_runtime_module.subprocess, "check_output", check_output)
    settings = DjevRuntimeSettings(
        pins_path=INTEGRATION / "pins.json", source_dir=tmp_path, python=tmp_path / "python"
    )
    return verify_environment(settings, PINS)


@pytest.mark.unit
def test_the_environment_check_passes_djev_s_pins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checked = _verify(monkeypatch, tmp_path, _environment())
    assert checked["sourceCommit"] == PINS["djevSource"]["revision"] and "installed" not in checked


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"installedPackages": {"torch": "2.14.0", "vllm": "x", "transformers": "5.17.0", "Pillow": "12.3.0"}},
            "differs from its pins",
        ),
        ({"installed": None}, "lacks djev's installed sources"),
        ({"modelRevision": "0" * 40}, "which pins.json does not name"),
    ],
)
def test_the_environment_check_refuses_what_djev_s_pins_do_not_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _verify(monkeypatch, tmp_path, _environment(**changes))


@pytest.mark.unit
def test_the_djev_checkout_must_be_the_pinned_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned, unmodified checkout"):
        _verify(monkeypatch, tmp_path, _environment(), revision="0" * 40)


@pytest.mark.unit
def test_runtime_and_decision_settings_refuse_impossible_combinations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1,024 to 32,768"):
        DjevRuntimeSettings(pins_path=tmp_path, source_dir=tmp_path, python=tmp_path, max_model_len=65536)
    with pytest.raises(ValueError, match="different ports"):
        DjevRuntimeSettings(pins_path=tmp_path, source_dir=tmp_path, python=tmp_path, api_port=9, model_port=9)
    with pytest.raises(ValueError, match="either a djev runtime to start or the url"):
        DjevSettings(pins_path=tmp_path)
    assert djev_decision.OPTIONS["isolation"] == "joint"


# ---------------------------------------------------------------- pins, serving, the harness check


@pytest.mark.unit
def test_pins_name_the_djev_source_and_diffusiongemma() -> None:
    assert PINS["djevSource"] == {
        "url": "https://github.com/Davipar/djev-dev.git",
        "revision": "3ce907e6835212f27ee82b4cee9039198c4abe35",
    }
    assert PINS["djevModels"] == {
        "diffusiongemma-26b-a4b-it": {
            "repo": "google/diffusiongemma-26B-A4B-it",
            "revision": "f7f5b7f5fa82ffc52addd066915886d497f5517b",
        }
    }


@pytest.mark.unit
def test_the_server_needs_a_djev_deployment_to_use_or_start(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        server_main(["--family", "djev", "--pins", "pins.json"])
    assert "--family djev needs --djev-url (a running deployment) or --djev-source" in capsys.readouterr().err


@pytest.mark.unit
def test_the_djev_check_counts_characters_as_djev_s_contract_does() -> None:
    check = DjevQuestionCheck.for_integration(INTEGRATION)
    check.check(seed_harness(INTEGRATION))
    defaults = json.loads((INTEGRATION / "config/questions.json").read_text())
    long = {**defaults, "intent.group.recruit": "x" * 501}
    spec = seed_harness(INTEGRATION).with_file("ra2/questions.json", HarnessFile("instructions", json.dumps(long)))
    with pytest.raises(
        CampaignRuleError, match=r"intent\.group\.recruit \(option\) is 501 characters; the limit is 500"
    ):
        check.check(spec)
