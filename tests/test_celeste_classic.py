"""Portable model/native-boundary checks; fixtures are not gameplay evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).parents[1] / "integrations/celeste-classic"
SPEC = importlib.util.spec_from_file_location("classic_run", HERE / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)
pytestmark = pytest.mark.unit


def response_for(request: dict) -> dict:
    return {
        "id": request["id"],
        "trimmed": [],
        "stateText": "\n".join(request["parts"]["essential"]),
        "tokenAudit": {
            "action": {
                "options": len(run.ACTIONS),
                "stateTruncated": False,
                "stateTokens": 100,
                "stateRoom": 1000,
                "inputTokens": 200,
            }
        },
        "answers": {
            "action": {"choice": "right", "probabilities": {name: 1 if name == "right" else 0 for name in run.ACTIONS}}
        },
    }


class PausedFixture:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.actions = []
        self.state = {"id": 1, "ticks": 0}

    def step(self, before: dict, action: str, frames: int) -> dict:
        self.actions.append((action, frames))
        self.state = {"id": 2, "ticks": frames}
        return self.state


class FixtureClient:
    def __init__(self, response: dict) -> None:
        self.response = response

    def decide(self, request: dict, *, raw_path: Path) -> dict:
        raw_path.write_text(json.dumps(self.response))
        return self.response


@pytest.mark.parametrize(
    "kind",
    [
        "zero",
        "two",
        "boolean",
        "negative",
        "nonfinite",
        "bad_sum",
        "missing_option",
        "extra_option",
        "wrong_id",
        "bool_id",
        "wrong_argmax",
        "unoffered",
        "missing_state",
        "changed_state",
        "missing_trim_audit",
        "required_removed",
        "clipped",
        "missing_audit",
        "missing_counts",
        "wrong_options",
        "bool_tokens",
        "oversize_state",
        "oversize_input",
        "extra_question",
        "missing_question",
    ],
)
def test_bad_model_never_reaches_action_boundary_and_keeps_raw(tmp_path: Path, kind: str) -> None:
    native = PausedFixture(tmp_path)
    request = run.request_for(native.state, 12)
    response = response_for(request)
    probabilities = response["answers"]["action"]["probabilities"]
    if kind in {"zero", "two", "boolean", "negative", "nonfinite", "bad_sum"}:
        value = {"zero": 0, "two": 2, "boolean": True, "negative": -0.1, "nonfinite": float("inf"), "bad_sum": 0.1}[
            kind
        ]
        response["answers"]["action"]["probabilities"] = dict.fromkeys(run.ACTIONS, value)
    elif kind == "missing_option":
        probabilities.pop("wait")
    elif kind == "extra_option":
        probabilities["cheat"] = 0
    elif kind == "wrong_id":
        response["id"] = 2
    elif kind == "bool_id":
        response["id"] = True
    elif kind == "wrong_argmax":
        response["answers"]["action"]["choice"] = "left"
    elif kind == "unoffered":
        response["answers"]["action"]["choice"] = "teleport"
    elif kind == "missing_state":
        response.pop("stateText")
    elif kind == "changed_state":
        response["stateText"] = "other"
    elif kind == "missing_trim_audit":
        response.pop("trimmed")
    elif kind == "required_removed":
        response["trimmed"] = [{"section": "essential", "text": request["parts"]["essential"][0]}]
    elif kind == "clipped":
        response["tokenAudit"]["action"]["stateTruncated"] = True
    elif kind == "missing_audit":
        response.pop("tokenAudit")
    elif kind == "missing_counts":
        response["tokenAudit"]["action"].pop("stateTokens")
    elif kind == "wrong_options":
        response["tokenAudit"]["action"]["options"] = 2
    elif kind == "bool_tokens":
        response["tokenAudit"]["action"]["inputTokens"] = True
    elif kind == "oversize_state":
        response["tokenAudit"]["action"]["stateTokens"] = 1001
    elif kind == "oversize_input":
        response["tokenAudit"]["action"]["maxInputTokens"] = 199
    elif kind == "extra_question":
        response["answers"]["extra"] = {}
    elif kind == "missing_question":
        response["answers"] = {}
    with pytest.raises(ValueError):
        run.decide_and_step(native, FixtureClient(response), native.state, 12)
    assert native.actions == [] and native.state == {"id": 1, "ticks": 0}
    assert json.loads((tmp_path / "response-000001.raw").read_text()) == response
    record = json.loads((tmp_path / "decisions.jsonl").read_text())
    assert record["request"] == request and record["error"] and "after" not in record


def test_valid_rounded_distribution_and_tie() -> None:
    request = run.request_for({"id": 1}, 12)
    response = response_for(request)
    response["answers"]["action"] = {"choice": "right", "probabilities": dict.fromkeys(run.ACTIONS, 0.0714)}
    assert run.checked_action(response, request) == ("right", 0.0714)


def test_permitted_optional_log_removal() -> None:
    request = run.request_for({"id": 1}, 12)
    request["parts"]["logs"] = ["old optional log", "new optional log"]
    response = response_for(request)
    response["trimmed"] = [{"section": "logs", "text": "old optional log"}]
    response["stateText"] += "\nnew optional log"
    assert run.checked_action(response, request)[0] == "right"
    response["trimmed"][0]["text"] = "unsubmitted log"
    with pytest.raises(ValueError):
        run.checked_action(response, request)


def test_valid_response_reaches_native_exactly_once(tmp_path: Path) -> None:
    native = PausedFixture(tmp_path)
    request = run.request_for(native.state, 12)
    result = run.decide_and_step(native, FixtureClient(response_for(request)), native.state, 12)
    assert native.actions == [("right", 12)] and result["ticks"] == 12


@pytest.mark.parametrize(
    "counter", ["nativeUpdates", "nativeDraws", "clockFrame", "ticks", "episodeTicks", "id", "lastMask", "lastFrames"]
)
def test_native_echo_and_real_engine_progress_must_match(counter: str) -> None:
    before = {"id": 1, "ticks": 0, "episodeTicks": 0, "nativeUpdates": 60, "nativeDraws": 60, "clockFrame": 0}
    after = {
        "id": 2,
        "ticks": 12,
        "episodeTicks": 12,
        "nativeUpdates": 72,
        "nativeDraws": 72,
        "clockFrame": 12,
        "lastMask": 2,
        "lastFrames": 12,
    }
    run.check_transition(before, after, 2, 12)
    bad = copy.deepcopy(after)
    bad[counter] += 1
    with pytest.raises(ValueError):
        run.check_transition(before, bad, 2, 12)


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"id":1,"id":2}', b'{"x":NaN}', b'{"x":1e9999}'])
def test_raw_bytes_saved_before_json_failure(tmp_path: Path, raw: bytes) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return raw

    client = run.DecisionClient("http://localhost:8809")
    with patch.object(run.urllib.request, "urlopen", return_value=Response()), pytest.raises(ValueError):
        client.decide({"id": 1}, raw_path=tmp_path / "response.raw")
    assert (tmp_path / "response.raw").read_bytes() == raw


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location("classic_" + name, HERE / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reason", ["wrong_revision", "output_exists", "dirty_resources", "modified_source"])
def test_builder_refuses_unpinned_or_modified_inputs_before_compiler(tmp_path: Path, reason: str) -> None:
    build = load_tool("build")
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    if reason == "output_exists":
        output.mkdir()
    answers = [b"not-the-pin\n" if reason == "wrong_revision" else (build.REVISION + "\n").encode()]
    if reason not in {"wrong_revision", "output_exists"}:
        answers += [b"data/gfx.bmp" if reason == "dirty_resources" else b""]
    if reason == "modified_source":
        (source / "celeste.c").write_bytes(b"modified")
        answers += [b"original"]
    with (
        patch.object(build.subprocess, "check_output", side_effect=answers),
        patch.object(build.subprocess, "run") as compiler,
        pytest.raises(ValueError),
    ):
        build.build(source, output)
    compiler.assert_not_called()
    if reason != "output_exists":
        assert not output.exists()


def test_missing_capture_frame_cannot_make_a_video(tmp_path: Path) -> None:
    encode = load_tool("encode")
    (tmp_path / "result.json").write_text(json.dumps({"frames": 2, "capture": True}))
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/000001.bmp").write_bytes(b"fixture")
    with patch.object(encode.subprocess, "run") as ffmpeg, pytest.raises(ValueError):
        encode.encode(tmp_path, tmp_path / "movie.mp4")
    ffmpeg.assert_not_called()


def test_existing_output_and_receipts_are_unchanged_on_start_failure(tmp_path: Path) -> None:
    output = tmp_path / "previous-run"
    output.mkdir()
    (output / "cleanup.json").write_bytes(b'{"previous": "owned receipt"}\n')
    (output / "observation.json").write_bytes(b'{"id": 8}\n')
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    native = run.Native(tmp_path / "unused-game", output)
    with patch.object(run.subprocess, "Popen") as process:
        try:
            with pytest.raises(ValueError, match="fresh directory"):
                native.start()
        finally:
            native.close()
    process.assert_not_called()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original
    assert native.owns_output is False
