"""djev wire-contract checks using an HTTP stand-in; no GPU or model weights."""

from __future__ import annotations

import contextlib
import copy
import json
import select
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from honing.backends.jev.djev import DjevDecisionModel, DjevSettings, checked_answers, pack_state, validate_questions

QUESTIONS = {"action": {"type": "choice", "instructions": "Next?", "criteria": {"A": "Wait", "B": "Move"}}}
PARTS = {"essential": ["坦克 at 30,40"], "details": ["炮弹 incoming"], "memory": [], "logs": ["tick 30"]}
ANSWER = {"type": "choice", "choice": "B", "probabilities": {"A": 0.2, "B": 0.8}, "confidence": 0.8}


@pytest.fixture
def upstream():
    class Handler(BaseHTTPRequestHandler):
        requests: ClassVar[list] = []
        status = 200
        reply: ClassVar[dict] = {
            "model": "dgemma",
            "answers": {"action": ANSWER},
            "usage": {"input_tokens": 127, "output_tokens": 10},
            "diagnostics": {"steps": 1, "samples": {"n": 2}, "timing": {"reads": 2}},
        }

        def log_message(self, *_):
            pass

        def send(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.path == "/v1/models"
            self.send({"data": [{"id": "dgemma"}]})

        def do_POST(self):
            assert self.path == "/v1/systemone"
            self.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send(self.reply, self.status)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/"])
def test_complete_state_and_native_distribution_are_preserved(upstream, suffix):
    url, handler = upstream
    original = copy.deepcopy(handler.reply)
    with DjevDecisionModel.load(DjevSettings(url + suffix, "dgemma", samples=2)) as model:
        result = model.predict(PARTS, QUESTIONS)
    assert handler.requests == [
        {
            "model": "dgemma",
            "state": "坦克 at 30,40\n炮弹 incoming\ntick 30",
            "questions": QUESTIONS,
            "samples": 2,
            "steps": 1,
        }
    ]
    assert result["answers"] == original["answers"]
    assert result["modelResult"] == original
    assert result["trimmed"] == []
    audit = result["tokenAudit"]["action"]
    assert audit["inputTokens"] == 127 and audit["options"] == 2
    assert audit["stateTokens"] is None and audit["stateRoom"] is None
    assert audit["accounting"] == "upstream-request-total"
    assert model.metadata["remoteWeightsVerified"] is False


def test_startup_refuses_an_unserved_model(upstream):
    with (
        pytest.raises(ValueError, match="does not advertise"),
        DjevDecisionModel.load(DjevSettings(upstream[0], "wrong")),
    ):
        pytest.fail("must fail before ready")


@pytest.mark.parametrize(
    "change",
    [
        {"model": "wrong"},
        {"answers": {}},
        {"usage": {}},
        {"usage": {"input_tokens": True}},
        {"answers": {"action": None}},
        {"answers": {"action": {**ANSWER, "choice": "A"}}},
        {"answers": {"action": {**ANSWER, "probabilities": {"A": 0.2, "C": 0.8}}}},
        {"answers": {"action": {**ANSWER, "probabilities": {"A": 0.2, "B": 0.7}}}},
        {"answers": {"action": {**ANSWER, "probabilities": {"A": float("nan"), "B": 0.8}}}},
        {"answers": {"action": {**ANSWER, "confidence": False}}},
        {"answers": {"action": {**ANSWER, "confidence": 0.4}}},
    ],
)
def test_invalid_upstream_answers_fail_without_a_fallback(upstream, change):
    url, handler = upstream
    handler.reply = {**handler.reply, **change}
    with pytest.raises(ValueError):
        DjevDecisionModel(DjevSettings(url, "dgemma")).predict(PARTS, QUESTIONS)
    assert len(handler.requests) == 1


def test_upstream_context_error_is_not_hidden_or_retried(upstream):
    url, handler = upstream
    handler.status, handler.reply = 422, {"error": {"message": "maximum context exceeded"}}
    with pytest.raises(RuntimeError, match=r"422.*maximum context exceeded"):
        DjevDecisionModel(DjevSettings(url, "dgemma")).predict(PARTS, QUESTIONS)
    assert len(handler.requests) == 1


def test_noul_and_score_keep_the_upstream_mean_and_legend():
    qs = {
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "high"]},
    }
    raw = {
        "urgent": {"type": "noul", "noul": 0.42},
        "risk": {
            "type": "score",
            "score": 0.75,
            "legend": {"0": "low", "1": "high"},
            "probabilities": {"0": 0.25, "1": 0.75},
            "confidence": 0.75,
        },
    }
    validate_questions(qs)
    assert checked_answers(qs, raw) == raw
    for field, bad in [("legend", {"0": "high", "1": "low"}), ("score", 1), ("score", False)]:
        with pytest.raises(ValueError):
            checked_answers(qs, {**raw, "risk": {**raw["risk"], field: bad}})


@pytest.mark.parametrize(
    "question",
    [
        {"type": "span", "instructions": "Who?"},
        {**QUESTIONS["action"], "depends_on": ["other"]},
        {**QUESTIONS["action"], "criteria": {str(i): "x" for i in range(27)}},
        {**QUESTIONS["action"], "criteria": None},
        {"type": "score", "instructions": "Risk?", "criteria": ["same", "same"]},
    ],
)
def test_unsupported_schemas_fail_before_the_upstream_call(upstream, question):
    url, handler = upstream
    with pytest.raises(ValueError):
        DjevDecisionModel(DjevSettings(url, "dgemma")).predict(PARTS, {"q": question})
    assert not handler.requests


@pytest.mark.parametrize("parts", [{"essential": "abc"}, {"essential": ["x"], "hidden": []}, {"essential": []}])
def test_invalid_context_is_not_silently_dropped(parts):
    with pytest.raises(ValueError):
        pack_state(parts)


@pytest.mark.parametrize(
    "setting",
    [
        {"samples": 0},
        {"samples": True},
        {"steps": 9},
        {"timeout": float("nan")},
        {"url": "file:///etc"},
        {"url": "http://u:p@host"},
        {"model": ""},
    ],
)
def test_invalid_connection_and_sampling_settings_are_refused(setting):
    with pytest.raises(ValueError):
        DjevSettings(**{"url": "http://localhost:8011", "model": "dgemma", **setting})


@pytest.mark.integration
def test_cli_bridge_and_real_ra2_node_client(upstream, tmp_path: Path, repo_root: Path):
    url, handler = upstream
    # Exercise the public launcher, HTTP queue, family check and match client together.
    with (tmp_path / "server.err").open("w+") as errors:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "honing.backends.jev.server",
                "--family",
                "djev",
                "--djev-url",
                url,
                "--model",
                "dgemma",
                "--port",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            cwd=repo_root,
        )
        try:
            assert select.select([process.stdout], [], [], 10)[0], "bridge startup timed out"
            ready = json.loads(process.stdout.readline())
            assert ready["ready"] and ready["metadata"]["family"] == "djev"
            script = """
import assert from 'node:assert/strict';
import {createDecisionClient,decisionClientOptions,checkServedFamily}
    from './integrations/ra2/integration/models/factory.mjs';
const options=decisionClientOptions({family:'djev','decision-server':process.argv[1]},'intent');
const client=createDecisionClient(options);
checkServedFamily(options,await client.ready);
const reply=await client.predict(JSON.parse(process.argv[2]));
assert.equal(reply.answers.action.choice,'B');
assert.deepEqual(reply.answers.action.probabilities,{A:.2,B:.8});
assert.equal(reply.tokenAudit.action.stateTokens,null);
client.close();
console.log('djev bridge / RA2 client: ok');
"""
            result = subprocess.run(
                [
                    "node",
                    "--input-type=module",
                    "-e",
                    script,
                    ready["url"],
                    json.dumps({"parts": PARTS, "questions": QUESTIONS}),
                ],
                cwd=repo_root,
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            )
            assert "djev bridge / RA2 client: ok" in result.stdout
            assert len(handler.requests) == 1
        finally:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
