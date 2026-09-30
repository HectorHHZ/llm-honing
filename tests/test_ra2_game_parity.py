"""The real-game parity launcher must pump the model queue, not just accept HTTP."""

from __future__ import annotations

import contextlib
import json
import urllib.request

import pytest
from integrations.ra2.parity.game_parity import serving_cpu_model

from tests.test_jev_server import QUESTION, FakeDecisionModel


@pytest.mark.integration
@pytest.mark.parametrize("fail", [False, True])
def test_parity_server_answers_requests_and_closes(fail: bool) -> None:
    model = FakeDecisionModel()
    expected = pytest.raises(RuntimeError, match="match failed") if fail else contextlib.nullcontext()
    with expected, serving_cpu_model(model) as server:
        body = {"id": "parity", "parts": {"essential": ["tick 30"]}, "questions": QUESTION}
        request = urllib.request.Request(
            server.url + "/v1/decide", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        # Calling only serve_forever leaves the model queue unconsumed and times out here.
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
        assert result["answers"]["root"]["choice"] == "power"
        assert result["id"] == "parity"
        if fail:
            raise RuntimeError("match failed")
    assert server.socket.fileno() == -1
