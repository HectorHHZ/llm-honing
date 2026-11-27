"""Fuzz the decision server's request parsing: any body is a request or a 400, never a dropped connection.

Hypothesis draws arbitrary bytes, arbitrary JSON (NaN and infinities
included) and near-valid requests. The examples it found are kept as plain
regression cases in ``tests/test_jev_server.py``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from honing.backends.jev.server import MAX_BRANCHES, DecisionServer, RequestError, parse_decide
from tests.test_jev_server import FakeDecisionModel, post_raw

JSON = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=8),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=8), inner, max_size=4),
    max_leaves=20,
)
OPTIONS = st.dictionaries(st.text(min_size=1, max_size=6), st.text(max_size=6), min_size=1, max_size=4)
QUESTION = st.fixed_dictionaries({"type": st.just("choice"), "instructions": st.text(max_size=8), "criteria": OPTIONS})
REQUEST = st.fixed_dictionaries(
    {"parts": st.dictionaries(st.text(max_size=6), st.lists(st.text(max_size=6), max_size=3), max_size=3)},
    optional={
        "id": JSON,
        "questions": st.dictionaries(st.text(min_size=1, max_size=6), QUESTION, max_size=3),
        "extra": JSON,
    },
)
BODY = st.one_of(
    st.binary(max_size=512),
    JSON.map(lambda value: json.dumps(value).encode()),
    REQUEST.map(lambda value: json.dumps(value).encode()),
    st.tuples(REQUEST, st.dictionaries(st.text(max_size=4), REQUEST | JSON, max_size=MAX_BRANCHES + 2)).map(
        lambda pair: json.dumps({**pair[0], "branches": pair[1]}).encode()
    ),
)


def accepted_or_refused(body: bytes) -> bool:
    """Whether ``body`` parses; anything but a ``RequestError`` fails the test."""
    try:
        first, branches = parse_decide(body)
    except RequestError:
        return False
    for request in (first, *branches.values()):
        assert isinstance(request.parts, dict) and isinstance(request.questions, dict) and request.questions
        json.dumps(request.id, allow_nan=False)  # the id goes back in the response, which must be strict JSON
    assert len(branches) <= MAX_BRANCHES
    return True


@pytest.mark.unit
@settings(max_examples=1500, deadline=None)
@given(BODY)
def test_any_body_is_a_request_or_a_request_error(body: bytes) -> None:
    accepted_or_refused(body)


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    server = DecisionServer(("127.0.0.1", 0), FakeDecisionModel())
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    yield server.url
    server.stop()
    thread.join(timeout=10)
    server.server_close()


@pytest.mark.unit
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(BODY.filter(bool))
def test_the_server_answers_every_body(server_url: str, body: bytes) -> None:
    status, reply = post_raw(server_url, body)
    if accepted_or_refused(body):
        # The fake model reads every question's criteria; one it cannot read is the model's failure, a 500.
        assert status in (200, 500)
    else:
        assert status == 400 and reply["error"]
