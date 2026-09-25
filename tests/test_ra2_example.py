"""The optional RA2 model boundary preserves options and required state on CPU."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
MODULE = Path(__file__).resolve().parents[1] / "examples/ra2/integration/laya/token_budget.py"
SPEC = importlib.util.spec_from_file_location("ra2_token_budget", MODULE)
assert SPEC is not None and SPEC.loader is not None
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)


class Tokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": text.split()}


class Agent:
    tok = Tokenizer()

    def _to_internal(self, question):
        return {"t": question["type"], "ins": question["instructions"], "options": list(question["criteria"].values())}


def options(question):
    return question["options"]


def sequence(tokenizer, state, question, max_len, head_max_len, *, state_ids):
    del tokenizer, state, head_max_len
    head = [0] * 8
    return head + state_ids[: max(0, max_len - len(head))], list(range(len(options(question))))


QUESTION = {"action": {"type": "choice", "instructions": "Choose", "criteria": {"A": "Wait", "B": "Attack"}}}


def test_ra2_optional_state_pruned_before_required_bindings():
    parts = {"essential": ["actor 12 target 34"], "details": ["detail"], "memory": ["old enemy"], "logs": ["old log"]}
    state, audit, trimmed = BUDGET.prepare_request(Agent(), parts, QUESTION, 12, 512, sequence, options)
    assert state == "actor 12 target 34"
    assert [entry["section"] for entry in trimmed] == ["logs", "memory", "details"]
    assert not audit["action"]["stateTruncated"]
    assert parts["logs"] == ["old log"]


def test_ra2_essential_overflow_refuses_inference():
    with pytest.raises(ValueError, match="Essential state"):
        BUDGET.prepare_request(Agent(), {"essential": ["actor 12 target 34"]}, QUESTION, 10, 512, sequence, options)


def test_ra2_long_option_refuses_silent_shortening():
    question = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"A": "word " * 49}}}
    with pytest.raises(ValueError, match="lossless option"):
        BUDGET.prepare_request(Agent(), {"essential": []}, question, 2048, 512, sequence, options)


def test_ra2_lost_option_markers_fail_closed():
    def missing_marker(*args, **kwargs):
        ids, markers = sequence(*args, **kwargs)
        return ids, markers[:-1]

    with pytest.raises(ValueError, match="Option marker loss"):
        BUDGET.prepare_request(Agent(), {"essential": ["normal view"]}, QUESTION, 64, 512, missing_marker, options)
