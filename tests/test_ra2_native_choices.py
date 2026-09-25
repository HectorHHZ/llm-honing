"""CPU protocol fixtures only; no native weights, GPU or game inference."""

import copy

import pytest
from examples.ra2.integration.models.choice_protocol import checked_answer, omni_messages, pack_request
from examples.ra2.integration.models.upstream_adapters import JevOmniChoices, NanoJevChoices

QUESTIONS = {
    "next": {
        "type": "choice",
        "instructions": "Select the visible target.",
        "criteria": {"Z": "Wait", "B": "Attack tank #81"},
    }
}
PARTS = {"essential": ["Normal player state"], "details": ["Unit details"], "memory": [], "logs": []}


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, enable_thinking=False):
        assert add_generation_prompt and tokenize and not enable_thinking
        return {"input_ids": [self.encode(messages[0]["content"])]}


def prepare_nano(payload, tokenizer, limit):
    row = payload["states"][0]
    return [
        {
            "qid": key,
            "candidate_ids": list(q["criteria"]),
            "leaf_tokens": [
                tokenizer.encode(row["state"] + q["instructions"] + text) for text in q["criteria"].values()
            ],
        }
        for key, q in row["questions"].items()
    ]


class NanoFixture:
    limit = 80
    tokenizer = Tokenizer()

    def predict(self, payload, *, batch_questions, temperature):
        self.payload = copy.deepcopy(payload)
        assert batch_questions == 1 and temperature == 1
        examples = prepare_nano(payload, self.tokenizer, self.limit)
        assert all(len(path) <= self.limit for e in examples for path in e["leaf_tokens"])
        return {
            "states": [
                {
                    "id": "decision",
                    "answers": {
                        "next": {
                            "type": "choice",
                            "choice": "B",
                            "probabilities": {"Z": 0.1, "B": 0.9},
                        }
                    },
                }
            ]
        }


def test_nano_keeps_native_choice_ids_and_exact_retained_state():
    predictor = NanoFixture()
    parts = {**PARTS, "logs": ["Old event " * 20, "Recent event " * 20]}
    original = copy.deepcopy(parts)
    result = NanoJevChoices(predictor, prepare_nano).predict(parts, QUESTIONS)
    assert result["answers"]["next"]["choice"] == "B"
    assert list(result["answers"]["next"]["probabilities"]) == ["Z", "B"]
    assert predictor.payload["states"][0]["state"] == result["stateText"]
    assert predictor.payload["states"][0]["questions"] == QUESTIONS
    assert [r["section"] for r in result["trimmed"]] == ["logs", "logs"]
    assert parts == original


def test_bad_nano_batch_is_not_replaced_by_a_local_choice():
    class MissingAnswer(NanoFixture):
        def predict(self, *args, **kwargs):
            return {"states": [{"id": "decision", "answers": {}}]}

    with pytest.raises(ValueError, match="partial question batch"):
        NanoJevChoices(MissingAnswer(), prepare_nano).predict(PARTS, QUESTIONS)


def test_omni_maps_numbered_probabilities_back_to_offered_key_order():
    seen = []

    def native(model, head, tokenizer, state, question, options):
        assert model is None and head is None
        assert options == ["Wait", "Attack tank #81"]
        messages = omni_messages(state, {"instructions": question, "criteria": dict(enumerate(options))})
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        seen.append(messages)
        return [0.8, 0.2]

    result = JevOmniChoices(None, None, Tokenizer(), native, max_length=512).predict(PARTS, QUESTIONS)
    assert result["answers"]["next"]["choice"] == "Z"  # First native index, not alphabetically first key.
    assert "1. Wait\n2. Attack tank #81" in seen[0][0]["content"]
    assert result["tokenAudit"]["next"]["stateTruncated"] is False


def test_omni_rejects_prompt_drift_before_native_forward():
    forwarded = []

    def drift(model, head, tokenizer, state, question, options):
        messages = omni_messages(state + " unexpected data", QUESTIONS["next"])
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        forwarded.append(True)
        return [1.0, 0.0]

    with pytest.raises(RuntimeError, match="differs from verified"):
        JevOmniChoices(None, None, Tokenizer(), drift, max_length=512).predict(PARTS, QUESTIONS)
    assert not forwarded


def test_oversized_essential_state_never_calls_model_or_truncates_choices():
    invoked = []

    def native(*args):
        invoked.append(True)
        return [1.0, 0.0]

    with pytest.raises(ValueError, match="Essential context"):
        JevOmniChoices(None, None, Tokenizer(), native, max_length=30).predict(PARTS, QUESTIONS)
    assert not invoked


def test_joint_budget_keeps_every_question_and_trims_whole_records_in_order():
    questions = {**QUESTIONS, "other": {**QUESTIONS["next"], "instructions": "Another command"}}
    parts = {"essential": ["ESS"], "details": ["DET"], "memory": ["MEM"], "logs": ["LOG1", "LOG2"]}
    original = copy.deepcopy(parts)

    def measure(state, qs):
        return {
            key: {"stateTokens": len(state), "inputTokens": len(state) + (5 if key == "next" else 8)} for key in qs
        }

    state, audit, trimmed = pack_request(parts, questions, measure, 11)
    assert state == "ESS"
    assert [(r["section"], r["text"]) for r in trimmed] == [
        ("logs", "LOG1"),
        ("logs", "LOG2"),
        ("memory", "MEM"),
        ("details", "DET"),
    ]
    assert set(audit) == set(questions)
    assert parts == original


@pytest.mark.parametrize(
    "probabilities,choice",
    [
        ({"Z": 1.0}, "Z"),
        ({"Z": float("nan"), "B": 0.0}, "Z"),
        ({"Z": 0.2, "B": 0.2}, "Z"),
        ({"Z": 0.2, "B": 0.8}, "Z"),
        ({"Z": True, "B": False}, "Z"),
    ],
)
def test_invalid_or_misbound_distributions_fail_closed(probabilities, choice):
    with pytest.raises(ValueError):
        checked_answer(QUESTIONS["next"]["criteria"], probabilities, choice)
