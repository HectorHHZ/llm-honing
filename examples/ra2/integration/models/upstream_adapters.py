"""Experimental native APIs; not yet connected to the RA2 match runner or gates.

Callers must reserve their GPU before constructing an upstream predictor. These
adapters do not fabricate decisions or substitute another model on failure.
"""

import copy
import time

from .choice_protocol import (
    canonical_ids,
    checked_answer,
    nano_payload,
    omni_messages,
    pack_request,
    token_digest,
)


class NanoJevChoices:
    """Wrap the pinned NanoJev DecisionPredictor and its prepare_examples function.

    The caller loads verified local weights with the official upstream API.
    CPU contract tests inject a fixture predictor; they do not validate weights.
    """

    def __init__(self, predictor, prepare_examples):
        self.predictor = predictor
        self.prepare_examples = prepare_examples

    def measure(self, state, questions):
        # This is a tokenizer-only measurement, not a larger inference limit.
        examples = self.prepare_examples(nano_payload(state, questions), self.predictor.tokenizer, 2**30)
        if len(examples) != len(questions) or {e["qid"] for e in examples} != set(questions):
            raise ValueError("NanoJev encoding changed the question set")
        tokens = len(self.predictor.tokenizer.encode(state, add_special_tokens=False))
        measured = {}
        for example in examples:
            if example["candidate_ids"] != list(questions[example["qid"]]["criteria"]):
                raise ValueError("NanoJev encoding reordered offered choices")
            if len(example["leaf_tokens"]) != len(example["candidate_ids"]):
                raise ValueError("NanoJev encoding omitted candidate paths")
            for path in example["leaf_tokens"]:
                canonical_ids(path)
            measured[example["qid"]] = {
                "stateTokens": tokens,
                "inputTokens": max(map(len, example["leaf_tokens"])),
                "encoding": "nanojev-native-candidate-paths",
                "candidatePathSha256": [token_digest(path) for path in example["leaf_tokens"]],
            }
        return measured

    def predict(self, parts, questions):
        started = time.perf_counter()
        state, audit, trimmed = pack_request(parts, questions, self.measure, self.predictor.limit)
        response = self.predictor.predict(nano_payload(state, questions), batch_questions=1, temperature=1.0)
        states = response.get("states")
        if not isinstance(states, list) or len(states) != 1 or states[0].get("id") != "decision":
            raise ValueError("NanoJev returned a different state batch")
        raw = states[0].get("answers", {})
        if set(raw) != set(questions):
            raise ValueError("NanoJev returned a partial question batch")
        if any(answer.get("type") != "choice" or "choice" not in answer for answer in raw.values()):
            raise ValueError("NanoJev returned an incomplete choice answer")
        answers = {
            key: checked_answer(question["criteria"], raw[key].get("probabilities"), raw[key].get("choice"))
            for key, question in questions.items()
        }
        return {
            "answers": answers,
            "stateText": state,
            "trimmed": trimmed,
            "tokenAudit": audit,
            "modelResult": response,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


class ExactPromptTokenizer:
    """Check the actual upstream prompt before any model forward can run."""

    def __init__(self, tokenizer, expected_digest, max_length):
        self.tokenizer = tokenizer
        self.expected_digest = expected_digest
        self.max_length = max_length
        self.checked_calls = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        value = self.tokenizer.apply_chat_template(*args, **kwargs)
        ids = canonical_ids(value)
        if len(ids) > self.max_length or token_digest(ids) != self.expected_digest:
            # Upstream retries TypeError/ValueError while formatting. This must
            # stop execution, not select a different template after preflight.
            raise RuntimeError("Actual Jev-Omni prompt differs from verified lossless encoding")
        self.checked_calls += 1
        return value


class JevOmniChoices:
    """Use the pinned merged-text loader's predict function and loaded objects."""

    def __init__(self, model, head, tokenizer, upstream_predict, max_length=2048):
        self.model, self.head, self.tokenizer = model, head, tokenizer
        self.upstream_predict = upstream_predict
        self.max_length = max_length

    def measure(self, state, questions):
        state_tokens = len(self.tokenizer.encode(state, add_special_tokens=False))
        measured = {}
        for key, question in questions.items():
            value = self.tokenizer.apply_chat_template(
                omni_messages(state, question), add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
            ids = canonical_ids(value)
            measured[key] = {
                "stateTokens": state_tokens,
                "inputTokens": len(ids),
                "promptSha256": token_digest(ids),
                "encoding": "jev-omni-native-numbered-options",
            }
        return measured

    def predict(self, parts, questions):
        started = time.perf_counter()
        state, audit, trimmed = pack_request(parts, questions, self.measure, self.max_length)
        answers = {}
        for key, question in questions.items():
            guard = ExactPromptTokenizer(self.tokenizer, audit[key]["promptSha256"], self.max_length)
            probabilities = self.upstream_predict(
                self.model, self.head, guard, state, question["instructions"], list(question["criteria"].values())
            )
            if guard.checked_calls != 1 or len(probabilities) != len(question["criteria"]):
                raise ValueError("Jev-Omni did not consume the exact verified prompt and choice set")
            answers[key] = checked_answer(
                question["criteria"], dict(zip(question["criteria"], probabilities, strict=True))
            )
        return {
            "answers": answers,
            "stateText": state,
            "trimmed": trimmed,
            "tokenAudit": audit,
            "modelResult": {"family": "jev-omni", "answers": copy.deepcopy(answers)},
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
