"""Decisions from any language model: lettered options, read from the next-token log-probabilities of the letters.

A model family with its own decision code (StartLux) keeps it. This adapter is
for every other checkpoint SGLang can load, a base or instruct model or one a
campaign has just trained, so that it can answer ``/v1/decide`` with no
family code at all.

Each question is rendered as one prompt: a fixed system line, then the state,
the question, and its options, one per line as ``A) <id>: <description>``.
With a chat template the prompt is the template's user turn and the opening
of the assistant turn, with thinking off where the template supports it;
without one it is plain text ending in ``Answer:``. The answer is read at the
next position: the log-probabilities of the listed letters, divided by the
temperature of the question type, normalised over those letters. Nothing is
generated.

- ``choice`` lists its criteria in the order given. Over 26 options, the
  options split into near-equal groups of at most 25, the 3 likeliest of each
  go to a final round, and the rest share a residual probability, as
  StartLux-Decision does.
- ``noul`` lists ``yes`` and ``no``, with the descriptions it gives; the
  answer is the probability of yes.
- ``score`` lists its levels lowest first; the answer is the expected level
  and the probability of each.

The state is packed and trimmed by whole entries until every prompt fits
``max_length``, exactly as for StartLux (:func:`honing.backends.jev.startlux.pack_request`).

The forward pass is any *scorer* with ``next_token_logprobs(prompts,
token_ids)``: an :class:`~honing.backends.sglang.SGLangEngine`, or for a parity
check the checkpoint's own transformers class.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from honing.backends.jev.startlux import checked_answer, pack_request, validate_questions
from honing.backends.sglang.checkpoint import checkpoint_checksum, checkpoint_identity, context_length

SYSTEM = "Read the state, then answer the question with the letter of exactly one listed option."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = len(LETTERS)
ENCODING = "lettered-prompt"


class Scorer(Protocol):
    def next_token_logprobs(self, prompts: Sequence[Sequence[int]], token_ids: Sequence[int]) -> list[list[float]]: ...


@dataclass(frozen=True)
class LetteredSettings:
    """How a checkpoint answers lettered questions.

    ``temperature_by_type`` divides the letters' log-probabilities before the
    softmax (1.0 reads the model's own distribution). ``wide_group``,
    ``wide_keep`` and ``wide_residual`` set the rounds of a choice over 26
    options.
    """

    max_length: int = 8192
    temperature_by_type: Mapping[str, float] = field(
        default_factory=lambda: {"choice": 1.0, "noul": 1.0, "score": 1.0}
    )
    wide_group: int = 25
    wide_keep: int = 3
    wide_residual: float = 1e-3

    def __post_init__(self) -> None:
        if not 0 < self.max_length <= 262144:
            raise ValueError("max_length must be between 1 and 262144")
        if set(self.temperature_by_type) != {"choice", "noul", "score"} or not all(
            t > 0 for t in self.temperature_by_type.values()
        ):
            raise ValueError("temperature_by_type needs a positive temperature for choice, noul and score")
        if not 2 <= self.wide_group < MAX_OPTIONS or not 1 <= self.wide_keep < self.wide_group:
            raise ValueError("wide_group must be 2 to 25 and wide_keep below it")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).strip()


def options_of(question: Mapping[str, Any]) -> list[tuple[str, str]]:
    """A question's options in prompt order, as (answer key, line text without its letter)."""
    kind, criteria = question["type"], question.get("criteria") or {}
    if kind == "noul":
        criteria = criteria if isinstance(criteria, Mapping) else {}
        return [
            (key, f"{name}: {_text(criteria.get(key))}" if _text(criteria.get(key)) else name)
            for key, name in (("true", "yes"), ("false", "no"))
        ]
    if kind == "score":
        if isinstance(criteria, Mapping):
            try:
                keys = sorted(criteria, key=float)
            except (TypeError, ValueError):
                keys = list(criteria)
            levels = [criteria[k] for k in keys]
        else:
            levels = list(criteria)
        return [(str(i), _text(level) or str(i)) for i, level in enumerate(levels)]
    return [(key, f"{key}: {_text(text)}" if _text(text) else key) for key, text in criteria.items()]


class LetteredPrompts:
    """Renders lettered prompts with a checkpoint's tokenizer and finds the letters' token ids after them."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.chat = bool(getattr(tokenizer, "chat_template", None))
        self.system = True
        if self.chat:
            try:
                self._text("(none)", "Which?", ["yes", "no"])
            except Exception:  # a template that raises on the system role; a second failure is a real one
                self.system = False
        probe = self.render("(none)", "Which?", ["yes", "no"])
        text = self._text("(none)", "Which?", ["yes", "no"])
        self.letters = self._letter_ids(probe, text)

    def _text(self, state: str, instructions: str, lines: Sequence[str]) -> str:
        body = "\n".join(f"{LETTERS[i]}) {line}" for i, line in enumerate(lines))
        user = f"State:\n{state or '(none)'}\n\nQuestion: {instructions.strip()}\nOptions:\n{body}"
        if not self.chat:
            return f"{SYSTEM}\n\n{user}\nAnswer:"
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if not self.system:  # the template refuses a system turn (Gemma): the line opens the user turn instead
            messages = [{"role": "user", "content": f"{SYSTEM}\n\n{user}"}]
        text: str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return text

    def render(self, state: str, instructions: str, lines: Sequence[str]) -> list[int]:
        ids: list[int] = self.tokenizer.encode(self._text(state, instructions, lines), add_special_tokens=False)
        return ids

    def _letter_ids(self, probe: list[int], text: str) -> list[int]:
        """The token of each letter as the next token: the bare letter, or with a leading space; one token each."""
        for prefix in ("", " "):
            ids = []
            for letter in LETTERS:
                token = self.tokenizer.encode(prefix + letter, add_special_tokens=False)
                if len(token) != 1 or self.tokenizer.encode(text + prefix + letter, add_special_tokens=False) != [
                    *probe,
                    token[0],
                ]:
                    break
                ids.append(token[0])
            else:
                if len(set(ids)) == MAX_OPTIONS:
                    return ids
        raise ValueError("the tokenizer does not read each option letter as one token after the prompt")


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    scaled = [v / temperature for v in values]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = math.fsum(weights)
    return [w / total for w in weights]


def _groups(keys: Sequence[str], size: int) -> list[list[str]]:
    count = -(-len(keys) // size)
    base, extra = divmod(len(keys), count)
    groups, start = [], 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        groups.append(list(keys[start:end]))
        start = end
    return groups


class LetteredDecisionModel:
    """A checkpoint answering decision requests through a scorer. ``concurrent`` follows the scorer."""

    def __init__(
        self,
        scorer: Scorer,
        tokenizer: Any,
        settings: LetteredSettings,
        metadata: Mapping[str, object],
        concurrent: bool = True,
    ) -> None:
        self._scorer = scorer
        self._prompts = LetteredPrompts(tokenizer)
        self._settings = settings
        self._metadata = {**metadata, "encoding": ENCODING, "temperatureByType": dict(settings.temperature_by_type)}
        self.concurrent = concurrent

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @property
    def letters(self) -> list[int]:
        return self._prompts.letters

    def _rows(self, state: str, question: Mapping[str, Any]) -> list[list[int]]:
        """The prompts a question can render: the first round and, for a wide choice, its longest final round."""
        options = options_of(question)
        lines = [line for _, line in options]
        if len(options) <= MAX_OPTIONS:
            return [self._prompts.render(state, question["instructions"], lines)]
        rows = [
            self._prompts.render(state, question["instructions"], group)
            for group in _groups(lines, self._settings.wide_group)
        ]
        finalists = sum(min(self._settings.wide_keep, len(g)) for g in _groups(lines, self._settings.wide_group))
        if finalists > MAX_OPTIONS:
            raise ValueError("a choice this wide needs more than two rounds, which the lettered adapter refuses")
        longest = sorted(lines, key=lambda line: len(self._prompts.tokenizer.encode(line)), reverse=True)
        rows.append(self._prompts.render(state, question["instructions"], longest[:finalists]))
        return rows

    def _measure(self, state: str, questions: Mapping[str, Any]) -> dict[str, int]:
        return {key: max(len(row) for row in self._rows(state, question)) for key, question in questions.items()}

    def _probabilities(self, prompts: list[list[int]], counts: list[int], kinds: list[str]) -> list[list[float]]:
        values = self._scorer.next_token_logprobs(prompts, self._prompts.letters) if prompts else []
        temperatures = self._settings.temperature_by_type
        return [_softmax(v[:n], temperatures[k]) for v, n, k in zip(values, counts, kinds, strict=True)]

    def _wide(self, state: str, question: Mapping[str, Any]) -> tuple[dict[str, float], int]:
        """A choice over 26 options in two rounds; returns its probabilities and the prompt tokens spent."""
        options = options_of(question)
        line_of = dict(options)
        groups = _groups([key for key, _ in options], self._settings.wide_group)
        prompts = [self._prompts.render(state, question["instructions"], [line_of[k] for k in g]) for g in groups]
        first = self._probabilities(prompts, [len(g) for g in groups], ["choice"] * len(groups))
        first_p = {k: p for g, ps in zip(groups, first, strict=True) for k, p in zip(g, ps, strict=True)}
        finalists = [
            k
            for g, ps in zip(groups, first, strict=True)
            for k, _ in sorted(zip(g, ps, strict=True), key=lambda x: -x[1])[: self._settings.wide_keep]
        ]
        final_prompt = self._prompts.render(state, question["instructions"], [line_of[k] for k in finalists])
        final = self._probabilities([final_prompt], [len(finalists)], ["choice"])[0]
        residual = self._settings.wide_residual
        rest = [k for k, _ in options if k not in set(finalists)]
        mass = math.fsum(first_p[k] for k in rest) or 1.0
        probabilities = {k: p * (1 - residual) for k, p in zip(finalists, final, strict=True)}
        probabilities.update({k: residual * first_p[k] / mass for k in rest})
        total = math.fsum(probabilities.values())
        return {k: probabilities[k] / total for k, _ in options}, sum(map(len, prompts)) + len(final_prompt)

    @staticmethod
    def _answer(question: Mapping[str, Any], keys: list[str], p: list[float] | dict[str, float]) -> dict[str, Any]:
        dist = dict(zip(keys, p, strict=True)) if isinstance(p, list) else p
        if question["type"] == "noul":
            return {"type": "noul", "noul": dist["true"]}
        if question["type"] == "score":
            return {
                "type": "score",
                "score": math.fsum(int(k) * v for k, v in dist.items()),
                "probabilities": dict(dist),
            }
        best = max(dist, key=dist.__getitem__)
        return {"type": "choice", "choice": best, "probabilities": dict(dist)}

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        started = time.perf_counter()
        validate_questions(questions)
        tokenizer = self._prompts.tokenizer

        def count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        state, audit, trimmed = pack_request(
            parts, questions, lambda text: self._measure(text, questions), count, self._settings.max_length
        )
        for entry in audit.values():
            entry["encoding"] = ENCODING
        raw: dict[str, Any] = {}
        tokens = 0
        batch: list[tuple[str, list[str], list[int]]] = []
        for key, question in questions.items():
            assert isinstance(question, Mapping)
            options = options_of(question)
            keys = [k for k, _ in options]
            if len(options) > MAX_OPTIONS:
                wide, spent = self._wide(state, question)
                raw[key] = self._answer(question, keys, wide)
                tokens += spent
            else:
                prompt = self._prompts.render(state, question["instructions"], [line for _, line in options])
                batch.append((key, keys, prompt))
        probabilities = self._probabilities(
            [prompt for _, _, prompt in batch],
            [len(keys) for _, keys, _ in batch],
            [str(questions[key]["type"]) for key, _, _ in batch],  # type: ignore[index]
        )
        for (key, keys, prompt), p in zip(batch, probabilities, strict=True):
            raw[key] = self._answer(questions[key], keys, p)  # type: ignore[arg-type]
            tokens += len(prompt)
        answers = {key: checked_answer(key, question, raw[key]) for key, question in questions.items()}  # type: ignore[arg-type]
        return {
            "answers": answers,
            "modelResult": {"answers": raw, "usage": {"input_tokens": tokens, "output_tokens": 0}},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


def load_tokenizer(checkpoint: Path) -> Any:
    """The checkpoint's tokenizer (transformers is needed in this process for it, torch is not)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint))


def load(checkpoint: Path, engine_settings: Any, settings: LetteredSettings, checksum: str | None = None) -> Any:
    """A context manager: the checkpoint on an SGLang engine, answering lettered decisions.

    Honing computes the weights' checksum itself, and refuses the checkpoint
    when ``checksum`` names another one. A checkpoint that reads fewer tokens
    than ``settings.max_length`` (OLMoE reads 4,096) gets its own limit, so the
    state is trimmed until every prompt fits the model, and the engine is
    started with room for that limit.
    """
    import contextlib
    import dataclasses

    from honing.backends.sglang.engine import SGLangEngine

    @contextlib.contextmanager
    def loading() -> Any:
        actual = checkpoint_checksum(checkpoint)
        if checksum is not None and actual != checksum:
            raise ValueError(f"{checkpoint} has weights {actual[:12]}, not the {checksum[:12]} asked for")
        tokenizer = load_tokenizer(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())
        limit = context_length(checkpoint)
        fitted = settings
        if limit is not None and settings.max_length > limit - 8:
            fitted = dataclasses.replace(settings, max_length=limit - 8)
        engine = dataclasses.replace(engine_settings, context_length=fitted.max_length + 8)
        model, revision = checkpoint_identity(checkpoint, actual)
        with SGLangEngine.launch(checkpoint, engine) as served:
            # model, family, revision and maxLen are what the game bridges read from the health route.
            metadata = {
                "family": "lettered",
                "model": model,
                "revision": revision,
                "path": str(checkpoint),
                "architecture": (config.get("architectures") or ["?"])[0],
                "weightsSha256": actual,
                "engine": "sglang",
                "engineVersions": served.versions,
                "prefixCache": engine.prefix_cache,
                "maxLen": fitted.max_length,
            }
            yield LetteredDecisionModel(served, tokenizer, fitted, metadata)

    return loading()
