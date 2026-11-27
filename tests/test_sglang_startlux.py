"""StartLux's decision code over an SGLang forward pass (``honing/backends/sglang/startlux.py``), with fakes.

The adapter subclasses the pinned StartLux checkout's ``StartLuxDecision`` and
replaces only its forward pass. Here that checkout, torch, the tokenizer and
the engine are stand-ins, so the tests check what the adapter itself decides:
what it reads from ``decision_config.json`` instead of loading weights, which
tokenizer it refuses, and how each question's letter log-probabilities are
cut to its own options.
"""

from __future__ import annotations

import json
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from honing.backends.sglang.startlux import sglang_startlux

LETTERS = [65, 66, 67, 68]  # the tokenizer's ids for A, B, C, D


class StartLuxDecision:
    """The pinned checkout's class, as far as the adapter relies on it."""

    def __init__(self) -> None:
        raise AssertionError("StartLux's __init__ loads the checkpoint's weights; the adapter must not call it")

    def _render(self, row: dict[str, Any], order: list[str]) -> list[int]:
        # One token for the prompt's text, then one per option in the order given.
        return [len(row["text"]), *(ord(option[0]) for option in order)]

    def letter_of(self, row: dict[str, Any]) -> str:
        """Not replaced: StartLux's choice over the listed options, from the adapter's log-probabilities."""
        (values,), _ = self._logits([row])
        best = max(range(len(values.data)), key=values.data.__getitem__)
        return row["options"][best]["id"]


class Tensor:
    def __init__(self, data: Sequence[float], dtype: str) -> None:
        self.data, self.dtype = list(data), dtype


class Engine:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values
        self.calls: list[tuple[list[list[int]], list[int]]] = []

    def next_token_logprobs(self, prompts: list[list[int]], token_ids: list[int]) -> list[list[float]]:
        self.calls.append((prompts, list(token_ids)))
        return self.values[: len(prompts)]


@pytest.fixture
def startlux_env(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """torch and transformers stand-ins; returns the paths a tokenizer was loaded from."""
    loaded: list[str] = []

    def from_pretrained(path: str) -> object:
        loaded.append(path)
        return types.SimpleNamespace(letters=LETTERS)

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(tensor=Tensor, float32="float32"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)),
    )
    return loaded


UPSTREAM = types.SimpleNamespace(StartLuxDecision=StartLuxDecision)
PROMPTS = types.SimpleNamespace(check_tokenizer=lambda tok: list(tok.letters))


def checkpoint(tmp_path: Path, **config: object) -> Path:
    settings = {"letter_token_ids": LETTERS, "temperature_by_type": {"choice": "2", "rank": 0.5}, **config}
    (tmp_path / "decision_config.json").write_text(json.dumps(settings))
    return tmp_path


def row(*options: str, text: str = "state") -> dict[str, Any]:
    return {"text": text, "options": [{"id": option} for option in options]}


@pytest.mark.unit
def test_the_adapter_reads_the_decision_config_instead_of_loading_weights(
    tmp_path: Path, startlux_env: list[str]
) -> None:
    engine = Engine([])

    decision = sglang_startlux(UPSTREAM, PROMPTS, engine, checkpoint(tmp_path), 4096)  # type: ignore[arg-type]

    assert isinstance(decision, StartLuxDecision)
    assert startlux_env == [str(tmp_path)]
    assert decision.letters == LETTERS
    assert decision.temperature == {"choice": 2.0, "rank": 0.5}
    assert (decision.group, decision.keep, decision.residual) == (25, 3, 1e-3)
    assert decision.max_length == 4096
    assert decision.sglang is engine
    assert (decision.share, decision.fast_kernels, decision.graphs, decision._media) == (False, True, {}, None)


@pytest.mark.unit
def test_a_wide_choice_in_the_config_replaces_the_defaults(tmp_path: Path, startlux_env: list[str]) -> None:
    path = checkpoint(tmp_path, wide_choice={"group": 10, "keep": 2, "residual": 0.01})

    decision = sglang_startlux(UPSTREAM, PROMPTS, Engine([]), path, 100)  # type: ignore[arg-type]

    assert (decision.group, decision.keep, decision.residual) == (10, 2, 0.01)


@pytest.mark.unit
def test_a_tokenizer_whose_letters_differ_from_the_config_is_refused(tmp_path: Path, startlux_env: list[str]) -> None:
    path = checkpoint(tmp_path, letter_token_ids=[65, 66, 67, 69])

    with pytest.raises(ValueError, match=r"tokenizer letter ids differ from decision_config\.json"):
        sglang_startlux(UPSTREAM, PROMPTS, Engine([]), path, 100)  # type: ignore[arg-type]


@pytest.mark.unit
def test_each_question_gets_the_log_probabilities_of_its_own_options_from_one_engine_call(
    tmp_path: Path, startlux_env: list[str]
) -> None:
    engine = Engine([[-0.1, -2.0, -3.0, -4.0], [-5.0, -0.2, -0.3, -9.0]])
    decision = sglang_startlux(UPSTREAM, PROMPTS, engine, checkpoint(tmp_path), 100)  # type: ignore[arg-type]

    values, tokens = decision._logits([row("A", "B"), row("C", "A", "B", text="longer state")])

    assert engine.calls == [([[5, 65, 66], [12, 67, 65, 66]], LETTERS)]
    assert [(value.data, value.dtype) for value in values] == [
        ([-0.1, -2.0], "float32"),
        ([-5.0, -0.2, -0.3], "float32"),
    ]
    assert tokens == 3 + 4


@pytest.mark.unit
def test_startlux_s_own_choice_runs_on_the_engine_s_log_probabilities(tmp_path: Path, startlux_env: list[str]) -> None:
    engine = Engine([[-3.0, -0.5, -0.1, -9.0]])
    decision = sglang_startlux(UPSTREAM, PROMPTS, engine, checkpoint(tmp_path), 100)  # type: ignore[arg-type]

    # The engine favours the third letter, but this question lists two options, so the choice is between A and B.
    assert decision.letter_of(row("A", "B")) == "B"


@pytest.mark.unit
def test_the_state_decide_resets_per_request_is_kept_apart_per_thread(tmp_path: Path, startlux_env: list[str]) -> None:
    decision = sglang_startlux(UPSTREAM, PROMPTS, Engine([]), checkpoint(tmp_path), 100)  # type: ignore[arg-type]
    seen: list[object] = []

    def other_request() -> None:
        seen.append(decision._kept)
        decision._kept = "theirs"
        seen.append(decision._kept)

    assert decision._kept is None
    decision._kept = "mine"
    thread = threading.Thread(target=other_request)
    thread.start()
    thread.join()

    assert seen == [None, "theirs"]
    assert decision._kept == "mine"
