"""Serving any checkpoint through SGLang (RFC #103): the lettered decision readout, the chat model, the policy
endpoints a campaign's runners resolve, and checkpoint checksums. The engine is stood in for by scorers and servers
with SGLang's shapes; the GPU checks are scripts/e2e/sglang_logprobs.py, scripts/e2e/sglang_lettered.py and the
``*_sglang.json`` runner check configs.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.sglang import SGLangChatModel, checkpoint_checksum, hf_snapshot
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints
from honing.backends.sglang.lettered import LetteredDecisionModel, LetteredPrompts, LetteredSettings, options_of
from honing.core import ChatMessage, ContractError, PolicyRef

# ---------------------------------------------------------------- the lettered readout


class CharTokenizer:
    """One token per character, so each letter is one token wherever it falls. ``template`` gives it a chat
    template; ``space`` makes letters tokens only after a space, as BPE tokenizers often do after "Answer:"."""

    def __init__(self, template: bool = False, space: bool = False) -> None:
        self.chat_template = "{chat}" if template else None
        self.space = space

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.space:  # " A" is one token (1000 + ord), a bare capital letter at the end is split in two
            out, i = [], 0
            while i < len(text):
                if text[i] == " " and i + 1 < len(text) and text[i + 1].isupper():
                    out.append(1000 + ord(text[i + 1]))
                    i += 2
                elif text[i].isupper() and i == len(text) - 1 and len(text) > 1:
                    out.extend([ord(text[i]), 0])
                    i += 1
                else:
                    out.append(ord(text[i]))
                    i += 1
            return out
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i - 1000) if i >= 1000 else chr(i) for i in ids if i)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"


class OptionScorer:
    """Scores the letter of each option by how many times the option's line mentions ``favourite``; records calls."""

    def __init__(self, tokenizer: CharTokenizer, favourite: str = "power") -> None:
        self.tokenizer = tokenizer
        self.favourite = favourite
        self.calls: list[int] = []

    def next_token_logprobs(self, prompts: Sequence[Sequence[int]], token_ids: Sequence[int]) -> list[list[float]]:
        self.calls.append(len(prompts))
        out = []
        for prompt in prompts:
            text = self.tokenizer.decode(prompt)
            lines = [line for line in text.split("\n") if len(line) > 2 and line[1] == ")"]
            scores = [float(line.count(self.favourite)) for line in lines]
            out.append([scores[i] if i < len(scores) else -30.0 for i in range(len(token_ids))])
        return out


PARTS = {"essential": ["tick 90", "credits 1200"], "details": ["enemy north"], "memory": [], "logs": ["t1", "t2"]}
CHOICE = {
    "build": {
        "type": "choice",
        "instructions": "What next?",
        "criteria": {"p": "build power", "r": "build refinery", "b": "power power barracks"},
    }
}


def model(
    scorer: Any = None, tokenizer: CharTokenizer | None = None, **settings: Any
) -> tuple[LetteredDecisionModel, OptionScorer]:
    tokenizer = tokenizer or CharTokenizer()
    scorer = scorer or OptionScorer(tokenizer)
    return LetteredDecisionModel(scorer, tokenizer, LetteredSettings(**settings), {"family": "lettered"}), scorer


def test_options_are_listed_in_order_with_yes_no_and_levels_lowest_first() -> None:
    assert options_of(CHOICE["build"]) == [
        ("p", "p: build power"),
        ("r", "r: build refinery"),
        ("b", "b: power power barracks"),
    ]
    assert options_of({"type": "noul", "instructions": "x", "criteria": {"true": "go"}}) == [
        ("true", "yes: go"),
        ("false", "no"),
    ]
    levels = {"type": "score", "instructions": "x", "criteria": {"10": "high", "2": "low", "5": "mid"}}
    assert options_of(levels) == [("0", "low"), ("1", "mid"), ("2", "high")]


@pytest.mark.parametrize("template", [False, True])
def test_a_choice_is_the_softmax_of_its_letters_at_the_type_s_temperature(template: bool) -> None:
    lettered, scorer = model(
        tokenizer=CharTokenizer(template=template), temperature_by_type={"choice": 2.0, "noul": 1.0, "score": 1.0}
    )
    answer = lettered.predict(PARTS, CHOICE)
    p = answer["answers"]["build"]["probabilities"]
    expected = [math.exp(s / 2.0) for s in (1.0, 0.0, 2.0)]
    assert p == pytest.approx({k: v / sum(expected) for k, v in zip("prb", expected, strict=True)})
    assert answer["answers"]["build"]["choice"] == "b"
    assert answer["tokenAudit"]["build"]["encoding"] == "lettered-prompt"
    assert scorer.calls == [1]


def test_every_question_of_a_request_goes_to_the_scorer_in_one_call() -> None:
    lettered, scorer = model()
    questions = {
        **CHOICE,
        "attack": {"type": "noul", "instructions": "Attack?", "criteria": {"true": "power up", "false": "wait"}},
        "threat": {"type": "score", "instructions": "Threat?", "criteria": ["none", "power", "power power"]},
    }
    answers = lettered.predict(PARTS, questions)["answers"]
    assert scorer.calls == [3]
    assert answers["attack"]["noul"] == pytest.approx(math.e / (math.e + 1))
    weights = [1.0, math.e, math.e**2]
    assert answers["threat"]["score"] == pytest.approx(sum(i * w for i, w in enumerate(weights)) / sum(weights))


def test_a_choice_with_one_option_is_refused_before_the_model() -> None:
    lettered, scorer = model()
    with pytest.raises(ValueError, match="at least two choices"):
        lettered.predict(PARTS, {"only": {"type": "choice", "instructions": "x", "criteria": {"go": "go"}}})
    assert scorer.calls == []


def test_a_choice_over_26_options_runs_a_final_round_on_each_group_s_three_likeliest() -> None:
    lettered, scorer = model()
    criteria = {f"o{i}": "power " * (i % 7) + f"cell {i}" for i in range(30)}
    answer = lettered.predict(PARTS, {"wide": {"type": "choice", "instructions": "Where?", "criteria": criteria}})
    p = answer["answers"]["wide"]["probabilities"]
    assert set(p) == set(criteria) and math.fsum(p.values()) == pytest.approx(1.0)
    assert scorer.calls == [2, 1]  # two groups of 15, then one final round of 6
    finalists = sorted(p, key=p.__getitem__, reverse=True)[:6]
    assert math.fsum(p[k] for k in finalists) == pytest.approx(1 - 1e-3)
    assert answer["answers"]["wide"]["choice"] in {"o6", "o13", "o20", "o27"}


def test_the_state_is_trimmed_by_whole_entries_until_every_prompt_fits() -> None:
    fitted, _ = model(max_length=10_000)
    long_parts = {**PARTS, "logs": [f"log entry {i}" for i in range(60)]}
    full = fitted.predict(long_parts, CHOICE)
    limit = max(a["inputTokens"] for a in full["tokenAudit"].values()) - 100
    small, _ = model(max_length=limit)
    answer = small.predict(long_parts, CHOICE)
    assert answer["trimmed"] and all(t["section"] == "logs" for t in answer["trimmed"])
    assert answer["trimmed"][0]["text"] == "log entry 0"  # oldest first
    assert max(a["inputTokens"] for a in answer["tokenAudit"].values()) <= limit
    tiny, _ = model(max_length=50)
    with pytest.raises(ValueError, match=STATE_OVERRUN.split(";")[0]):
        tiny.predict(long_parts, CHOICE)


def test_letters_are_found_after_a_space_when_the_tokenizer_joins_them_to_it() -> None:
    plain = LetteredPrompts(CharTokenizer())
    assert plain.letters[:2] == [ord("A"), ord("B")]
    spaced = LetteredPrompts(CharTokenizer(space=True))
    assert spaced.letters[:2] == [1000 + ord("A"), 1000 + ord("B")]


def test_a_tokenizer_that_splits_the_letters_is_refused() -> None:
    class Splitting(CharTokenizer):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [t for c in text for t in (ord(c), 1)]

    with pytest.raises(ValueError, match="one token"):
        LetteredPrompts(Splitting())


# ---------------------------------------------------------------- the chat model


class _Chat(BaseHTTPRequestHandler):
    server: Any

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        self.server.seen = (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        body = json.dumps(
            {"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def chat_server() -> Iterator[Any]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Chat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_the_chat_model_posts_openai_messages_and_reports_tokens(chat_server: Any) -> None:
    chat = SGLangChatModel(f"http://127.0.0.1:{chat_server.server_address[1]}/v1/", "abc")
    reply = chat.complete([ChatMessage("system", "s"), ChatMessage("user", "u")], max_tokens=5, temperature=0.0)
    assert (reply.text, reply.input_tokens, reply.output_tokens, reply.cost_usd) == ("hello", 7, 2, None)
    assert chat_server.seen == (
        "/v1/chat/completions",
        {
            "model": "abc",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
    )


# ---------------------------------------------------------------- checkpoints and policy endpoints


def write_checkpoint(directory: Path, fill: bytes = b"\0") -> Path:
    """A tiny Hugging Face checkpoint: a config and one safetensors file with a 2 x 3 tensor."""
    directory.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
    }
    (directory / "config.json").write_text(json.dumps(config))
    header = json.dumps({"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + fill * 24)
    return directory


def test_a_checkpoint_is_named_by_its_shards_names_and_hashes(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "m")
    shard = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    assert checkpoint_checksum(checkpoint) == hashlib.sha256(f"model.safetensors:{shard}\n".encode()).hexdigest()
    snapshot = hf_snapshot(checkpoint)
    assert snapshot.parameters == 6 and len(snapshot.architecture.layers) == 2
    assert snapshot.checkpoint.checksum == checkpoint_checksum(checkpoint)


def policy(checkpoint: Path) -> PolicyRef:
    snapshot = hf_snapshot(checkpoint)
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


class FakeStarts:
    """Records what the endpoints start and stop, in place of SGLang."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def __call__(self, checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> str:
        self.started.append((checkpoint.name, gpu))
        stack.callback(self.stopped.append, checkpoint.name)
        return f"http://{checkpoint.name}.{gpu}/v1"


@pytest.mark.parametrize("runtime", ["sglang", "jev_decision"])
def test_endpoints_start_a_checkpoint_once_and_stop_the_least_recently_used_when_gpus_run_out(
    tmp_path: Path, runtime: str
) -> None:
    a, b, c = (policy(write_checkpoint(tmp_path / name, bytes([i]))) for i, name in enumerate("abc"))
    starts = FakeStarts()
    settings = ServingSettings(python=Path("python"), gpus=("0", "1"), runtime=runtime)  # type: ignore[arg-type]
    with SGLangPolicyEndpoints(settings, start=starts) as endpoints:
        first = endpoints.resolve(a)
        assert endpoints.resolve(a) == first and starts.started == [("a", "0")]
        assert (first.base_url, first.runtime, first.served_model) == (
            "http://a.0/v1",
            runtime,
            a.checkpoint.checksum[:16],
        )
        assert first.tokenizer_location == a.checkpoint.location
        endpoints.resolve(b)
        endpoints.resolve(a)  # a is now the most recently used
        endpoints.resolve(c)  # no GPU free: b goes
        assert starts.stopped == ["b"] and starts.started[-1] == ("c", "1")
        assert endpoints.serving == {a.checkpoint.checksum: "0", c.checkpoint.checksum: "1"}
    assert sorted(starts.stopped) == ["a", "b", "c"]


def test_endpoints_refuse_weights_other_than_the_policy_names_and_non_hf_checkpoints(tmp_path: Path) -> None:
    good = policy(write_checkpoint(tmp_path / "m"))
    starts = FakeStarts()
    endpoints = SGLangPolicyEndpoints(ServingSettings(python=Path("python"), gpus=("0",)), start=starts)
    (tmp_path / "m" / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ContractError, match="does not hold"):
        endpoints.resolve(good)
    (tmp_path / "m" / "config.json").unlink()
    with pytest.raises(ContractError, match="not a Hugging Face checkpoint"):
        endpoints.resolve(good)
    assert starts.started == []


def test_a_server_that_fails_to_start_leaves_its_gpu_free(tmp_path: Path) -> None:
    def failing(checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> str:
        raise RuntimeError("no engine")

    p = policy(write_checkpoint(tmp_path / "m"))
    endpoints = SGLangPolicyEndpoints(ServingSettings(python=Path("python"), gpus=("0",)), start=failing)
    with pytest.raises(RuntimeError, match="no engine"):
        endpoints.resolve(p)
    assert endpoints.serving == {}


def test_a_checkpoint_is_named_by_its_hub_repository_or_its_directory(tmp_path: Path) -> None:
    from honing.backends.sglang.checkpoint import checkpoint_identity

    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / "70d244cc"
    snapshot.mkdir(parents=True)
    assert checkpoint_identity(snapshot, "f" * 64) == ("Qwen/Qwen3-1.7B", "70d244cc")
    assert checkpoint_identity(tmp_path / "iter4", "abc" * 22) == ("iter4", ("abc" * 22)[:12])
