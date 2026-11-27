"""CPU stand-ins for what the decision-model trainers load: NanoJev's predictor, Kev's package, Laya's package and
its tokenizer, each around a model of a few hundred weights, and a mode that runs code written for one CUDA device
on the CPU.

The trainers in ``honing.backends.jev`` load their models through upstream code that needs a GPU, and put tensors on
``cuda`` themselves. These stand-ins keep the names, arguments and checkpoint files the trainers touch and nothing
else, so the trainers' own steps run on CPU torch: pins, checksums, examples, the update policy, the optimizer, the
loops, the saved layout and the manifest. Numerics on the real models are the GPU check's (integrations/ra2/parity).
Import this module only after ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import subprocess
import sys
import types
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.overrides import TorchFunctionMode

from honing.core import DataContract, DecisionQuestion, DecisionSample, TrainingBatch

VOCAB = 64
HIDDEN = 8
BASE, BASE_REVISION = "Qwen/Qwen3-0.6B", "qwen-revision"


def word_id(word: str) -> int:
    """A stable token id for a word, never 0 (the pad id)."""
    return 1 + sum(map(ord, word)) % (VOCAB - 1)


class WordTokenizer:
    """One token per whitespace-separated word, with the attributes and methods the trainers use."""

    pad_token_id = 0
    eos_token_id = 0
    mask_token = "[MASK]"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [word_id(word) for word in text.split()]

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text('{"model": "words"}\n')


# ---------------------------------------------------------------- the machine


def _on_cpu(value: Any) -> Any:
    if isinstance(value, torch.device) and value.type == "cuda":
        return torch.device("cpu")
    if isinstance(value, str) and value.split(":")[0] == "cuda":
        return "cpu"
    return value


class _CudaOnCpu(TorchFunctionMode):
    def __torch_function__(
        self, func: Any, types: Any, args: Sequence[Any] = (), kwargs: Mapping[str, Any] | None = None
    ) -> Any:
        return func(*map(_on_cpu, args), **{key: _on_cpu(value) for key, value in (kwargs or {}).items()})


@contextlib.contextmanager
def cuda_on_cpu() -> Iterator[None]:
    """Run code written for one CUDA device on the CPU: a tensor or module sent to ``cuda`` stays on the CPU, and
    ``torch.autocast("cuda")`` turns itself off, as it does wherever CUDA is missing."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        with _CudaOnCpu():
            yield


@contextlib.contextmanager
def any_gpu(gpu: str, lock_dir: Path | None = None) -> Iterator[str]:
    """``gpu_lock`` without nvidia-smi or a lock file."""
    yield f"GPU-fake-{gpu}"


PROCESS_VARIABLES = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "USE_TF",
    "CUBLAS_WORKSPACE_CONFIG",
)


@dataclass
class ProcessState:
    """What a trainer asked of torch's process-wide settings, recorded instead of applied."""

    threads: list[int] = field(default_factory=list)
    deterministic: list[bool] = field(default_factory=list)


def keep_process_state(monkeypatch: pytest.MonkeyPatch) -> ProcessState:
    """Undo, after the test, what a trainer changes for the whole process: the import path and the Hugging Face
    variables; record torch's thread count and determinism rather than change them for later tests."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in PROCESS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    state = ProcessState()
    monkeypatch.setattr(torch, "set_num_threads", state.threads.append)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", state.deterministic.append)
    return state


def pinned_checkout(directory: Path, files: Mapping[str, str] | None = None) -> str:
    """A git checkout holding ``files``, committed; returns its revision, which a pins file then names. Runs git."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in (files or {}).items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text)
    git = ["git", "-C", str(directory), "-c", "user.name=honing", "-c", "user.email=honing@invalid"]
    git += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "--no-verify", "--allow-empty", "-m", "pinned"], check=True)
    return subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()


def install(monkeypatch: pytest.MonkeyPatch, modules: Mapping[str, types.ModuleType]) -> None:
    """Make ``import name`` return each stand-in for the test's duration."""
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _seeded(seed: int, build: Any) -> Any:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build()


# ---------------------------------------------------------------- typed decisions

QUESTION = DecisionQuestion("choice", "Next?", (("wait", "Wait"), ("build", "Build"), ("attack", "Attack")))
CUES = {"wait": "all quiet", "build": "credits high", "attack": "enemy close"}
DECISIONS = DataContract("typed_decisions")


def decision(
    tick: int,
    chosen: str,
    group: str,
    *,
    reward: float | None = None,
    behavior: float | None = None,
    release: str | None = None,
    cue: str | None = None,
) -> DecisionSample:
    """One question about a state whose cue names the right answer (``cue``, by default the chosen option);
    ``behavior`` is the rollout's probability of the chosen option."""
    probs = None
    if behavior is not None:
        rest = (1.0 - behavior) / 2
        probs = (("next", tuple((key, behavior if key == chosen else rest) for key, _ in QUESTION.options)),)
    return DecisionSample(
        parts=(("essential", (f"tick {tick}", CUES[cue or chosen])),),
        questions=(("next", QUESTION),),
        chosen=(("next", chosen),),
        behavior_probs=probs,
        reward=reward,
        group=group,
        source_release=release,
        source_harness="seed" if release else None,
    )


def teacher_batch(train: int = 12, validation: int = 6) -> TrainingBatch:
    """Taught decisions on seed 7, and held-out ones on seed 101 (the trainers' default validation groups)."""
    keys = [key for key, _ in QUESTION.options]
    rows = [decision(i, keys[i % 3], f"map:7:{i}") for i in range(train)]
    rows += [decision(100 + i, keys[i % 3], f"map:101:{i}") for i in range(validation)]
    return TrainingBatch("teacher", DECISIONS, decisions=tuple(rows))


def rollout_batch() -> TrainingBatch:
    """Rollouts in groups of three choices on one state: the cue's option earned 1 and the others 0, except in one
    group whose choices all earned 1, which carries no signal."""
    keys = list(CUES)
    rows = [
        decision(
            tick,
            key,
            f"g{tick}",
            reward=float(key == keys[tick % 3]),
            behavior=0.4,
            release="r0001",
            cue=keys[tick % 3],
        )
        for tick in (1, 2, 3, 4)
        for key in keys
    ]
    rows += [decision(9, key, "same", reward=1.0, release="r0001", cue="wait") for key in CUES]
    return TrainingBatch("rollouts", DECISIONS, decisions=tuple(rows))


# ---------------------------------------------------------------- NanoJev


class _NanoBackbone(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, hidden)
        self.layer = torch.nn.Linear(hidden, hidden)
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.checkpointing: dict[str, Any] | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.checkpointing = dict(gradient_checkpointing_kwargs or {})

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.layer(self.embed_tokens(ids).mean(0)))


class TinyNanoJev(torch.nn.Module):
    """NanoJev's module names (``backbone``, ``norm``, ``scalar``) on a bag of words: one feature per candidate
    path, normalized by ``norm``, scored by the scalar head. Absent options score -1e4."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.backbone = _NanoBackbone(hidden)
        self.norm = torch.nn.LayerNorm(hidden, bias=False)
        self.scalar = torch.nn.Linear(hidden, 1)

    def forward(self, examples: Sequence[Mapping[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, None]:
        rows = [torch.stack([self.backbone(torch.tensor(path)) for path in e["leaf_tokens"]]) for e in examples]
        h = self.norm(torch.nn.utils.rnn.pad_sequence(rows, batch_first=True))
        logits = self.scalar(h).squeeze(-1)
        present = torch.tensor([[j < len(e["leaf_tokens"]) for j in range(h.shape[1])] for e in examples])
        return logits.masked_fill(~present, -1e4), None


def prepare_examples(request: Mapping[str, Any], tokenizer: WordTokenizer, max_length: int) -> list[dict[str, Any]]:
    """NanoJev's encoding, reduced: a path is the state, the question and one candidate, then an end token."""
    (row,) = request["states"]
    examples = []
    for qid, question in row["questions"].items():
        prefix = tokenizer.encode(f"State: {row['state']} Question: {question['instructions']}")
        options = question["criteria"].items()
        leaves = [prefix + tokenizer.encode(f"Candidate: {k}: {v} Decision:") + [0] for k, v in options]
        if max(map(len, leaves)) > max_length:
            raise ValueError("a candidate path exceeds max_length")
        example = {"qid": qid, "state_id": row["id"], "type": "choice", "candidate_ids": list(question["criteria"])}
        examples.append({**example, "leaf_tokens": leaves})
    return examples


class DecisionPredictor:
    """NanoJev's predictor: the model loaded from ``best.safetensors``, its tokenizer and length limit, and how it
    was asked to run."""

    def __init__(self, checkpoint: str, max_length: int, device_name: str, precision: str) -> None:
        config = json.loads((Path(checkpoint) / "backbone_config" / "config.json").read_text())
        self.model = TinyNanoJev(int(config["hidden_size"]))
        self.model.load_state_dict(load_file(str(Path(checkpoint) / "best.safetensors")), strict=True)
        self.tokenizer = WordTokenizer()
        self.limit = max_length
        self.device_name, self.precision = device_name, precision


NANOJEV_PREDICTOR = "from tests.fakes.decision_models import DecisionPredictor, prepare_examples  # noqa: F401\n"


def nanojev_upstream() -> types.ModuleType:
    """What ``import_nanojev`` returns: NanoJev's predictor module."""
    return _module("honing_nanojev_predict", DecisionPredictor=DecisionPredictor, prepare_examples=prepare_examples)


def nanojev_checkpoint(directory: Path, *, seed: int = 0, base_revision: str = BASE_REVISION) -> Path:
    """A NanoJev release: ``best.safetensors`` from a seeded TinyNanoJev, its run and backbone configs, a tokenizer."""
    model = _seeded(seed, TinyNanoJev)
    (directory / "backbone_config").mkdir(parents=True)
    (directory / "tokenizer").mkdir()
    (directory / "tokenizer" / "tokenizer.json").write_text("{}")
    run_config = {"model": BASE, "resolved_model_revision": base_revision, "set_head": "attention"}
    (directory / "config.json").write_text(json.dumps(run_config))
    backbone = {"hidden_size": HIDDEN, "num_attention_heads": 2, "intermediate_size": 16, "num_hidden_layers": 1}
    (directory / "backbone_config" / "config.json").write_text(json.dumps(backbone))
    save_file(model.state_dict(), str(directory / "best.safetensors"))
    return directory


def nanojev_pins(path: Path, source_revision: str) -> Path:
    release = {"repo": "C-Tianyu/NanoJev", "revision": "release-rev", "base": BASE, "baseRevision": BASE_REVISION}
    pins = {"nanojevSource": {"url": "local", "revision": source_revision}, "nanojevModels": {"nanojev": release}}
    path.write_text(json.dumps(pins))
    return path


# ---------------------------------------------------------------- Kev

KEV_BASE = "Qwen/Qwen3.5-0.8B"
SERVE_MAX_STATE, SERVE_MAX_BRANCH = 32, 48


@dataclass
class KevMeta:
    """``kev.checkpoint.Meta``: what a checkpoint says about its base and head."""

    base: str
    base_revision: str
    lora: dict[str, Any] | None = None
    head_dim: int = HIDDEN
    option_isolation: bool = True
    special_embeddings: bool = False
    weights_dtype: str = "bf16"
    lora_placement: str = "all"
    head: dict[str, Any] | None = None
    temperature: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def write_meta(directory: str, meta: KevMeta) -> None:
    torch.save(meta.head, Path(directory) / "head.pt")
    fields = {key: value for key, value in vars(meta).items() if key != "head"}
    (Path(directory) / "kev_meta.json").write_text(json.dumps(fields, sort_keys=True))


def read_meta(directory: str) -> KevMeta:
    fields = json.loads((Path(directory) / "kev_meta.json").read_text())
    return KevMeta(**fields, head=torch.load(Path(directory) / "head.pt", weights_only=True))


class _KevLm(torch.nn.Module):
    """The base language model, frozen, with a trainable low-rank adapter."""

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.base = torch.nn.Embedding(VOCAB, hidden)
        with torch.no_grad():
            self.base.weight.copy_(torch.randn(VOCAB, hidden, generator=torch.Generator().manual_seed(11)))
        self.base.weight.requires_grad_(False)
        self.lora_a = torch.nn.Linear(hidden, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, hidden, bias=False)
        self.rank = rank
        self.config = types.SimpleNamespace(use_cache=True)
        self.checkpointing: dict[str, Any] | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.checkpointing = dict(gradient_checkpointing_kwargs or {})

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.base(ids)
        return torch.tanh(x + self.lora_b(self.lora_a(x)))

    def save_pretrained(self, directory: str) -> None:
        adapter = {k: v.detach().contiguous() for k, v in self.state_dict().items() if k.startswith("lora")}
        save_file(adapter, str(Path(directory) / "adapter_model.safetensors"))
        (Path(directory) / "adapter_config.json").write_text(json.dumps({"r": self.rank}))


class _KevHead(torch.nn.Module):
    def __init__(self, hidden: int, head_dim: int) -> None:
        super().__init__()
        self.state = torch.nn.Linear(hidden, head_dim)
        self.option = torch.nn.Linear(hidden, head_dim)
        self.temperature = 1.0


class KevDecisionModel(torch.nn.Module):
    """``kev.model.DecisionModel``: ``lm`` (frozen base plus adapter) and a pointer ``head`` that scores each option
    span against the state."""

    def __init__(self, base: str, tok: Any, device: str, **options: Any) -> None:
        super().__init__()
        lora = options["lora"] or {"r": 2}
        self.lm = _KevLm(HIDDEN, int(lora["r"]))
        self.head = _KevHead(HIDDEN, int(options["head_dim"]))
        self.base_name, self.device_name, self.options = base, device, options

    def encode(
        self, tok: Any, record: Mapping[str, Any], *, max_state: int, max_branch: int, strict: bool
    ) -> dict[str, Any]:
        (question,) = record["questions"]
        state = tok(record["state"])["input_ids"]
        if strict and len(state) + 1 > max_state:
            raise ValueError("state overruns max_state")
        ids, starts = [1, *state, 2, *tok(question["instr"])["input_ids"]], []
        for option in question["options"]:
            starts.append(len(ids))
            ids += [3, *tok(option)["input_ids"]]
        return {"ids": ids, "opt_idx": starts, "label": question["label"]}

    def forward_batch(self, encoded: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
        out = []
        for item in encoded:
            h = self.lm(torch.tensor(item["ids"]))
            starts = [*item["opt_idx"], len(item["ids"])]
            options = torch.stack([h[a:b].mean(0) for a, b in itertools.pairwise(starts)])
            scores = self.head.option(options) @ self.head.state(h[: starts[0]].mean(0))
            out.append(scores.unsqueeze(0))
        return out


class _SystemOneRequest:
    def __init__(self, state: str, questions: dict[str, dict[str, Any]]) -> None:
        self.state, self.questions = state, questions


def _to_record(request: _SystemOneRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    questions, meta = [], []
    for key, q in request.questions.items():
        questions.append({"instr": q["instructions"], "options": [f"{k}: {v}" for k, v in q["criteria"].items()]})
        questions[-1]["label"] = 0
        meta.append({"id": key, "type": "choice", "keys": list(q["criteria"])})
    return {"state": request.state, "questions": questions}, meta


def kev_package(base_config_dir: Path) -> dict[str, types.ModuleType]:
    """Kev's ``api``, ``model`` and ``checkpoint`` modules; ``resolve_run`` finds every base at
    ``base_config_dir``, which holds its ``config.json``."""

    class Checkpoint:
        def __init__(self, path: str) -> None:
            self.path = Path(path)
            self.meta = read_meta(path)

        def warm_start(self, model: KevDecisionModel, meta: KevMeta) -> dict[str, Any]:
            adapter = load_file(str(self.path / "adapter_model.safetensors"))
            model.lm.load_state_dict({**model.lm.state_dict(), **adapter}, strict=True)
            model.head.load_state_dict(self.meta.head or {}, strict=True)
            model.head.temperature = self.meta.temperature
            return {"path": str(self.path), "weightsDtype": meta.weights_dtype}

    def load_tokenizer(base: str, revision: str) -> WordTokenizer:
        return WordTokenizer()

    model = _module("kev.model", DecisionModel=KevDecisionModel, load_tokenizer=load_tokenizer)
    model.__dict__.update(
        user_tokens=lambda tok, text: tok(text)["input_ids"],
        SERVE_MAX_STATE=SERVE_MAX_STATE,
        SERVE_MAX_BRANCH=SERVE_MAX_BRANCH,
    )
    checkpoint = _module(
        "kev.checkpoint", Checkpoint=Checkpoint, Meta=KevMeta, resolve_run=lambda run: str(base_config_dir)
    )
    checkpoint.__dict__.update(write_meta=write_meta, read_meta=read_meta)
    return {
        "kev": _module("kev"),
        "kev.api": _module("kev.api", SystemOneRequest=_SystemOneRequest, to_record=_to_record),
        "kev.model": model,
        "kev.checkpoint": checkpoint,
    }


def kev_base_config(directory: Path) -> Path:
    """The Qwen base's ``config.json``, as the Hugging Face cache holds it."""
    directory.mkdir(parents=True, exist_ok=True)
    text = {"hidden_size": HIDDEN, "num_attention_heads": 2, "intermediate_size": 16, "num_hidden_layers": 2}
    (directory / "config.json").write_text(json.dumps({"text_config": text}))
    return directory


def kev_checkpoint(directory: Path, *, seed: int = 0, base_revision: str = BASE_REVISION) -> Path:
    """A Kev release: the adapter, ``head.pt`` and its meta at a calibrated temperature, and a tokenizer."""
    options = {"lora": {"r": 2}, "head_dim": HIDDEN}
    model = _seeded(seed, lambda: KevDecisionModel(KEV_BASE, WordTokenizer(), "cuda", **options))
    directory.mkdir(parents=True)
    model.lm.save_pretrained(str(directory))
    head = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    meta = KevMeta(KEV_BASE, base_revision, lora={"r": 2}, head=head, temperature=1.7)
    meta.extra = {"args": {"lora_targets": "attention"}}
    write_meta(str(directory), meta)
    WordTokenizer().save_pretrained(directory)
    return directory


def kev_pins(path: Path, source_revision: str) -> Path:
    release = {"repo": "C-Tianyu/Kev", "revision": "release-rev", "base": KEV_BASE, "baseRevision": BASE_REVISION}
    pins = {"kevSource": {"url": "local", "revision": source_revision}, "kevModels": {"kev-0.8b": release}}
    path.write_text(json.dumps(pins))
    return path


# ---------------------------------------------------------------- Laya

LAYA_QTYPES = {"choice": 0, "multi": 1, "score": 2}


class _LayaConfig:
    """A Hugging Face config: its values, and ``save_pretrained``."""

    def __init__(self, **values: Any) -> None:
        self.values = values

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text(json.dumps(self.values))


class _LayaEncoder(torch.nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.embeddings = torch.nn.Embedding(VOCAB, hidden)
        self.layer = torch.nn.Linear(hidden, hidden)
        self.config = _LayaConfig(**config)
        self.checkpointing: dict[str, Any] | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.checkpointing = dict(gradient_checkpointing_kwargs or {})

    def forward(self, ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.layer(self.embeddings(ids))) * attention[..., None]


class _LayaHead(torch.nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.state = torch.nn.Linear(hidden, hidden)
        self.option = torch.nn.Linear(hidden, hidden)
        self.value = torch.nn.Linear(hidden, 1)


class TinyLaya(torch.nn.Module):
    """Laya's module names (``encoder``, ``head``) and forward: token ids, attention, option positions, option mask
    and question type in; option logits and the head's auxiliary value out."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.encoder = _LayaEncoder(config)
        self.head = _LayaHead(int(config["hidden_size"]))
        self.head_checkpointing = False

    def forward(
        self,
        ids: torch.Tensor,
        attention: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(ids, attention)
        context = h.sum(1) / attention.sum(1, keepdim=True)
        options = h.gather(1, positions[..., None].expand(-1, -1, h.shape[-1]))
        logits = (self.head.option(options) * self.head.state(context)[:, None]).sum(-1)
        return logits, self.head.value(context).squeeze(-1)


class _LayaAgent:
    @staticmethod
    def _to_internal(question: Mapping[str, Any]) -> dict[str, Any]:
        return {"t": question["type"], "ins": question["instructions"], "options": list(question["criteria"].values())}


def _render_options(question: Mapping[str, Any]) -> list[str]:
    return list(question["options"])


def _build_sequence(
    tok: Any, state: str, question: Mapping[str, Any], max_len: int, head_max_len: int, state_ids: Any = None
) -> tuple[list[int], list[int]]:
    """``[CLS] state, question, then [MASK] option`` per option; each marker is the option's first word."""
    state_ids = tok(state)["input_ids"] if state_ids is None else state_ids
    head = tok(question["t"] + " question: " + question["ins"])["input_ids"]
    options = [tok(" " + option)["input_ids"] for option in _render_options(question)]
    room = max_len - 1 - len(head) - sum(1 + len(option) for option in options)
    ids, markers = [1, *state_ids[: max(room, 0)], *head], []
    for option in options:
        markers.append(len(ids) + 1)
        ids += [2, *option]
    return ids, markers


def laya_package() -> dict[str, types.ModuleType]:
    """Laya's ``agent`` and ``common`` modules, and the ``transformers`` tokenizer loader they need."""

    def build_model(cfg: Mapping[str, Any], encoder_dir: str) -> TinyLaya:
        return TinyLaya(json.loads((Path(encoder_dir) / "config.json").read_text()))

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path: Path, *, local_files_only: bool = False) -> WordTokenizer:
            if not local_files_only or not (Path(path) / "tokenizer.json").exists():
                raise OSError("this stand-in loads a local tokenizer only")
            return WordTokenizer()

    common = _module("laya.common", QTYPES=LAYA_QTYPES, build_model=build_model, build_sequence=_build_sequence)
    common.__dict__.update(render_options=_render_options)
    return {
        "laya": _module("laya"),
        "laya.agent": _module("laya.agent", Agent=_LayaAgent),
        "laya.common": common,
        "transformers": _module("transformers", AutoTokenizer=AutoTokenizer),
    }


LAYA_ENCODER = {"hidden_size": HIDDEN, "num_attention_heads": 2, "intermediate_size": 16, "num_hidden_layers": 1}


def laya_checkpoint(directory: Path, *, seed: int = 0, dtype: torch.dtype = torch.bfloat16) -> Path:
    """A Laya release: ``model.safetensors`` from a seeded TinyLaya, the encoder config, a tokenizer, and the agent
    config with calibrated temperatures."""
    model = _seeded(seed, lambda: TinyLaya(LAYA_ENCODER))
    directory.mkdir(parents=True)
    weights = {k: v.to(dtype).contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(directory / "model.safetensors"))
    model.encoder.config.save_pretrained(directory / "encoder")
    WordTokenizer().save_pretrained(directory / "tokenizer")
    agent = {"temperature": [1.4, 1.2, 0.9], "temperature_by_options": {"3": 1.1}, "max_len": 64}
    (directory / "rl_agent_config.json").write_text(json.dumps(agent))
    return directory
