"""Load a pinned StartLux-Decision checkpoint and answer typed questions with it.

StartLux-Decision (``StartLuxLabs/Startlux-Decision``, weights under
``startlux-models/``) is a family of dense decision models on a Qwen3.5
backbone. Each question is rendered as its own chat prompt with lettered
options, and the answer is read from the next-token logits of the option
letters after one forward pass, so nothing is generated. Honing pins three
sizes: ``startlux-0.8b``, ``startlux-4b`` and ``startlux-27b``.

Inference goes through StartLux's own :class:`startlux_decision.StartLuxDecision`
at the commit in ``pins.json`` (``startluxSource``), with its prompt rendering,
per-type temperatures, CUDA graphs and wide-choice rounds unchanged. Honing adds
what it adds for the other local families: the source must be the pinned,
unmodified checkout, every weight shard must match the object id the model
repository publishes for it, the GPU is locked for this process, and the state
is packed from its four sections and trimmed by whole entries (never truncated)
until every prompt fits ``max_length``, with every trim reported. An overrun of
the essential state is refused with the wording ``RA2Grader`` scores as 0.

One StartLux path is turned off. When two or more questions share a prompt
beginning of 4,096 tokens or more, StartLux runs that beginning once and the
questions on top of it, but at the pinned commit that path reads the option
letters under bf16 autocast: its probabilities are rounded to bf16, sum to
anywhere from 0.998 to 1.002, and differ from the same question asked alone by
up to 0.02. With it off, such requests run as one padded eager batch, whose
readout is fp32 and matches the question asked alone; on the 27B model three
questions of 6,500 to 9,500 tokens take 2 to 3 seconds instead of 0.8.

Requests run one at a time and the model keeps nothing between them, so an
answer does not depend on the requests before it.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.jev.laya import gpu_lock
from honing.backends.jev.token_budget import SECTIONS
from honing.core import ArchitectureSpec, CheckpointRef, LayerShape, ModelSnapshot

FAMILY = "startlux"
# What a checkpoint download needs: weights, configs and tokenizer. The repository's copy of the inference code is
# not used; the code comes from the pinned source checkout.
DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
QUESTION_TYPES = ("choice", "noul", "score")


@dataclass(frozen=True)
class StartLuxSettings:
    """Everything needed to load one StartLux-Decision checkpoint on one GPU.

    ``model`` names an entry of ``pins.json``'s ``startluxModels``.
    ``max_length`` bounds every rendered prompt in tokens. The checkpoints read
    up to 262,144, but without the shared-prefix path a prompt runs in one
    eager batch of at most 65,536 tokens; the default keeps a decision in the
    tens of milliseconds. ``graphs`` records StartLux's CUDA graphs at
    start-up, its fast path for short requests.
    """

    pins_path: Path
    source_dir: Path
    model: str = "startlux-4b"
    gpu: str = "0"
    lock_dir: Path | None = None
    max_length: int = 8192
    graphs: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_length <= 65536:
            raise ValueError("max_length must be between 1 and 65536")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def weight_files(checkpoint: Path) -> list[Path]:
    """The checkpoint's weight shards, in the order the index lists them."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    return [checkpoint / name for name in sorted(set(index["weight_map"].values()))]


def weights_checksum(checkpoint: Path) -> str:
    """One checksum over every weight shard: the SHA-256 of the shards' names and SHA-256s."""
    lines = "".join(f"{path.name}:{_sha256(path)}\n" for path in weight_files(checkpoint))
    return hashlib.sha256(lines.encode()).hexdigest()


def verify_published(checkpoint: Path) -> str:
    """Check every shard against the object id the hub published for it, and return the checkpoint's checksum.

    In a Hugging Face cache snapshot each shard is a link to a blob named by
    the SHA-256 the repository published for that file, so the file's own hash
    must equal the name of the blob it links to. Only that first link is read:
    the blob may itself link into a further content store.
    """
    lines = []
    for path in weight_files(checkpoint):
        if not path.is_symlink():
            raise ValueError(f"{path} is not in a Hugging Face cache snapshot")
        published = Path(os.readlink(path)).name
        actual = _sha256(path)
        if actual != published:
            raise ValueError(f"StartLux weight hash mismatch: {path.name}")
        lines.append(f"{path.name}:{actual}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def import_startlux(source_dir: Path, pins: Mapping[str, Any]) -> tuple[Any, Any, str]:
    """StartLux's model and prompt modules from the pinned, unmodified checkout, and the checkout's revision."""
    revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != pins["startluxSource"]["revision"] or dirty:
        raise ValueError("StartLux requires the pinned, unmodified source checkout in pins.json")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    model = importlib.import_module("startlux_decision.model")
    prompts = importlib.import_module("startlux_decision.jevfmt")
    return model, prompts, revision


def resolve_checkpoint(settings: StartLuxSettings, pins: Mapping[str, Any]) -> dict[str, Any]:
    """The pinned checkpoint's directory in the Hugging Face cache, verified against the published hashes."""
    if settings.model not in pins.get("startluxModels", {}):
        raise ValueError(f"Unknown StartLux model {settings.model!r}")
    from huggingface_hub import snapshot_download

    pin = pins["startluxModels"][settings.model]
    path = Path(snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
    return {
        "repo": pin["repo"],
        "revision": pin["revision"],
        "path": str(path),
        "weightsSha256": verify_published(path),
    }


# ---------------------------------------------------------------- request handling


def validate_questions(questions: Mapping[str, Any]) -> None:
    """Refuse a question StartLux would answer differently from what was asked."""
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Expected at least one question")
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Question IDs must be nonempty strings")
        if not isinstance(question, Mapping) or not set(question) <= {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields")
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"Question {key} must be one of {', '.join(QUESTION_TYPES)}")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"Question {key} needs instructions")
        if question["type"] == "choice":
            criteria = question.get("criteria")
            if not isinstance(criteria, Mapping) or len(criteria) < 2:
                raise ValueError(f"Question {key} needs at least two choices")


def _probabilities(values: list[Any], what: str, *, distribution: bool = True) -> None:
    if not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1 for p in values
    ):
        raise ValueError(f"Invalid model probability in {what}")
    if distribution and abs(math.fsum(values) - 1) > 1e-5:
        raise ValueError(f"Model probabilities in {what} do not sum to one")


def option_count(question: Mapping[str, Any]) -> int:
    """How many options StartLux reads for a question: the choices, the score levels, or yes and no."""
    return 2 if question["type"] == "noul" else len(question.get("criteria") or ())


def checked_answer(key: str, question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """Refuse a partial or invalid answer; a choice keeps the offered order and must be its distribution's argmax."""
    if not isinstance(answer, Mapping) or answer.get("type") != question["type"]:
        raise ValueError(f"StartLux answered question {key} with another type")
    if question["type"] == "noul":
        _probabilities([answer.get("noul")], key, distribution=False)
        return dict(answer)
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError(f"StartLux returned no probabilities for {key}")
    if question["type"] == "score":
        _probabilities(list(probabilities.values()), key)
        return dict(answer)
    keys = list(question["criteria"])
    if set(probabilities) != set(keys):
        raise ValueError("Model probabilities must cover exactly the offered choices")
    values = [probabilities[k] for k in keys]
    _probabilities(values, key)
    best = keys[max(range(len(keys)), key=values.__getitem__)]
    if answer.get("choice") != best:
        raise ValueError("Model choice differs from deterministic argmax in offered order")
    return {**answer, "choice": best, "probabilities": dict(zip(keys, values, strict=True))}


def pack_request(
    parts: Mapping[str, Any],
    questions: Mapping[str, Any],
    measure: Callable[[str], Mapping[str, int]],
    count: Callable[[str], int],
    max_length: int,
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Remove optional whole entries only, keeping all essential state and every option, until each prompt fits.

    ``measure`` returns the longest rendered prompt of each question for a
    state, and ``count`` the tokens of the state alone. Entries leave ``logs``
    oldest first, then ``memory`` and ``details`` newest first.
    """
    validate_questions(questions)
    if not isinstance(parts, Mapping) or set(parts) - set(SECTIONS):
        raise ValueError("Unknown context sections")
    retained: dict[str, list[str]] = {}
    for section in SECTIONS:
        rows = parts.get(section, [])
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError("Context sections must contain complete text records")
        retained[section] = list(rows)
    trimmed: list[dict[str, str]] = []
    while True:
        state = "\n".join(row for section in SECTIONS for row in retained[section])
        lengths = measure(state)
        if set(lengths) != set(questions):
            raise ValueError("StartLux rendering omitted a question")
        state_tokens = count(state)
        audit = {
            key: {
                "stateTokens": state_tokens,
                "stateRoom": max_length - (lengths[key] - state_tokens),
                "inputTokens": lengths[key],
                "maxInputTokens": max_length,
                "options": option_count(question),
                "encoding": "startlux-lettered-prompt",
                "stateTruncated": False,
            }
            for key, question in questions.items()
        }
        if all(lengths[key] <= max_length for key in questions):
            return state, audit, trimmed
        for section in ("logs", "memory", "details"):
            if retained[section]:
                row = retained[section].pop(0 if section == "logs" else -1)
                trimmed.append({"section": section, "text": row})
                break
        else:
            raise ValueError(STATE_OVERRUN)


def rendered_rows(prompts: Any, state: str, question: Mapping[str, Any], group: int) -> list[dict[str, Any]]:
    """The prompts StartLux renders for one question: one, or for a choice over 26 options, one per group.

    The groups split the options as ``StartLuxDecision._wide`` splits them for
    its first round. A one-option choice is answered without the model.
    """
    criteria = question.get("criteria") or {}
    if question["type"] == "choice" and isinstance(criteria, Mapping) and len(criteria) > prompts.MAX_OPTIONS:
        keys = list(criteria)
        count = -(-len(keys) // group)
        size, extra = divmod(len(keys), count)
        rows, start = [], 0
        for index in range(count):
            end = start + size + (1 if index < extra else 0)
            subset = {key: criteria[key] for key in keys[start:end]}
            rows.append(prompts.from_systemone(state, dict(question, criteria=subset)))
            start = end
        return rows
    return [prompts.from_systemone(state, dict(question))]


class StartLuxDecisionModel:
    """A loaded StartLux-Decision checkpoint. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(self, engine: Any, prompts: Any, metadata: Mapping[str, object], max_length: int) -> None:
        self._engine = engine
        self._prompts = prompts
        self._metadata = dict(metadata)
        self._max_length = max_length

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: StartLuxSettings) -> Iterator[StartLuxDecisionModel]:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false"
        )
        pins = json.loads(settings.pins_path.read_text())
        upstream, prompts, revision = import_startlux(settings.source_dir, pins)
        with gpu_lock(settings.gpu, settings.lock_dir) as uuid:
            model_record = resolve_checkpoint(settings, pins)
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                import transformers

                torch.manual_seed(1)
                engine = upstream.StartLuxDecision(
                    model_record["path"],
                    device="cuda",
                    max_length=settings.max_length,
                    graphs=settings.graphs,
                    images=False,
                )
                graph_difference = cls._prepare(engine)
            metadata = {
                "family": FAMILY,
                "model": settings.model,
                **model_record,
                "sourceCommit": revision,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": settings.gpu,
                "gpuUuid": uuid,
                "dtype": "bfloat16",
                "fastKernels": bool(engine.fast_kernels),
                "cudaGraphs": len(engine.graphs),
                "sharedPrefix": False,
                "graphSelfTestMaxDifference": graph_difference,
                "temperatureByType": dict(engine.temperature),
                "maxLen": settings.max_length,
                "python": sys.version.split()[0],
            }
            yield cls(engine, prompts, metadata, settings.max_length)

    @staticmethod
    def _prepare(engine: Any) -> float:
        """Turn off the shared-prefix path, then run StartLux's start-up check and return its largest difference.

        The check is the one ``startlux_decision.server`` makes, with its
        threshold: one request on the graph and eager paths, and the graphs are
        dropped if they disagree.
        """
        engine.share = False
        state = "Tick 30. Credits 2400. Army 3. An enemy barracks is visible to the north."
        questions = {
            "next": {
                "type": "choice",
                "instructions": "Which structure should be built next?",
                "criteria": {"power": "Power plant", "refinery": "Ore refinery", "barracks": "Barracks"},
            }
        }
        engine.decide(state, questions)
        if not engine.graphs:
            return 0.0
        difference = float(engine.self_test(state, questions))
        if difference > 0.02:
            engine.graphs = {}
        return difference

    def _measure(self, state: str, questions: Mapping[str, Any]) -> dict[str, int]:
        prompts, tokenizer = self._prompts, self._engine.tok
        group = getattr(self._engine, "group", 25)
        lengths = {}
        for key, question in questions.items():
            rows = rendered_rows(prompts, state, question, group)
            lengths[key] = max(len(prompts.render_ids(row, tokenizer, max_length=2**31)[0]) for row in rows)
        return lengths

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        started = time.perf_counter()
        tokenizer = self._engine.tok
        state, audit, trimmed = pack_request(
            parts,
            questions,
            lambda text: self._measure(text, questions),
            lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            self._max_length,
        )
        with contextlib.redirect_stdout(sys.stderr):
            raw, usage = self._engine.decide(state, dict(questions))
        if set(raw) != set(questions):
            raise ValueError("StartLux returned a partial question batch")
        answers = {key: checked_answer(key, question, raw[key]) for key, question in questions.items()}  # type: ignore[arg-type]
        return {
            "answers": answers,
            "modelResult": {"answers": raw, "usage": usage, "temperatureByType": self._metadata["temperatureByType"]},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


def startlux_snapshot(checkpoint: Path) -> ModelSnapshot:
    """A StartLux-Decision checkpoint as a ModelSnapshot: the text decoder's shape, parameters, and checksum."""
    from honing.backends.jev.laya import safetensors_parameters

    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    layer = LayerShape(attention_heads=int(text["num_attention_heads"]), ffn_size=int(text["intermediate_size"]))
    return ModelSnapshot(
        checkpoint=CheckpointRef(
            location=str(checkpoint), weight_format="hf_safetensors", checksum=weights_checksum(checkpoint)
        ),
        architecture=ArchitectureSpec(
            hidden_size=int(text["hidden_size"]), layers=(layer,) * int(text["num_hidden_layers"])
        ),
        parameters=sum(safetensors_parameters(path) for path in weight_files(checkpoint)),
    )
