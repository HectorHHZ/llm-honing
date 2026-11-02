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

With ``engine="sglang"`` (RFC #103, prototype) the forward pass runs in an
SGLang server started from its own environment instead, and StartLux's decision
code reads the option letters from the log-probabilities it returns
(:mod:`honing.backends.sglang.startlux`). Honing still checks the weights
against the published hashes and reports their checksum. Requests then run
concurrently, batched by the engine, and its radix cache reuses prompt
beginnings across questions and requests unless ``prefix_cache`` is off.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.kev import QUESTION_OVERRUN, STATE_OVERRUN
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
    engine: str = "native"
    sglang_python: Path | None = None
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_length <= 65536:
            raise ValueError("max_length must be between 1 and 65536")
        if self.engine not in ("native", "sglang"):
            raise ValueError("engine must be native or sglang")
        if self.engine == "sglang" and self.sglang_python is None:
            raise ValueError("engine sglang needs sglang_python, the interpreter of the SGLang environment")


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
    """Refuse a partial or invalid answer. A choice's probabilities are returned in the offered order, and its
    choice must be an offered option with the highest probability.

    StartLux breaks a tie by its own order, which for a choice over 26 options
    is the finalists' first-round ranking, so a tied choice is kept as StartLux
    made it.
    """
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
    choice = answer.get("choice")
    if choice not in probabilities or probabilities[choice] != max(values):
        raise ValueError("Model choice is not an offered option with the highest probability")
    return {**answer, "probabilities": dict(zip(keys, values, strict=True))}


def pack_request(
    parts: Mapping[str, Any],
    questions: Mapping[str, Any],
    measure: Callable[[str], Mapping[str, int]],
    count: Callable[[str], int],
    max_length: int,
    *,
    validate: Callable[[Mapping[str, Any]], None] = validate_questions,
    encoding: str = "startlux-lettered-prompt",
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Remove optional whole entries only, keeping all essential state and every option, until each prompt fits.

    ``measure`` returns the longest rendered prompt of each question for a
    state, and ``count`` the tokens of the state alone. Entries leave ``logs``
    oldest first, then ``memory`` and ``details`` newest first. ``validate`` and
    ``encoding`` let another family that renders whole prompts (Nimble) reuse
    this with its own question rules and audit label.
    """
    validate(questions)
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
            raise ValueError("Prompt rendering omitted a question")
        state_tokens = count(state)
        audit = {
            key: {
                "stateTokens": state_tokens,
                "stateRoom": max_length - (lengths[key] - state_tokens),
                "inputTokens": lengths[key],
                "maxInputTokens": max_length,
                "options": option_count(question),
                "encoding": encoding,
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


def later_round_sizes(options: int, group: int, keep: int, max_options: int) -> list[int]:
    """How many options each prompt after the first round can hold, for a choice over ``max_options`` options.

    This follows ``StartLuxDecision._wide``: the options split into
    near-equal groups of at most ``group``, and the top ``keep`` of each go on.
    Finalists over ``max_options`` run as a new wide choice, whose largest
    group is listed, and the last round is one prompt over all finalists.
    """
    sizes: list[int] = []
    while True:
        groups = -(-options // group)
        size, extra = divmod(options, groups)
        options = sum(min(keep, size + (1 if index < extra else 0)) for index in range(groups))
        if options <= max_options:
            return [*sizes, options]
        sizes.append(-(-options // -(-options // group)))


def longest_round(prompts: Any, tokenizer: Any, state: str, question: Mapping[str, Any], size: int) -> dict[str, Any]:
    """The longest prompt a later round of ``size`` options can render, as a row to render.

    Which options reach a later round depends on the model's first-round
    scores, so the row holds the ``size`` options whose lines take the most
    tokens. Each id gets a trailing ``_``, which keeps it from reading as a
    bare letter or number, so every line shows its id and description, the
    longest form StartLux renders for it. The Qwen tokenizers split at line
    breaks, so a prompt's length is the sum of its lines and no set of
    finalists renders a longer prompt.
    """
    criteria = question["criteria"]

    def line_tokens(key: str) -> int:
        return len(tokenizer.encode(f"{key}_: {prompts.value_text(criteria[key]) or ''}", add_special_tokens=False))

    longest = sorted(criteria, key=line_tokens, reverse=True)[:size]
    return prompts.from_systemone(state, dict(question, criteria={f"{key}_": criteria[key] for key in longest}))


class StartLuxDecisionModel:
    """A loaded StartLux-Decision checkpoint. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(
        self, engine: Any, prompts: Any, metadata: Mapping[str, object], max_length: int, concurrent: bool = False
    ) -> None:
        self._engine = engine
        self._prompts = prompts
        self._metadata = dict(metadata)
        self._max_length = max_length
        # An SGLang engine batches concurrent requests itself; the native model answers one at a time.
        self.concurrent = concurrent

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
        if settings.engine == "sglang":
            with cls._load_sglang(settings, pins, upstream, prompts, revision) as model:
                yield model
            return
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

    @classmethod
    @contextlib.contextmanager
    def _load_sglang(
        cls, settings: StartLuxSettings, pins: Mapping[str, Any], upstream: Any, prompts: Any, revision: str
    ) -> Iterator[StartLuxDecisionModel]:
        from honing.backends.sglang.engine import SGLangEngine, SGLangSettings
        from honing.backends.sglang.startlux import sglang_startlux

        assert settings.sglang_python is not None
        with gpu_lock(settings.gpu, settings.lock_dir) as uuid:
            model_record = resolve_checkpoint(settings, pins)
            checkpoint = Path(model_record["path"])
            engine_settings = SGLangSettings(
                python=settings.sglang_python,
                gpu=uuid,
                prefix_cache=settings.prefix_cache,
                # Room for the longest prompt StartLux renders and the one token asked for after it.
                context_length=settings.max_length + 8,
                log_path=Path(os.environ.get("SGLANG_LOG", os.devnull)),
            )
            with SGLangEngine.launch(checkpoint, engine_settings) as engine:
                decision = sglang_startlux(upstream, prompts, engine, checkpoint, settings.max_length)
                metadata = {
                    "family": FAMILY,
                    "model": settings.model,
                    **model_record,
                    "sourceCommit": revision,
                    "engine": "sglang",
                    "engineVersions": engine.versions,
                    "prefixCache": settings.prefix_cache,
                    "gpu": settings.gpu,
                    "gpuUuid": uuid,
                    "sharedPrefix": False,
                    "temperatureByType": dict(decision.temperature),
                    "maxLen": settings.max_length,
                    "python": sys.version.split()[0],
                }
                yield cls(decision, prompts, metadata, settings.max_length, concurrent=True)

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
        """The longest prompt each question can render: its first-round prompts and, for a choice over 26 options,
        the longest prompt each later round can render."""
        prompts, tokenizer = self._prompts, self._engine.tok
        group, keep = self._engine.group, self._engine.keep
        lengths = {}
        for key, question in questions.items():
            rows = rendered_rows(prompts, state, question, group)
            criteria = question.get("criteria") or {}
            if len(rows) > 1:
                for size in later_round_sizes(len(criteria), group, keep, prompts.MAX_OPTIONS):
                    rows.append(longest_round(prompts, tokenizer, state, question, size))
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
            try:
                raw, usage = self._engine.decide(state, dict(questions))
            except ValueError as error:
                # StartLux's own limit check. The measure above should prevent it; if a prompt still overruns, the
                # refusal uses the wording RA2Grader scores as 0.
                if re.fullmatch(r"length \d+ > \d+", str(error)):
                    raise ValueError(QUESTION_OVERRUN.format(key=", ".join(questions), detail=str(error))) from error
                raise
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
