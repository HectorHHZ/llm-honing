"""Load the pinned Nimble-v4-max checkpoint and answer typed questions with it.

Nimble-v4-max (``bespokelabs/Nimble-v4-max``) is Bespoke Labs' decision model, a
27B fine-tune of Qwen3.8-27B. One prompt holds the whole request, the state and
every question in compact plain text, followed by ``Requested field: <name>``.
The request part is read both ways in the full-attention layers and runs once;
each question's short tail runs on a copy of its cache. Every option is one
token, and an answer is the softmax over the options' logits at the answer
position, so nothing is generated.

Inference goes through Bespoke's own ``NimbleEngine``
(``code/nimble/evaluation/decision_index_engine.py`` in the dataset
``bespokelabs/nimble-v4-max-decision-index``) at the revision in ``pins.json``
(``nimbleSource``), with the settings of the published Decision Index run:
shared prefix on and at most 48 question tails per copied cache. The engine
subclasses the Decision Index kit's ``Engine``, so the kit is pinned too
(``decisionIndexKit``). Honing adds what it adds for the other local families:
every source file must match the SHA-256 pinned for it and the kit must be the
pinned, unmodified checkout; every weight shard must match the object id the
hub published for it and the model's own ``weights_sha256.json``; the GPU is
locked for this process; and the state is packed from its four sections and
trimmed by whole entries (never truncated) until every prompt fits
``max_length``, with every trim reported. An overrun of the essential state is
refused with the wording ``RA2Grader`` scores as 0.

The engine answers a score question as a choice over the level indices; this
module returns it as a score, with the expected level and the levels' text, as
the djev bridge does.

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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.kev import QUESTION_OVERRUN
from honing.backends.jev.laya import gpu_lock
from honing.backends.jev.startlux import pack_request, startlux_snapshot, weight_files
from honing.core import ModelSnapshot

FAMILY = "nimble"
# What a checkpoint download needs: weights, their published hashes, the configs and the schema contract.
DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja"]
# The engine loads the base model's tokenizer at the revision the contract names, not the checkpoint's copy.
BASE_TOKENIZER_PATTERNS = ["tokenizer*", "vocab.json", "merges.txt", "*.jinja", "config.json"]
SOURCE_PATTERNS = ["code/**"]
QUESTION_TYPES = ("choice", "noul", "score")
MAX_OPTIONS = 255
# The published Decision Index run's engine settings (runs/nimble-v4-max/environment.json in the source dataset).
MAX_TAILS = 48


@dataclass(frozen=True)
class NimbleSettings:
    """Everything needed to load the Nimble-v4-max checkpoint on one GPU.

    ``source_dir`` is the pinned snapshot of the source dataset, holding
    ``code/nimble``; ``kit_dir`` is a checkout of the Decision Index kit.
    ``model`` names an entry of ``pins.json``'s ``nimbleModels``.
    ``max_length`` bounds every rendered prompt in tokens. The model reads up
    to 32,768; the default keeps a decision well under a second.
    """

    pins_path: Path
    source_dir: Path
    kit_dir: Path
    model: str = "nimble-v4-max"
    gpu: str = "0"
    lock_dir: Path | None = None
    max_length: int = 8192

    def __post_init__(self) -> None:
        if not 0 < self.max_length <= 32768:
            raise ValueError("max_length must be between 1 and 32768")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_source(source_dir: Path, kit_dir: Path, pins: Mapping[str, Any]) -> tuple[str, str]:
    """Check the engine source file by file and the kit's checkout; return the source and kit revisions.

    The source is a dataset snapshot, not a git checkout, so every ``.py``
    file under ``code/`` must match the SHA-256 pinned for it, and no other
    Python file may be there to be imported.
    """
    source = pins["nimbleSource"]
    expected: Mapping[str, str] = source["files"]
    found = {path.relative_to(source_dir).as_posix() for path in (source_dir / "code").rglob("*.py")}
    if found != set(expected):
        raise ValueError("Nimble source files differ from the pinned file list in pins.json")
    for name, digest in expected.items():
        if _sha256(source_dir / name) != digest:
            raise ValueError(f"Nimble source file {name} differs from its pinned SHA-256")
    kit = pins["decisionIndexKit"]
    revision = subprocess.check_output(["git", "-C", str(kit_dir), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(kit_dir), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != kit["revision"] or dirty:
        raise ValueError("Nimble requires the pinned, unmodified Decision Index kit checkout in pins.json")
    return source["revision"], revision


def import_nimble(source_dir: Path, kit_dir: Path, pins: Mapping[str, Any]) -> tuple[Any, Any, str, str]:
    """Bespoke's engine module and the kit's refusal type, from the verified source and kit."""
    source_revision, kit_revision = verify_source(source_dir, kit_dir, pins)
    for path in (str(kit_dir), str(source_dir / "code")):
        if path not in sys.path:
            sys.path.insert(0, path)
    engine = importlib.import_module("nimble.evaluation.decision_index_engine")
    unsupported = importlib.import_module("decision_index.engines").Unsupported
    return engine, unsupported, source_revision, kit_revision


def verify_published(checkpoint: Path) -> str:
    """Check every shard against the hub's object id and the model's ``weights_sha256.json``; return a checksum.

    In a Hugging Face cache snapshot each shard is a link to a blob named by
    the SHA-256 the repository published for that file, so the file's own hash
    must equal the name of the blob it links to, and the model card's own list
    of shard hashes must agree.
    """
    listed = json.loads((checkpoint / "weights_sha256.json").read_text())
    lines = []
    for path in weight_files(checkpoint):
        if not path.is_symlink():
            raise ValueError(f"{path} is not in a Hugging Face cache snapshot")
        actual = _sha256(path)
        if actual != Path(os.readlink(path)).name or actual != listed.get(path.name):
            raise ValueError(f"Nimble weight hash mismatch: {path.name}")
        lines.append(f"{path.name}:{actual}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def resolve_checkpoint(settings: NimbleSettings, pins: Mapping[str, Any]) -> dict[str, Any]:
    """The pinned checkpoint's directory, verified, after checking its contract names the pinned base tokenizer."""
    if settings.model not in pins.get("nimbleModels", {}):
        raise ValueError(f"Unknown Nimble model {settings.model!r}")
    from huggingface_hub import snapshot_download

    pin = pins["nimbleModels"][settings.model]
    path = Path(snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
    contract = json.loads((path / "schema_config.json").read_text())
    if (contract.get("model"), contract.get("revision")) != (pin["base"], pin["baseRevision"]):
        raise ValueError("The Nimble checkpoint's tokenizer contract differs from pins.json")
    return {
        "repo": pin["repo"],
        "revision": pin["revision"],
        "path": str(path),
        "weightsSha256": verify_published(path),
        "base": pin["base"],
        "baseRevision": pin["baseRevision"],
        "contractTask": contract.get("task"),
        "prefixAttention": contract.get("prefix_attention"),
    }


# ---------------------------------------------------------------- request handling


def validate_questions(questions: Mapping[str, Any]) -> None:
    """Refuse a question Nimble would read differently from what was asked."""
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Expected at least one question")
    for key, question in questions.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Question IDs must be nonempty strings")
        if not isinstance(question, Mapping) or not set(question) <= {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields")
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise ValueError(f"Question {key} must be one of {', '.join(QUESTION_TYPES)}")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"Question {key} needs instructions")
        criteria = question.get("criteria")
        if kind == "choice":
            valid = (
                isinstance(criteria, Mapping)
                and 2 <= len(criteria) <= MAX_OPTIONS
                and all(isinstance(k, str) and k.strip() for k in criteria)
            )
        elif kind == "score":
            valid = (
                isinstance(criteria, list)
                and 2 <= len(criteria) <= MAX_OPTIONS
                and all(isinstance(level, str) and level.strip() for level in criteria)
            )
        else:
            valid = criteria is None or (
                isinstance(criteria, Mapping)
                and set(criteria) in ({"true", "false"}, {"true"}, {"false"}, {"yes", "no"}, set())
            )
        if not valid:
            raise ValueError(f"Question {key} has {kind} criteria Nimble cannot read")


def _probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def _distribution(probabilities: Any, keys: list[str], what: str) -> list[float]:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(keys):
        raise ValueError(f"Model probabilities for {what} must cover exactly the offered options")
    values = [probabilities[k] for k in keys]
    if not all(_probability(p) for p in values) or abs(math.fsum(values) - 1) > 1e-5:
        raise ValueError(f"Invalid model probabilities for {what}")
    return values


def checked_answer(key: str, question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """The engine's answer in Honing's form, refusing a partial or invalid one.

    A yes/no answer keeps the probability of yes. A choice keeps the offered
    order, adds its confidence, and must choose its argmax. The engine answers
    a score as a choice over the level indices; it becomes a score with the
    expected level and the levels' text.
    """
    if not isinstance(answer, Mapping):
        raise ValueError(f"Nimble returned no answer for {key}")
    kind = question["type"]
    if kind == "noul":
        if answer.get("type") != "noul" or not _probability(answer.get("noul")):
            raise ValueError(f"Invalid Nimble yes/no answer for {key}")
        return {"type": "noul", "noul": answer["noul"]}
    if answer.get("type") != "choice":
        raise ValueError(f"Nimble answered question {key} with another type")
    keys = list(question["criteria"]) if kind == "choice" else [str(i) for i in range(len(question["criteria"]))]
    values = _distribution(answer.get("probabilities"), keys, key)
    probabilities = dict(zip(keys, values, strict=True))
    best = max(values)
    if answer.get("choice") not in probabilities or probabilities[answer["choice"]] != best:
        raise ValueError(f"Nimble's choice for {key} is not an offered option with the highest probability")
    if kind == "choice":
        return {"type": "choice", "choice": answer["choice"], "confidence": best, "probabilities": probabilities}
    return {
        "type": "score",
        "score": math.fsum(i * p for i, p in enumerate(values)),
        "confidence": best,
        "probabilities": probabilities,
        "legend": dict(zip(keys, question["criteria"], strict=True)),
    }


def engine_state(state: str) -> str:
    """The state as the engine reads it: it passes an empty state in its JSON form, ``""``."""
    return json.dumps(state) if not state.strip() else state


class NimbleDecisionModel:
    """A loaded Nimble checkpoint. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(
        self, engine: Any, module: Any, unsupported: type[Exception], metadata: Mapping[str, object], max_length: int
    ) -> None:
        self._engine = engine
        self._module = module
        self._unsupported = unsupported
        self._metadata = dict(metadata)
        self._max_length = max_length

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: NimbleSettings) -> Iterator[NimbleDecisionModel]:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false"
        )
        pins = json.loads(settings.pins_path.read_text())
        module, unsupported, source_revision, kit_revision = import_nimble(settings.source_dir, settings.kit_dir, pins)
        with gpu_lock(settings.gpu, settings.lock_dir) as uuid:
            model_record = resolve_checkpoint(settings, pins)
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                import transformers

                engine = module.NimbleEngine(
                    adapter=model_record["path"], max_tokens=settings.max_length, max_tails=MAX_TAILS
                )
                engine.warmup()
            metadata = {
                "family": FAMILY,
                "model": settings.model,
                **model_record,
                "sourceRevision": source_revision,
                "kitRevision": kit_revision,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": settings.gpu,
                "gpuUuid": uuid,
                "dtype": "bfloat16",
                "sharedPrefix": bool(engine.shared_prefix),
                "maxTails": MAX_TAILS,
                "maxLen": settings.max_length,
                "license": "cc-by-nc-4.0",
                "python": sys.version.split()[0],
            }
            model = cls(engine, module, unsupported, metadata, settings.max_length)
            with contextlib.redirect_stdout(sys.stderr):
                model.warmup()
            yield model

    def warmup(self) -> None:
        """Compile representative serving paths before the caller can report readiness.

        The kit's short color question leaves the longer prefill and shared-prefix
        cache/tail kernels cold. Their first compilation can exceed the game
        clients' deadlines. Exercise single questions and batches of two and three,
        including the long single-question path above 4096 tokens. Use the actual
        prompt builder to keep every request within the configured context limit.
        This covers common serving shapes, not every possible dynamic batch.
        """
        for count in (1, 2, 3):
            questions = {
                f"action{index}": {
                    "type": "choice",
                    "instructions": "Choose the action supported by the state.",
                    "criteria": {f"option{option}": f"Action {option}" for option in range(options)},
                }
                for index, options in enumerate((9, 3, 2)[:count])
            }
            for target in sorted({min(self._max_length, size) for size in (512, 2048, 4608)}):
                if max(self._measure("", questions).values()) > target:
                    continue  # A deliberately tiny context may not fit this question batch.
                low, high = 0, target
                while low < high:
                    middle = (low + high + 1) // 2
                    if max(self._measure("The current action is safe. " * middle, questions).values()) <= target:
                        low = middle
                    else:
                        high = middle - 1
                self.predict({"essential": ["The current action is safe. " * low]}, questions)

    def _measure(self, state: str, questions: Mapping[str, Any]) -> dict[str, int]:
        """Each question's prompt length, from the engine's own prompt builder."""
        module = self._module
        body = {
            "state": engine_state(state),
            "questions": {k: module.native_question(q) for k, q in questions.items()},
        }
        context, schema = module.native_schema(body)
        prompt = self._engine.prepare(self._engine.tokenizer, context, schema, 2**31)
        return {name: len(ids) for name, ids in zip(prompt.names, prompt.full_ids, strict=True)}

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        started = time.perf_counter()
        tokenizer = self._engine.tokenizer
        state, audit, trimmed = pack_request(
            parts,
            questions,
            lambda text: self._measure(text, questions),
            lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            self._max_length,
            validate=validate_questions,
            encoding="nimble-compact-prompt",
        )
        with contextlib.redirect_stdout(sys.stderr):
            try:
                raw, _ = self._engine(state, dict(questions))
            except self._unsupported as error:
                # The engine's own limit check. The measure above should prevent it; if a prompt still overruns, the
                # refusal uses the wording RA2Grader scores as 0.
                if "maximum context length" in str(error):
                    raise ValueError(QUESTION_OVERRUN.format(key=", ".join(questions), detail=str(error))) from error
                raise ValueError(f"Nimble refused the request: {error}") from error
        answers_raw = raw.get("answers", {}) if isinstance(raw, Mapping) else {}
        if set(answers_raw) != set(questions):
            raise ValueError("Nimble returned a partial question batch")
        answers = {key: checked_answer(key, question, answers_raw[key]) for key, question in questions.items()}  # type: ignore[arg-type]
        return {
            "answers": answers,
            # The RA2 client stack reads modelResult.answers as the answers it acts on, so they are the converted
            # answers; the engine's own output is kept beside them.
            "modelResult": {"answers": answers, "engineAnswers": answers_raw, "usage": raw.get("usage")},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


def nimble_snapshot(checkpoint: Path) -> ModelSnapshot:
    """The checkpoint as a ModelSnapshot: the Qwen3.5 text decoder's shape, parameters, and checksum."""
    return startlux_snapshot(checkpoint)
