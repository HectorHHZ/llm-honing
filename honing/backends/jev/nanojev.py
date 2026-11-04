"""Load a pinned NanoJev checkpoint and answer typed choice questions with it.

NanoJev (``C-Tianyu/NanoJev``, source ``TianyuCodings/NanoJev``) is a decision
model on a full Qwen3-0.6B backbone: every option is scored on its own
candidate path (state, question, then the candidate), a scalar head and a set
attention head turn the path ends into option logits. This module is the one
implementation of NanoJev inference in Honing; the HTTP decision server and the
RA2 integration's stdio worker both use it.

Inference goes through NanoJev's own ``DecisionPredictor`` and
``prepare_examples`` (``scripts/predict_toy_decisions.py``) at the commit in
``pins.json``, as the RA2 experiment in #49 did, and the request handling
follows #49's NanoJev adapter: the state packs the four sections and is
trimmed by whole entries until every candidate path fits the input limit, the
returned distribution must cover exactly the offered options, and its argmax
must be the reported choice. Honing adds what it adds for Laya and Kev: the
source must be the pinned, unmodified checkout, the model files must match the
checksums the model repository publishes (``SHA256_MANIFEST.json``), a trained
checkpoint must match its recorded hash, the GPU is locked for this process,
and an overrun is refused with the wording ``RA2Grader`` scores as 0.

``NanoJevSettings.inference`` can instead score a request's candidate paths as
one token tree (``honing.backends.jev.nanojev_tree``): the same model, inputs
and answer checks, with the shared prefixes computed once. That model
(:class:`NanoJevTreeDecisionModel`) also answers several requests in one forward
and splits a call into CPU and GPU work for the decision server.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from honing.backends.jev.kev import STATE_OVERRUN
from honing.backends.jev.laya import gpu_lock
from honing.backends.jev.nanojev_tree import (
    INFERENCE_MODES,
    PrefetchedTokenizer,
    TreeCall,
    TreeScorer,
    nanojev_segments,
)
from honing.backends.jev.token_budget import SECTIONS
from honing.core import ArchitectureSpec, CheckpointRef, LayerShape, ModelSnapshot

FAMILY = "nanojev"
REQUIRED_FILES = (
    "best.safetensors",
    "config.json",
    "backbone_config/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
# What a checkpoint download needs: the files above, and the published checksums.
DOWNLOAD_PATTERNS = ["best.safetensors", "config.json", "backbone_config/*", "tokenizer/*", "SHA256_MANIFEST.json"]


@dataclass(frozen=True)
class NanoJevSettings:
    """Everything needed to load one NanoJev checkpoint on one GPU.

    ``model`` names an entry of ``pins.json``'s ``nanojevModels``.
    ``max_length`` is the longest candidate path in tokens; 2,048 is what #49
    ran (the checkpoint was trained up to 8,192). ``precision`` is ``bf16``
    (fp32 parameters, bf16 autocast forward, NanoJev's default) or ``fp32``.
    ``inference`` is ``reference`` (NanoJev's ``DecisionPredictor``, one padded
    row per candidate path), ``tree`` or ``tree-compiled`` (one forward over the
    request's token tree; see ``nanojev_tree``). ``tree-compiled`` needs ``bf16``.
    """

    pins_path: Path
    source_dir: Path
    model: str = "nanojev"
    gpu: str = "0"
    checkpoint_manifest: Path | None = None
    lock_dir: Path | None = None
    max_length: int = 2048
    precision: str = "bf16"
    inference: str = "reference"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if self.inference not in INFERENCE_MODES:
            raise ValueError(f"inference must be one of {', '.join(INFERENCE_MODES)}")
        if self.inference == "tree-compiled" and self.precision != "bf16":
            raise ValueError("tree-compiled runs under bf16 autocast; use precision bf16, or inference tree")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def weights_checksum(checkpoint: Path) -> str:
    """A NanoJev checkpoint's weights are one file, backbone and heads together."""
    return _sha256(checkpoint / "best.safetensors")


def import_nanojev(source_dir: Path, pins: Mapping[str, Any]) -> tuple[ModuleType, str]:
    """NanoJev's predictor module from the pinned, unmodified checkout, and the checkout's revision."""
    revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != pins["nanojevSource"]["revision"] or dirty:
        raise ValueError("NanoJev requires the pinned, unmodified source checkout in pins.json")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    spec = importlib.util.spec_from_file_location(
        "honing_nanojev_predict", source_dir / "scripts/predict_toy_decisions.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("NanoJev's predictor module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, revision


def verify_published(checkpoint: Path) -> None:
    """Check the files NanoJev loads against the checksums the model repository publishes."""
    published = json.loads((checkpoint / "SHA256_MANIFEST.json").read_text())
    expected = {entry["path"]: entry["sha256"] for entry in published["files"]}
    for name in REQUIRED_FILES:
        if name not in expected:
            raise ValueError(f"The published NanoJev checksums omit {name}")
        if _sha256(checkpoint / name) != expected[name]:
            raise ValueError(f"NanoJev file hash mismatch: {name}")


def resolve_checkpoint(settings: NanoJevSettings, pins: Mapping[str, Any], revision: str) -> dict[str, Any]:
    """The checkpoint directory and what it is: the pinned release, or a trained checkpoint named by its manifest."""
    if settings.model not in pins.get("nanojevModels", {}):
        raise ValueError(f"Unknown NanoJev model {settings.model!r}")
    pin = pins["nanojevModels"][settings.model]
    if settings.checkpoint_manifest is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
        verify_published(path)
        model: dict[str, Any] = {"repo": pin["repo"], "revision": pin["revision"], "path": str(path)}
    else:
        trained = json.loads(settings.checkpoint_manifest.read_text())
        if trained.get("kind") != "trained-nanojev" or trained.get("sourceRevision") != revision:
            raise ValueError("Unrecognized trained checkpoint manifest")
        if trained.get("parent", {}).get("revision") != pin["revision"]:
            raise ValueError("Trained checkpoint parent does not match selected model")
        path = Path(trained["path"])
        model = {key: trained[key] for key in ("repo", "revision", "path", "parent")}
        model["trainingManifestSha256"] = hashlib.sha256(settings.checkpoint_manifest.read_bytes()).hexdigest()
    checksum = weights_checksum(path)
    if settings.checkpoint_manifest is not None and checksum != trained["weightsSha256"]:
        raise ValueError("Trained checkpoint weight hash mismatch")
    run_config = json.loads((path / "config.json").read_text())
    if (run_config.get("model"), run_config.get("resolved_model_revision")) != (pin["base"], pin["baseRevision"]):
        raise ValueError("The NanoJev checkpoint's base differs from pins.json")
    return {**model, "weightsSha256": checksum, "base": pin["base"], "baseRevision": pin["baseRevision"]}


# ---------------------------------------------------------------- request handling (as #49's NanoJev adapter)


def validate_questions(questions: Mapping[str, Any]) -> None:
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("Expected at least one choice question")
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Question IDs must be nonempty strings")
        if not isinstance(question, Mapping) or set(question) != {"type", "instructions", "criteria"}:
            raise ValueError("Unexpected question fields")
        if question["type"] != "choice" or not isinstance(question["instructions"], str):
            raise ValueError("Only explicit choice questions are supported")
        criteria = question["criteria"]
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 255:
            raise ValueError("Expected 2-255 ordered choices")
        if not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in criteria.items()):
            raise ValueError("Every choice needs an ID and a description")


def checked_answer(criteria: Mapping[str, Any], probabilities: Any, choice: Any = None) -> dict[str, Any]:
    """Keep the offered order; refuse a partial or invalid distribution, or a choice that is not its argmax."""
    keys = list(criteria)
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(keys):
        raise ValueError("Model probabilities must cover exactly the offered choices")
    values = [probabilities[key] for key in keys]
    if not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1 for p in values
    ):
        raise ValueError("Invalid model probability")
    if abs(math.fsum(values) - 1) > 1e-5:
        raise ValueError("Model probabilities do not sum to one")
    best = keys[max(range(len(keys)), key=values.__getitem__)]
    if choice is not None and choice != best:
        raise ValueError("Model choice differs from deterministic argmax in offered order")
    return {"choice": best, "probabilities": dict(zip(keys, values, strict=True))}


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def payload(state: str, questions: Mapping[str, Any]) -> dict[str, Any]:
    return {"states": [{"id": "decision", "state": state, "questions": copy.deepcopy(dict(questions))}]}


def measure(
    prepare_examples: Callable[..., list[dict[str, Any]]],
    tokenizer: Any,
    state: str,
    questions: Mapping[str, Any],
    keep: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Each question's candidate paths as NanoJev encodes them, without a length limit: a tokenizer-only measure.

    ``keep``, when given, holds only this state's examples afterwards; ``prepare_examples`` checks the length
    limit and nothing else, so once every path fits they are the examples the model scores.
    """
    examples = prepare_examples(payload(state, questions), tokenizer, 2**30)
    if keep is not None:
        keep.clear()
        keep[state] = examples
    if len(examples) != len(questions) or {e["qid"] for e in examples} != set(questions):
        raise ValueError("NanoJev encoding changed the question set")
    tokens = len(tokenizer.encode(state, add_special_tokens=False))
    measured = {}
    for example in examples:
        if example["candidate_ids"] != list(questions[example["qid"]]["criteria"]):
            raise ValueError("NanoJev encoding reordered offered choices")
        if len(example["leaf_tokens"]) != len(example["candidate_ids"]):
            raise ValueError("NanoJev encoding omitted candidate paths")
        measured[example["qid"]] = {
            "stateTokens": tokens,
            "inputTokens": max(map(len, example["leaf_tokens"])),
            "encoding": "nanojev-native-candidate-paths",
            "candidatePathSha256": [token_digest(path) for path in example["leaf_tokens"]],
        }
    return measured


def pack_request(
    parts: Mapping[str, Any],
    questions: Mapping[str, Any],
    count: Callable[[str], dict[str, dict[str, Any]]],
    max_length: int,
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Remove optional whole entries only, keeping all essential state and every option, until each path fits."""
    validate_questions(questions)
    if not isinstance(parts, Mapping) or set(parts) - set(SECTIONS):
        raise ValueError("Unknown context sections")
    retained: dict[str, list[str]] = {}
    for section in SECTIONS:
        rows = parts.get(section, [])
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError("Context sections must contain complete text records")
        retained[section] = list(rows)
    if not retained["essential"]:
        raise ValueError("Essential normal context is required")
    trimmed: list[dict[str, str]] = []
    while True:
        state = "\n".join(row for section in SECTIONS for row in retained[section])
        measurements = count(state)
        if set(measurements) != set(questions):
            raise ValueError("Native encoding omitted a question")
        audit = {}
        for key, entry in measurements.items():
            total, state_tokens = entry["inputTokens"], entry["stateTokens"]
            if type(total) is not int or type(state_tokens) is not int or not 0 <= state_tokens <= total:
                raise ValueError("Invalid native token accounting")
            audit[key] = {
                **entry,
                "stateRoom": max_length - (total - state_tokens),
                "options": len(questions[key]["criteria"]),
                "stateTruncated": False,
                "maxInputTokens": max_length,
            }
        if all(a["inputTokens"] <= max_length for a in audit.values()):
            return state, audit, trimmed
        for section in ("logs", "memory", "details"):
            if retained[section]:
                row = retained[section].pop(0 if section == "logs" else -1)
                trimmed.append({"section": section, "text": row})
                break
        else:
            raise ValueError(STATE_OVERRUN)


class NanoJevDecisionModel:
    """A loaded NanoJev checkpoint. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(
        self,
        predictor: Any,
        prepare_examples: Any,
        metadata: Mapping[str, object],
        scorer: TreeScorer | None = None,
    ) -> None:
        self._predictor = predictor
        self._prepare_examples = prepare_examples
        self._metadata = dict(metadata)
        self._scorer = scorer

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: NanoJevSettings) -> Iterator[NanoJevDecisionModel]:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false"
        )
        pins = json.loads(settings.pins_path.read_text())
        upstream, revision = import_nanojev(settings.source_dir, pins)
        with gpu_lock(settings.gpu, settings.lock_dir) as uuid:
            model_record = resolve_checkpoint(settings, pins, revision)
            with contextlib.redirect_stdout(sys.stderr):
                import torch
                import transformers

                torch.set_num_threads(4)
                torch.manual_seed(1)
                predictor = upstream.DecisionPredictor(
                    model_record["path"],
                    max_length=settings.max_length,
                    device_name="cuda:0",
                    precision=settings.precision,
                )
                scorer = None
                if settings.inference != "reference":
                    scorer = TreeScorer(predictor, upstream, settings.inference)
            metadata = {
                "family": FAMILY,
                "model": settings.model,
                **model_record,
                "sourceCommit": revision,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": settings.gpu,
                "gpuUuid": uuid,
                "maxLen": settings.max_length,
                "parameterStorage": "float32",
                "forwardAutocast": "bfloat16" if settings.precision == "bf16" else "disabled",
                "inference": settings.inference,
                "python": sys.version.split()[0],
            }
            if scorer is None:
                yield cls(predictor, upstream.prepare_examples, metadata)
            else:
                yield NanoJevTreeDecisionModel(predictor, upstream, metadata, scorer)

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        started = time.perf_counter()
        tokenizer = self._predictor.tokenizer
        state, audit, trimmed = pack_request(
            parts,
            questions,
            lambda text: measure(self._prepare_examples, tokenizer, text, questions),
            self._predictor.limit,
        )
        with contextlib.redirect_stdout(sys.stderr):
            if self._scorer is None:
                response = self._predictor.predict(payload(state, questions), batch_questions=1, temperature=1.0)
            else:
                response = self._scorer.predict(payload(state, questions))
        return self._reply(questions, response, state, audit, trimmed, (time.perf_counter() - started) * 1000)

    @staticmethod
    def _reply(
        questions: Mapping[str, Any],
        response: Mapping[str, Any],
        state: str,
        audit: Mapping[str, Any],
        trimmed: list[dict[str, str]],
        latency_ms: float,
    ) -> dict[str, object]:
        """The server's answer from NanoJev's response, after checking it answered every question once."""
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
            "latencyMs": latency_ms,
        }


@dataclass
class _Encoded:
    """One request as the tree model encodes it: its packed state, token audit and trim record, NanoJev's
    payload and validated states for it, and its examples."""

    questions: Mapping[str, Any]
    state: str
    audit: dict[str, dict[str, Any]]
    trimmed: list[dict[str, str]]
    states: list[dict[str, Any]]
    examples: list[dict[str, Any]]


@dataclass
class _Prepared:
    """A call's CPU work: its requests, their merged forward, and how long the encoding took."""

    requests: list[_Encoded]
    call: TreeCall
    prepare_ms: float


class NanoJevTreeDecisionModel(NanoJevDecisionModel):
    """NanoJev with a token-tree inference mode, as :meth:`NanoJevDecisionModel.load` returns it for one.

    Besides ``predict`` it answers several requests in one forward
    (``predict_many``), each as ``predict`` would, and splits a call in two:
    ``prepare`` packs and encodes the requests and builds their tree, CPU work
    that may run on any thread, and ``run`` scores them on the GPU. Each request
    is encoded once, with one tokenizer call for every string NanoJev encodes,
    and the packing's measurement gives the paths the model scores.
    """

    def __init__(
        self, predictor: Any, upstream: ModuleType, metadata: Mapping[str, object], scorer: TreeScorer
    ) -> None:
        super().__init__(predictor, upstream.prepare_examples, metadata, scorer)
        self._upstream, self._tree = upstream, scorer
        self._encoding = threading.Lock()  # the tokenizer is shared; prepare may run on several request threads

    def _encode(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> _Encoded:
        tokenizer = self._predictor.tokenizer
        kept: dict[str, list[dict[str, Any]]] = {}

        def count(text: str) -> dict[str, dict[str, Any]]:
            prefetched = PrefetchedTokenizer(tokenizer, [text, *nanojev_segments(text, questions)])
            return measure(self._prepare_examples, prefetched, text, questions, kept)

        state, audit, trimmed = pack_request(parts, questions, count, self._predictor.limit)
        states = self._upstream.validate_request(payload(state, questions))
        return _Encoded(questions, state, audit, trimmed, states, kept[state])

    def prepare(self, requests: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]) -> _Prepared:
        started = time.perf_counter()
        with self._encoding:
            encoded = [self._encode(parts, questions) for parts, questions in requests]
        call = self._tree.prepare([example for request in encoded for example in request.examples])
        return _Prepared(encoded, call, (time.perf_counter() - started) * 1000)

    def run(self, prepared: _Prepared) -> list[dict[str, object]]:
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            probabilities = self._tree.score(prepared.call)
        replies, offset = [], 0
        for request in prepared.requests:
            count = len(request.examples)
            response = self._tree.answer(
                request.states,
                request.examples,
                probabilities[offset : offset + count],
                prepared.call,
                len(prepared.requests),
            )
            offset += count
            latency_ms = prepared.prepare_ms + (time.perf_counter() - started) * 1000
            replies.append(
                self._reply(request.questions, response, request.state, request.audit, request.trimmed, latency_ms)
            )
        return replies

    def predict_many(
        self, requests: Sequence[tuple[Mapping[str, object], Mapping[str, object]]]
    ) -> list[dict[str, object]]:
        return self.run(self.prepare(requests))

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        return self.predict_many([(parts, questions)])[0]


def nanojev_snapshot(checkpoint: Path) -> ModelSnapshot:
    """A NanoJev checkpoint as a ModelSnapshot: the backbone's shape, parameter count, and weight checksum."""
    from honing.backends.jev.laya import safetensors_parameters

    config = json.loads((checkpoint / "backbone_config" / "config.json").read_text())
    layer = LayerShape(attention_heads=int(config["num_attention_heads"]), ffn_size=int(config["intermediate_size"]))
    weights = checkpoint / "best.safetensors"
    return ModelSnapshot(
        checkpoint=CheckpointRef(location=str(checkpoint), weight_format="hf_safetensors", checksum=_sha256(weights)),
        architecture=ArchitectureSpec(
            hidden_size=int(config["hidden_size"]), layers=(layer,) * int(config["num_hidden_layers"])
        ),
        parameters=safetensors_parameters(weights),
    )
