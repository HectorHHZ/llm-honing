"""Reverify native runtime files without loading model objects or allocating a GPU."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from .upstream_adapters import JevOmniChoices, NanoJevChoices
from .worker import load_module, verify_nano_source, verify_prepared


def verify_runtime(runtime, pins):
    family = runtime.get("family")
    manifest, expected = verify_prepared(Path(runtime["preparedManifestPath"]), family, pins)
    if runtime.get("learnedHead") is not None:
        from .head_checkpoint import trained_runtime

        expected = trained_runtime(expected, Path(runtime["learnedHead"]["manifestPath"]))
    elif runtime.get("checkpointKind") not in {None, "prepared-upstream-model"}:
        raise ValueError("Trained native runtime is missing its learned artifact")
    for key, value in expected.items():
        if runtime.get(key) != value:
            raise ValueError(f"Recorded native runtime provenance differs: {key}")
    if family == "nanojev":
        source, revision = verify_nano_source(Path(runtime["sourceRoot"]), pins)
        if str(source) != runtime["sourceRoot"] or revision != runtime.get("sourceCommit"):
            raise ValueError("Recorded NanoJev source differs")
    if (
        runtime.get("protocol") != "native-choice-jsonl-v1"
        or runtime.get("modelWeightsLoaded") is not True
        or runtime.get("parameterStorage") != "float32"
        or runtime.get("forwardAutocast") != "bfloat16"
        or type(runtime.get("maxLen")) is not int
        or runtime["maxLen"] <= 0
    ):
        raise ValueError("Recorded native execution settings differ")
    return manifest


def make_measurer(runtime):
    """Call only native packing APIs: no predictor construction or model loading."""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer

    directory = Path(runtime["path"])
    if runtime["family"] == "nanojev":
        source = Path(runtime["sourceRoot"])
        sys.path.insert(0, str(source))
        upstream = load_module(source / "scripts/predict_toy_decisions.py", "ra2_nano_encoding_verify")
        tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
        predictor = SimpleNamespace(tokenizer=tokenizer, limit=runtime["maxLen"])
        return NanoJevChoices(predictor, upstream.prepare_examples)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return JevOmniChoices(None, None, tokenizer, None, max_length=runtime["maxLen"])
