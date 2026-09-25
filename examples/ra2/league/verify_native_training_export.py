"""Reconstruct native supervision from approved source rows and check native encoding."""

import argparse
import json
import sys
from pathlib import Path

from export_native_training import (
    ROOT,
    adapter_for,
    digest,
    pack_row,
    read_json,
    separate_encoded,
    validate_row,
    verified_sources,
)
from integration.models.choice_protocol import token_digest
from integration.models.worker import load_module


def verify(directory):
    manifest = read_json((directory / "manifest.json").read_text())
    if (
        manifest["kind"] != "exported-native-ra2-supervision"
        or (directory / "error.json").exists()
        or manifest["learningPerformed"]
    ):
        raise ValueError("Expected a completed data export, not training")
    for name, expected in manifest["outputSha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Exported file changed: " + name)
    for name, expected in manifest["sourceCodeSha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Packing source changed: " + name)
    check = Path(manifest["dataCheckPath"])
    if (
        digest(check) != manifest["dataCheckSha256"]
        or digest(ROOT / "league/protocol.json") != manifest["protocolSha256"]
    ):
        raise ValueError("Approved source registry/protocol changed")
    protocol = read_json((ROOT / "league/protocol.json").read_text())
    sources = verified_sources(check, protocol)
    if [x[0] for x in sources] != manifest["sources"]:
        raise ValueError("Different registered source set")
    runtime = manifest["nativeRuntimeInputs"]
    source = Path(runtime["sourceRoot"]) if "sourceRoot" in runtime else None
    adapter, actual_runtime = adapter_for(
        manifest["family"], Path(runtime["preparedManifestPath"]), source, manifest["maxLength"]
    )
    if runtime != actual_runtime:
        raise ValueError("Different native packing inputs")
    reconstructed = []
    for approved, path, episode in sources:
        with (path / "samples.jsonl").open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = read_json(line)
                validate_row(row, episode)
                reconstructed.append(
                    pack_row(
                        row,
                        approved,
                        number,
                        manifest["family"],
                        adapter.measure,
                        manifest["maxLength"],
                        manifest["validationSeed"],
                    )
                )
    expected, removals = separate_encoded(reconstructed)
    if (
        len(reconstructed) != manifest["rawRows"]
        or read_json((directory / "removed-inputs.json").read_text()) != removals
    ):
        raise ValueError("Input coverage/deduplication audit mismatch")
    counts = {}
    for split in ["train", "dev"]:
        with (directory / f"{split}.jsonl").open() as stream:
            actual = [read_json(line) for line in stream if line.strip()]
        if actual != expected[split]:
            raise ValueError("Export differs from original approved normal inputs and labels")
        counts[split] = len(actual)
    independently_encoded = 0
    if manifest["family"] == "nanojev":
        sys.path.insert(0, str(source / "scripts"))
        upstream = load_module(source / "scripts/train_pipeline_decisions.py", "ra2_native_training_export_verifier")
        # This upstream function prepares examples only; it constructs no model.
        examples, audit = upstream.load_training_examples(
            directory, adapter.predictor.tokenizer, manifest["maxLength"]
        )
        for example in examples:
            row = example["source"]
            key = example["qid"]
            if (
                example["candidate_ids"] != list(row["questions"][key]["criteria"])
                or example["gold_index"] != row["metadata"]["goldIndex"]
            ):
                raise ValueError("Official trainer changed option order or label")
            paths = [token_digest(ids) for ids in example["leaf_tokens"]]
            if paths != row["metadata"]["tokenAudit"][key]["candidatePathSha256"]:
                raise ValueError("Official trainer encoding differs from inference packing")
            if max(map(len, example["leaf_tokens"])) > manifest["maxLength"]:
                raise ValueError("Native trainer exceeded declared input budget")
            independently_encoded += 1
    else:
        for rows in expected.values():
            for row in rows:
                measured = adapter.measure(row["state"], row["questions"])
                for key, audit in row["metadata"]["tokenAudit"].items():
                    if (
                        measured[key]["promptSha256"] != audit["promptSha256"]
                        or measured[key]["inputTokens"] != audit["inputTokens"]
                    ):
                        raise ValueError("Native numbered-choice prompt mismatch")
                independently_encoded += 1
    if (
        counts["train"] != manifest["trainingExamples"]
        or counts["dev"] != manifest["validationExamples"]
        or independently_encoded != sum(counts.values())
    ):
        raise ValueError("Output record counts disagree")
    report = {
        "verified": True,
        "family": manifest["family"],
        "manifestSha256": digest(directory / "manifest.json"),
        "verifierSha256": digest(Path(__file__)),
        "trainingExamples": counts["train"],
        "validationExamples": counts["dev"],
        "everyApprovedInputReconstructed": True,
        "retainedNativeEncodingsRecomputed": independently_encoded,
        "nanoOfficialTrainingPreparerUsed": manifest["family"] == "nanojev",
        "essentialAndOptionsPreserved": True,
        "nativeInputSplitsDisjoint": True,
        "classifierValidationFromTrainProtocolOnly": True,
        "modelObjectsLoaded": False,
        "gpuAllocated": False,
        "learningPerformed": False,
        "notGameplayEvidence": True,
    }
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    verify(args.directory.resolve())
