# ruff: noqa: E402
# Standalone example imports follow its explicit sys.path bootstrap.
"""Export approved normal RA2 labels with the native model's lossless encoding.

This prepares supervision, not weights. Native split `dev` is classifier
validation using TRAIN-protocol episodes only; it is not game-development or
sealed-test data. No game outcome or privileged result field becomes input.
"""

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.models.choice_protocol import pack_request
from integration.models.provenance import make_measurer
from integration.models.worker import digest, read_json, verify_nano_source, verify_prepared


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_sources(data_check, protocol):
    approval = read_json(data_check.read_text())
    if approval.get("passed") is not True or approval.get("developmentOrHeldoutUsed") is not False:
        raise ValueError("Require an approved training-only source registry")
    sources = approval["trainingSources"]
    if not sources or len({x["path"] for x in sources}) != len(sources):
        raise ValueError("Training sources must be nonempty and unique")
    manifests = []
    for source in sources:
        directory = Path(source["path"]).resolve(strict=True)
        for file, key in [("samples.jsonl", "samplesSha256"), ("manifest.json", "manifestSha256")]:
            if digest(directory / file) != source[key]:
                raise ValueError("Approved training data changed: " + str(directory / file))
        manifest = read_json((directory / "manifest.json").read_text())
        result = read_json((directory / "result.json").read_text())
        if (
            manifest.get("kind") not in {"teacher-demonstration", "dagger-labels"}
            or result.get("kind") != manifest["kind"]
            or result.get("notModelEvaluation") is not True
            or manifest.get("notModelEvaluation") is not True
        ):
            raise ValueError("Expected explicitly labeled demonstrations or corrections")
        if (
            manifest["split"] != "train"
            or manifest["map"] not in protocol["train"]["maps"]
            or manifest["seed"] not in protocol["train"]["seeds"]
        ):
            raise ValueError("Refusing non-training episodes")
        for key in ["map", "seed", "seat"]:
            if source[key] != manifest[key] or result[key] != manifest[key]:
                raise ValueError("Episode provenance disagrees: " + key)
        for file, expected in manifest["sourceSha256"].items():
            relative = Path(file)
            if relative.is_absolute() or ".." in relative.parts or digest(directory / "source" / relative) != expected:
                raise ValueError("Source snapshot mismatch")
        manifests.append((source, directory, manifest))
    return manifests


def validate_row(row, manifest):
    for key in ["split", "map", "seed", "seat"]:
        if row[key] != manifest[key]:
            raise ValueError("Training row does not belong to approved episode")
    if type(row.get("stage")) is not int or row["stage"] < 1 or type(row.get("tick")) is not int or row["tick"] < 0:
        raise ValueError("Invalid staged decision provenance")
    if len(row.get("questions", {})) != 1 or set(row.get("answers", {})) != set(row["questions"]):
        raise ValueError("Expected one staged question with a label")
    for key, question in row["questions"].items():
        if set(row["answers"][key]) != {"choice"} or row["answers"][key]["choice"] not in question["criteria"]:
            raise ValueError("Supervision must name one offered choice")


def encoded_identity(audit, family):
    if len(audit) != 1:
        raise ValueError("One staged request is required")
    value = next(iter(audit.values()))
    if family == "nanojev":
        encoding = value["candidatePathSha256"]
        if len(encoding) != value["options"]:
            raise ValueError("Native paths omitted an option")
    elif family == "jev-omni":
        encoding = value["promptSha256"]
    else:
        raise ValueError("Unsupported native family")
    return hash_value([family, encoding])


def pack_row(row, source, line_number, family, measure, max_length, validation_seed):
    state, audit, trimmed = pack_request(row["parts"], row["questions"], measure, max_length)
    key = next(iter(row["questions"]))
    question = row["questions"][key]
    choice = row["answers"][key]["choice"]
    input_id = encoded_identity(audit, family)
    return {
        "id": hash_value([source["manifestSha256"], line_number, family]),
        "state_id": input_id,
        "family_id": "ra2-normal-staged-decision",
        "split": "dev" if row["seed"] == validation_seed else "train",
        "state": state,
        "questions": row["questions"],
        "gold": {key: choice},
        "gold_label_kind": "hard_gold_unspecified",
        "metadata": {
            "labelMeaning": (
                "Heuristic expert's chosen action; not deterministic truth, " "observed reward, or a model probability"
            ),
            "source_group_id": source["manifestSha256"],
            "sourcePath": source["path"],
            "sourceManifestSha256": source["manifestSha256"],
            "sourceSamplesSha256": source["samplesSha256"],
            "sourceLine": line_number,
            "originalRowSha256": hash_value(row),
            "map": row["map"],
            "seed": row["seed"],
            "seat": row["seat"],
            "tick": row["tick"],
            "stage": row["stage"],
            "normalOnlyApprovedSource": True,
            "protocolSplit": "train",
            "goldIndex": list(question["criteria"]).index(choice),
            "encodingFamily": family,
            "nativeInputSha256": input_id,
            "tokenAudit": audit,
            "trimmed": trimmed,
        },
    }


def separate_encoded(records):
    unique = {"train": {}, "dev": {}}
    removals = []
    for row in records:
        target = unique[row["split"]]
        key = row["state_id"]
        old = target.get(key)
        if old and old["metadata"]["goldIndex"] != row["metadata"]["goldIndex"]:
            raise ValueError("Conflicting labels for identical native inputs")
        if old:
            removals.append(
                {"id": row["id"], "keptId": old["id"], "reason": "duplicate_native_input", "split": row["split"]}
            )
        else:
            target[key] = row
    for key in list(unique["dev"]):
        if key in unique["train"]:
            old, row = unique["train"][key], unique["dev"][key]
            if old["metadata"]["goldIndex"] != row["metadata"]["goldIndex"]:
                raise ValueError("Conflicting labels across native splits")
            removals.append(
                {"id": row["id"], "keptId": old["id"], "reason": "validation_input_seen_in_training", "split": "dev"}
            )
            del unique["dev"][key]
    return {key: list(value.values()) for key, value in unique.items()}, removals


def adapter_for(family, prepared, source_root, max_length):
    pins = read_json((ROOT / "league/model-sources.json").read_text())
    _manifest, runtime = verify_prepared(prepared, family, pins)
    runtime["maxLen"] = max_length
    if family == "nanojev":
        if source_root is None:
            raise ValueError("NanoJev requires its pinned source checkout")
        source, revision = verify_nano_source(source_root, pins)
        runtime.update(sourceRoot=str(source), sourceCommit=revision)
    return make_measurer(runtime), runtime


def export(args):
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    protocol = read_json((ROOT / "league/protocol.json").read_text())
    if args.validation_seed not in protocol["train"]["seeds"]:
        raise ValueError("Classifier validation must use training-protocol seeds")
    sources = verified_sources(args.data_check, protocol)
    adapter, runtime = adapter_for(args.family, args.prepared_manifest, args.source_root, args.max_length)
    args.out.mkdir(parents=True)
    records, counts = [], collections.Counter()
    try:
        for source, directory, manifest in sources:
            with (directory / "samples.jsonl").open() as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = read_json(line)
                    validate_row(row, manifest)
                    records.append(
                        pack_row(
                            row, source, number, args.family, adapter.measure, args.max_length, args.validation_seed
                        )
                    )
                    counts["rows"] += 1
                    if counts["rows"] % 1000 == 0:
                        print(
                            json.dumps({"event": "packed", "family": args.family, "rows": counts["rows"]}), flush=True
                        )
        splits, removals = separate_encoded(records)
        if not splits["train"] or not splits["dev"]:
            raise ValueError("Need nonempty native train and novel classifier validation")
        for name, rows in splits.items():
            with (args.out / f"{name}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (args.out / "removed-inputs.json").write_text(json.dumps(removals, indent=2) + "\n")
        report = {
            "kind": "exported-native-ra2-supervision",
            "family": args.family,
            "dataCheckPath": str(args.data_check.resolve()),
            "dataCheckSha256": digest(args.data_check),
            "protocolSha256": digest(ROOT / "league/protocol.json"),
            "sourceCodeSha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    Path(__file__),
                    ROOT / "integration/models/choice_protocol.py",
                    ROOT / "integration/models/upstream_adapters.py",
                    ROOT / "integration/models/provenance.py",
                    ROOT / "integration/models/worker.py",
                    ROOT / "league/model-sources.json",
                ]
            },
            "nativeRuntimeInputs": runtime,
            "validationSeed": args.validation_seed,
            "maxLength": args.max_length,
            "rawRows": len(records),
            "trainingExamples": len(splits["train"]),
            "validationExamples": len(splits["dev"]),
            "removedInputs": len(removals),
            "removalReasons": dict(collections.Counter(x["reason"] for x in removals)),
            "stageCounts": {
                name: dict(collections.Counter(row["metadata"]["stage"] for row in rows))
                for name, rows in splits.items()
            },
            "trimmedRequests": sum(bool(row["metadata"]["trimmed"]) for row in records),
            "sources": [source for source, _, _ in sources],
            "outputSha256": {
                name: digest(args.out / name) for name in ["train.jsonl", "dev.jsonl", "removed-inputs.json"]
            },
            "normalOnlyApprovedSources": True,
            "modelObjectsLoaded": False,
            "gpuAllocated": False,
            "learningPerformed": False,
            "gameplayEvaluationPerformed": False,
            "devMeaning": (
                "Classifier validation from train-protocol seed only; " "not engine development or heldout games"
            ),
            "noEvaluationOrSealedTrainingData": True,
        }
        (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in [
                        "family",
                        "rawRows",
                        "trainingExamples",
                        "validationExamples",
                        "removedInputs",
                        "trimmedRequests",
                        "learningPerformed",
                    ]
                }
            ),
            flush=True,
        )
        return report
    except BaseException as error:
        (args.out / "error.json").write_text(
            json.dumps({"error": str(error), "packedRows": len(records)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["nanojev", "jev-omni"])
    parser.add_argument("--data-check", required=True, type=Path)
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validation-seed", type=int, default=105)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    export(args)
