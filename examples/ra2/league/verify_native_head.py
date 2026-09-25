"""Reload a learned native head in a fresh process and reproduce saved logits."""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from integration.models.worker import digest, gpu_lock, load_native, read_json, verify_nano_source, verify_prepared
from train_native_head import (
    ROOT,
    body_state,
    checked_dataset,
    forward_logits,
    freeze_body_and_select_head,
    head_state,
    load_head_artifact,
    official_equivalence,
    prepare_row,
    tensor_set_digest,
)


def verify(directory, gpu):
    import torch

    manifest = read_json((directory / "checkpoint.json").read_text())
    local = read_json((directory / "verification.json").read_text())
    family = manifest["family"]
    if (
        not local["verified"]
        or local["checkpointSha256"] != digest(directory / "checkpoint.json")
        or (directory / "error.json").exists()
    ):
        raise ValueError("Require a completed native gradient and reload diagnostic")
    for name, sha in manifest["sourceSha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Native training source changed")
    if digest(directory / "final-probes.json") != manifest["finalProbesSha256"]:
        raise ValueError("Stored native logits changed")
    pins = read_json((ROOT / "league/model-sources.json").read_text())
    parent, runtime = verify_prepared(Path(manifest["parent"]["preparedManifestPath"]), family, pins)
    if runtime != manifest["parent"]:
        raise ValueError("Different native parent")
    data_path = Path(manifest["dataManifestPath"])
    if (
        digest(data_path) != manifest["dataManifestSha256"]
        or digest(data_path.parent / "verification.json") != manifest["dataVerificationSha256"]
    ):
        raise ValueError("Native training data proof changed")
    data, rows = checked_dataset(data_path.parent, family, runtime)
    lookup = {row["id"]: row for row in rows}
    probes = read_json((directory / "final-probes.json").read_text())
    source = None
    if family == "nanojev":
        source, _ = verify_nano_source(Path(data["nativeRuntimeInputs"]["sourceRoot"]), pins)
    with gpu_lock(gpu) as uuid:
        torch.set_num_threads(4)
        torch.manual_seed(manifest["settings"]["seed"])
        with contextlib.redirect_stdout(sys.stderr):
            adapter = load_native(family, parent, source, data["maxLength"])
        if tensor_set_digest(head_state(adapter, family)) != manifest["initialHeadTensorSha256"]:
            raise ValueError("Fresh upstream head differs from training initialization")
        if tensor_set_digest(body_state(adapter, family)) != manifest["frozenBodyTensorSha256"]:
            raise ValueError("Fresh parent body/buffers differ from frozen training parent")
        load_head_artifact(adapter, family, directory / "checkpoint.json", runtime)
        if tensor_set_digest(head_state(adapter, family)) != manifest["finalHeadTensorSha256"]:
            raise ValueError("Reloaded native tensors differ")
        freeze_body_and_select_head(adapter, family)
        independent = []
        for probe in probes:
            row = lookup[probe["rowId"]]
            if row["id"] not in manifest["selectedTrainingRows"] or row["metadata"]["goldIndex"] != probe["goldIndex"]:
                raise ValueError("Probe was not one of the declared training rows")
            if list(next(iter(row["questions"].values()))["criteria"]) != probe["candidateIds"]:
                raise ValueError("Reload probe option order differs")
            encoded = prepare_row(adapter, family, row, data["maxLength"])
            with torch.no_grad():
                actual = forward_logits(adapter, family, row, encoded).cpu()
            torch.testing.assert_close(actual, torch.tensor(probe["logits"], dtype=torch.float32), rtol=0, atol=0)
            independent.append(official_equivalence(adapter, family, row, encoded))
        if tensor_set_digest(body_state(adapter, family)) != manifest["frozenBodyTensorSha256"]:
            raise ValueError("Reload/inference changed native backbone")
        report = {
            "verified": True,
            "family": family,
            "freshProcessReload": True,
            "savedLogitsBitwiseEqual": True,
            "officialNativePredictionEquivalent": True,
            "probeCount": len(probes),
            "frozenBodyAndBuffersEqualToPinnedParent": True,
            "learnedHeadTensorDigestVerified": True,
            "effectiveWeightSetSha256": manifest["effectiveWeightSetSha256"],
            "checkpointSha256": digest(directory / "checkpoint.json"),
            "verifierSha256": digest(Path(__file__)),
            "gpuUuid": uuid,
            "probeEvidence": independent,
            "actualLearningVerified": True,
            "notGameplayEvidence": True,
            "notMigrationQualification": True,
            "notChampionEvidence": True,
        }
        (directory / "independent-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "probeEvidence"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    verify(args.directory, args.gpu)
