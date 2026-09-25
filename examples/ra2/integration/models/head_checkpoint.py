"""Explicit learned native head artifacts layered on a verified upstream parent.

This format changes only the decision head. It is not an upstream checkpoint,
full-backbone training, or evidence of improved gameplay.
"""

import hashlib
import json
from pathlib import Path

from .worker import digest, read_json

NANO_HEAD_PREFIXES = ("norm.", "scalar.", "set_project.", "set_attention.", "set_output.")


def head_module(adapter, family):
    if family == "nanojev":
        return adapter.predictor.model
    if family == "jev-omni":
        return adapter.head
    raise ValueError("Unsupported native family")


def allowed_head_key(key, family):
    if family == "nanojev":
        return key.startswith(NANO_HEAD_PREFIXES)
    if family == "jev-omni":
        return key in {"mu", "sd", "linear.weight", "linear.bias"}
    return False


def head_state(adapter, family):
    return {k: v for k, v in head_module(adapter, family).state_dict().items() if allowed_head_key(k, family)}


def apply_head_state(adapter, family, state):
    import torch

    target = head_state(adapter, family)
    if not target or set(state) != set(target):
        raise ValueError("Learned head must contain exactly the native head tensors")
    for key, value in state.items():
        original = target[key]
        if value.shape != original.shape or value.dtype != original.dtype or not torch.isfinite(value).all():
            raise ValueError("Learned head shape, dtype or values differ: " + key)
        if family == "jev-omni" and key in {"mu", "sd"} and not torch.equal(value.cpu(), original.cpu()):
            raise ValueError("Head-only learning must preserve native normalization buffers")
    module = head_module(adapter, family)
    all_state = module.state_dict()
    all_state.update(state)
    module.load_state_dict(all_state, strict=True)


def freeze_body_and_select_head(adapter, family):
    module = head_module(adapter, family)
    if family == "jev-omni":
        adapter.model.requires_grad_(False)
        adapter.model.eval()
    module.requires_grad_(False)
    selected = {}
    for name, parameter in module.named_parameters():
        if allowed_head_key(name, family):
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError("No trainable native head parameters")
    module.eval()  # No dropout or running-statistics updates in this diagnostic.
    return selected


def effective_weight_digest(base_digest, head_digest):
    value = {"baseWeightSetSha256": base_digest, "nativeHeadSha256": head_digest}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_head_artifact(path, family, parent_runtime):
    if (Path(path).resolve().parent / "error.json").exists():
        raise ValueError("Refusing a native head from a failed training run")
    manifest = read_json(Path(path).read_text())
    if manifest.get("kind") != "trained-native-decision-head" or manifest.get("family") != family:
        raise ValueError("Expected an explicitly trained head for this native family")
    parent = manifest["parent"]
    for key in ["weightSetSha256", "preparedManifestSha256", "preparedManifestPath"]:
        if parent[key] != parent_runtime[key]:
            raise ValueError("Learned head belongs to another native parent: " + key)
    if manifest["weightFilename"] != "head.safetensors":
        raise ValueError("Unexpected native head filename")
    weights = Path(path).resolve().parent / manifest["weightFilename"]
    if digest(weights) != manifest["headSha256"]:
        raise ValueError("Learned native head was modified")
    if (
        effective_weight_digest(parent["weightSetSha256"], manifest["headSha256"])
        != manifest["effectiveWeightSetSha256"]
    ):
        raise ValueError("Composite native weight identity differs")
    if manifest["trainingScope"] != "native-head-only" or manifest["backboneUpdated"] is not False:
        raise ValueError("Unsupported native training scope")
    return manifest, weights


def load_head_artifact(adapter, family, path, parent_runtime):
    from safetensors.torch import load_file

    manifest, weights = verify_head_artifact(path, family, parent_runtime)
    apply_head_state(adapter, family, load_file(str(weights), device="cpu"))
    return manifest


def trained_runtime(parent_runtime, path):
    """Expose learned weights as a new identity; retain exact upstream lineage."""
    path = Path(path).resolve(strict=True)
    manifest, _ = verify_head_artifact(path, parent_runtime["family"], parent_runtime)
    proof_path = path.parent / "independent-verification.json"
    proof = read_json(proof_path.read_text())
    if (
        proof.get("verified") is not True
        or proof.get("checkpointSha256") != digest(path)
        or proof.get("effectiveWeightSetSha256") != manifest["effectiveWeightSetSha256"]
        or proof.get("family") != parent_runtime["family"]
        or any(
            proof.get(key) is not True
            for key in [
                "freshProcessReload",
                "savedLogitsBitwiseEqual",
                "officialNativePredictionEquivalent",
                "frozenBodyAndBuffersEqualToPinnedParent",
                "learnedHeadTensorDigestVerified",
                "actualLearningVerified",
            ]
        )
        or type(manifest.get("actualOptimizerSteps")) is not int
        or manifest["actualOptimizerSteps"] < 1
    ):
        raise ValueError("Learned native head requires independent gradient and reload evidence")
    return {
        **parent_runtime,
        "checkpointKind": "trained-native-decision-head",
        "baseModel": {key: parent_runtime[key] for key in ["repo", "revision", "weightSetSha256"]},
        "repo": "local:ra2/trained-native-decision-head",
        "revision": manifest["effectiveWeightSetSha256"],
        "weightSetSha256": manifest["effectiveWeightSetSha256"],
        "weightDigestDefinition": "sha256 of sorted compact JSON containing baseWeightSetSha256 and nativeHeadSha256",
        "learnedHead": {
            "manifestPath": str(path),
            "manifestSha256": digest(path),
            "independentVerificationSha256": digest(proof_path),
            "headSha256": manifest["headSha256"],
            "trainingScope": manifest["trainingScope"],
            "actualOptimizerSteps": manifest["actualOptimizerSteps"],
        },
    }
