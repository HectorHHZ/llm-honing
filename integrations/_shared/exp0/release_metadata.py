"""Publication metadata for an Exp 0 v0 release, added without changing the frozen inputs or labels (Griddly v0).

Called by ``bundle.py release``. Writes into the bundle: ``split-manifest.json`` (reset groups, their split
and episode ids, unused reserve groups), ``checkpoint-references.json`` (clean and alternate checkpoints:
repo, revision, weights SHA-256 from the services' health records, optional artifact URI and per-file
SHA-256 of a local checkpoint directory), ``evaluation/weight-probes.json`` (alternate-vs-reference
probability changes on the probed requests, by collection condition, reported separately from diagnosis
scores) and ``release-manifest.json`` (code revision, dataset location, pins of the shared code).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream
from contract import (
    CONTRACT_VERSION,
    guard_decision,
    guard_response,
    multi,
    normalized,
    plan_question_keys,
    read_json,
    read_rows,
    same_request,
    sha256_file,
    write_json,
)

LOCAL_FIELDS = ("path", "gpuUuid")


def checkpoint_entry(
    plan_sha: str, manifests: list[dict[str, Any]], directory: Path | None, uri: str | None
) -> dict[str, Any]:
    serving = [m["pins"]["checkpoint"] for m in manifests if m["pins"]["checkpoint"]["weightsSha256"] == plan_sha]
    if not serving:
        raise ValueError(f"no scored episode was served by checkpoint {plan_sha}")
    portable = {key: value for key, value in serving[0].items() if key not in LOCAL_FIELDS}
    if uri is None and portable.get("repo") and portable.get("revision"):
        # Referenced, not re-uploaded: the public Hugging Face repo at its pinned revision.
        uri = f"https://huggingface.co/{portable['repo']}/tree/{portable['revision']}"
    entry: dict[str, Any] = {
        "weights_sha256": plan_sha,
        "repo": portable.get("repo"),
        "revision": portable.get("revision"),
        "health_metadata": portable,
        "artifact_uri": uri,
        "files_sha256": None,
    }
    if directory is not None:
        files = {
            path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        if plan_sha not in files.values():
            raise ValueError(f"the checkpoint directory {directory} holds no file with SHA-256 {plan_sha}")
        entry["files_sha256"] = files
    return entry


def _answers(request: dict[str, Any], raw: str, keys: list[str]) -> dict[str, dict[str, float]]:
    if multi(keys):
        return {key: value["normalized"] for key, value in guard_decision(request, raw, keys)["questions"].items()}
    return {keys[0]: normalized(guard_response(request, raw, keys[0])["probabilities"])}


def weight_probe_summary(
    evaluation: Path, index: list[dict[str, Any]], tolerance: float, question_key
) -> dict[str, Any]:
    """Alternate-vs-reference changes on the probed requests. ``question_key``: the plan's question or (0.2.0) the
    list of its questions; with several questions a decision counts once, by its largest change over questions,
    and "most probable option changed" means in any question."""
    keys = [question_key] if isinstance(question_key, str) else list(question_key)
    reference = {row["episode_id"]: row["rows"] for row in read_rows(evaluation / "probe-reference.jsonl")}
    alternate = {row["episode_id"]: row["rows"] for row in read_rows(evaluation / "probe-alternate.jsonl")}
    effects: dict[str, list[float]] = {
        condition: [] for condition in ("no_injection", "harness", "weights", "controller")
    }
    changed_choice: dict[str, int] = dict.fromkeys(effects, 0)
    for row in index:
        for clean, other in zip(reference[row["episode_id"]], alternate[row["episode_id"]], strict=True):
            if not same_request(clean["logged"], other["logged"]):
                raise ValueError("weight probes must compare the same request")
            left = _answers(clean["logged"], clean["logged_reply_raw"], keys)
            right = _answers(other["logged"], other["logged_reply_raw"], keys)
            effects[row["condition"]].append(max(abs(left[k][o] - right[k][o]) for k in keys for o in left[k]))
            changed_choice[row["condition"]] += any(
                max(left[k], key=left[k].get) != max(right[k], key=right[k].get) for k in keys
            )
    return {
        "tolerance": tolerance,
        "by_collection_condition": {
            condition: {
                "decisions": len(values),
                "differences_above_tolerance": sum(value > tolerance for value in values),
                "most_probable_option_changed": changed_choice[condition],
                "mean_max_option_difference": math.fsum(values) / len(values) if values else None,
                "max_option_difference": max(values) if values else None,
            }
            for condition, values in effects.items()
        },
    }


def split_manifest(plan: dict[str, Any], index: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for row in index:
        group = groups.setdefault(row["scenario_group"], {"split": row["split"], "episode_ids": []})
        if group["split"] != row["split"]:
            raise ValueError("a reset group spans more than one split")
        group["episode_ids"].append(row["episode_id"])
    plan_splits = {group["id"]: group["split"] for group in plan["groups"]}
    for name, group in groups.items():
        if plan_splits.get(name) != group["split"]:
            raise ValueError(f"reset group {name} has another split than in the plan")
    return {
        "groups": {name: groups[name] for name in sorted(groups)},
        "plan_group_splits": plan_splits,
        "unused_plan_groups": sorted(set(plan_splits) - set(groups)),
        "rule": "all attempts of one reset group (clean and changed) share its split; a group is one statistical unit",
    }


def add_release_metadata(
    out: Path,
    index: list[dict[str, Any]],
    code_revision: str,
    checkpoint_dirs: dict[str, Path | None],
    artifact_uris: dict[str, str | None],
) -> None:
    plan = read_json(out / "collection.json")
    evaluator = out / "evaluator"
    manifests = [read_json(evaluator / row["run"] / "manifest.json") for row in index]
    write_json(out / "split-manifest.json", split_manifest(plan, index))
    write_json(
        out / "checkpoint-references.json",
        {
            "schema": "honing-exp0-checkpoint-reference-v1",
            "reference": checkpoint_entry(
                plan["clean_weights_sha256"], manifests, checkpoint_dirs.get("clean"), artifact_uris.get("clean")
            ),
            "alternate": checkpoint_entry(
                plan["alternate_weights_sha256"],
                manifests,
                checkpoint_dirs.get("alternate"),
                artifact_uris.get("alternate"),
            ),
        },
    )
    write_json(
        out / "evaluation" / "weight-probes.json",
        weight_probe_summary(
            out / "evaluation",
            index,
            plan["weight_difference_tolerance"],
            plan_question_keys(plan) if multi(plan_question_keys(plan)) else plan_question_keys(plan)[0],
        ),
    )
    write_json(
        out / "release-manifest.json",
        {
            "schema": "honing-exp0-release-v1",
            "game_id": plan["game_id"],
            "code_git_commit": code_revision,
            "hf_dataset": "honinglab/exp0-data",
            "hf_directory": plan["dataset_dir"],
            "hf_revision": None,
            "hf_revision_note": "filled in the delivery receipt after the user-approved upload; nothing is uploaded by these tools",
            "contract_version": CONTRACT_VERSION,
            "dry_run": bool(plan.get("dry_run")),
            "shared_code": {
                "upstream_main": upstream.UPSTREAM_SHA,
                "probe_identity_revision": upstream.PROBE_REV,
                "probe_identity_blob": upstream.PROBE_BLOB,
            },
        },
    )
