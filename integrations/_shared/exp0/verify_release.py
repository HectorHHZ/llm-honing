"""Validate an extracted Exp 0 v0 release and reproduce its offline scores (Griddly v0 ``verify_release.py``).

  python3 -I -B tools/verify_release.py <bundle>     # <bundle> = .../<game-id>/v0

Standard library only (with ``tools/schema_lite.py`` beside it); needs no API key, no model and no game.
Checks: the file set and every SHA-256 against ``sha256.json``; the evaluator index against its JSON schema;
40 distinct neutral episodes, 10/10/10/10, matching the frozen scored ids and the attempt log; every scored
attempt's manifest, decisions and result against their schemas and against the index (condition, identity,
outcome); the checkpoints against the frozen plan; reset groups in one split each and consistent with the
split manifest; the component evidence of every verified episode against the raw records (harness:
logged request differs from the clean render at the evidence row; controller: selection differs from the
execution there; weights: the recorded probabilities equal the matched clean row's or the re-ask reply's,
differ by more than the tolerance, and come from distinct checkpoints); diagnosis inputs carry only
``episode_id`` and ``text`` and no evaluator metadata; every probe identity check passed; and ``score.py``
run from the saved replies reproduces ``evaluation/scores.json`` exactly. When the bundle declares pre-registered
variant runs (``evaluation/variants.json``, added 2026-11-26), each variant's prompt digest is checked against its
settings and ``score.py`` on the ``read`` replies + the variant's replies must reproduce its ``scores-NAME.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONDITIONS = ("no_injection", "harness", "weights", "controller")
LABEL_FIELDS = (
    "verified_component",
    "annotation_status",
    "injection_layer",
    "clean_request",
    "intended_action",
    "no_injection",
    "fault_family",
    "fault_instance",
    "weightsSha256",
)
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
TOOLS = Path(__file__).resolve().parent


def _schema_lite():
    spec = importlib.util.spec_from_file_location("_exp0_schema_lite", TOOLS / "schema_lite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def same_request(left: Any, right: Any) -> bool:
    """Order-sensitive request equality: the order of options is part of what the model sees."""
    dump = lambda value: json.dumps(value, separators=(",", ":"), ensure_ascii=False)  # noqa: E731
    return left is not None and right is not None and dump(left) == dump(right)


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = math.fsum(values.values())
    return {key: value / total for key, value in values.items()}


def reply_probabilities(raw: str, request: dict[str, Any], question_key: str) -> dict[str, float]:
    reply = json.loads(raw)
    probabilities = reply["answers"][question_key]["probabilities"]
    options = list(request["questions"][question_key]["criteria"])
    check(set(probabilities) == set(options), "a reply's probabilities do not cover the request's options")
    check(
        abs(math.fsum(probabilities.values()) - 1) <= len(options) * 0.5e-4 + 1e-12,
        "a reply's probabilities do not sum to one",
    )
    return normalized({option: probabilities[option] for option in options})


def plan_keys(plan: dict[str, Any]) -> list[str]:
    """The questions of every request (contract 0.2.0: ``question_keys`` for several, else ``question_key``)."""
    return list(plan["question_keys"]) if plan.get("question_keys") else [plan["question_key"]]


def keyed(value: Any, keys: list[str]) -> Any:
    return value if len(keys) > 1 else {keys[0]: value}


def fault_instances(plan: dict[str, Any]) -> set[str]:
    out = set()
    for value in plan["faults"].values():
        for fault in (value if isinstance(value, list) else [value]):
            out.add(fault["instance"])
    return out


def verify(root: Path) -> dict[str, Any]:
    lite = _schema_lite()
    hashes = load(root / "sha256.json")
    actual = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in root.rglob("*")
        if path.is_file() and path != root / "sha256.json"
    }
    check(hashes == actual, "file checksums or file set differ")
    evaluator, evaluation = root / "evaluator", root / "evaluation"
    index = load(evaluator / "episodes.json")
    dataset = load(root / "dataset.json")
    plan = load(root / "collection.json")
    release = load(root / "release-manifest.json")
    checkpoints = load(root / "checkpoint-references.json")
    split = load(root / "split-manifest.json")
    schemas = {
        name: lite.load_schema(root / "schemas" / f"{name}.schema.json")
        for name in ("episode-index", "manifest", "decision", "result", "collection", "attempt")
    }
    check(
        release["schema"] == "honing-exp0-release-v1"
        and re.fullmatch(r"[0-9a-f]{40}", release["code_git_commit"]) is not None,
        "invalid release code revision",
    )
    check(
        release["hf_directory"] == plan["dataset_dir"]
        and root.as_posix().rstrip("/").endswith(plan["dataset_dir"].rstrip("/")),
        "the bundle directory differs from the plan's dataset directory",
    )
    check(not lite.errors(plan, schemas["collection"]), "the collection plan violates its schema")
    problems = lite.errors(index, schemas["episode-index"])
    check(not problems, f"the evaluator index violates its schema: {problems[:3]}")
    keys = plan_keys(plan)
    tolerance = plan["weight_difference_tolerance"]
    per = plan["per_condition"]
    dry_run = bool(plan.get("dry_run"))
    check(
        per == 10 or (dry_run and release.get("dry_run") is True and dataset.get("dry_run") is True),
        "a v0 release has 10 episodes per condition (only a marked dry run may hold fewer)",
    )
    ids = {row["episode_id"] for row in index}
    check(len(index) == len(ids) == dataset["episode_count"] == 4 * per, f"expected {4 * per} distinct episodes")
    check(ids == set(dataset["scored_episode_ids"]), "the episode index differs from the frozen scoring ids")
    check(
        Counter(row["condition"] for row in index) == Counter(dict.fromkeys(CONDITIONS, per)),
        f"expected {per}/{per}/{per}/{per}",
    )
    pattern = re.compile(rf"{plan['neutral_id_prefix']}[0-9]{{3}}")
    check(all(pattern.fullmatch(row["episode_id"]) for row in index), "episode ids are not neutral")
    attempts = rows(evaluator / "attempts.jsonl")
    check(all(not lite.errors(item, schemas["attempt"]) for item in attempts), "an attempt record violates its schema")
    check(len(attempts) == dataset["attempts"], "the attempt count differs from the dataset report")
    check(
        {item["run"] for item in attempts if item["selected_for_scoring"]} == {row["run"] for row in index},
        "the selected episodes differ from the attempt log",
    )
    check(
        checkpoints["reference"]["weights_sha256"] == plan["clean_weights_sha256"]
        and checkpoints["alternate"]["weights_sha256"] == plan["alternate_weights_sha256"]
        and plan["clean_weights_sha256"] != plan["alternate_weights_sha256"],
        "checkpoint references differ from the plan",
    )
    plan_splits = {group["id"]: group["split"] for group in plan["groups"]}
    reask = (
        {item["run"]: item for item in rows(evaluator / "weights-reask.jsonl")}
        if (evaluator / "weights-reask.jsonl").is_file()
        else {}
    )
    grouped: dict[str, list[str]] = defaultdict(list)
    decisions_checked = 0
    for row in index:
        episode = row["episode_id"]
        run = (evaluator / row["run"]).resolve()
        check(run.parent == evaluator.resolve(), f"invalid episode path for {episode}")
        manifest, result = load(run / "manifest.json"), load(run / "result.json")
        trace = rows(run / "decisions.jsonl")
        check(not (run / "error.json").exists(), f"{episode} is an incomplete attempt")
        check(
            not lite.errors(manifest, schemas["manifest"]) and not lite.errors(result, schemas["result"]),
            f"{episode}: manifest or result violates its schema",
        )
        check(
            all(not lite.errors(item, schemas["decision"]) for item in trace),
            f"{episode}: a decision row violates its schema",
        )
        decisions_checked += len(trace)
        condition = row["condition"]
        check(row["injection_layer"] == manifest["condition"] == condition, f"injection differs for {episode}")
        identity = row["identity"]
        check(
            identity["weights_sha256"] == manifest["pins"]["checkpoint"]["weightsSha256"]
            and identity["scenario"] == manifest["scenario"]
            and identity["harness"] == manifest["pins"]["harness"]
            and identity["controller"] == manifest["pins"]["controller"]
            and identity["game_build"] == manifest["pins"]["game_build"],
            f"identity differs for {episode}",
        )
        expected_sha = plan["alternate_weights_sha256" if condition == "weights" else "clean_weights_sha256"]
        check(
            manifest["pins"]["checkpoint"]["weightsSha256"]
            == expected_sha
            == result["checkpoint_end"]["weightsSha256"],
            f"{episode} was not served by the plan's checkpoint for its condition",
        )
        check(
            row["native_outcome"] == {"status": result["status"], "reason": result["reason"]}
            and row["replay"]["status"] == result["status"]
            and result["reason"] in plan["outcomes"][result["status"]],
            f"native outcome differs for {episode}",
        )
        check(
            len(trace) == result["rows"] and [item["index"] for item in trace] == list(range(len(trace))),
            f"incomplete decision trace for {episode}",
        )
        check(
            plan_splits.get(row["scenario_group"]) == row["split"] == split["groups"][row["scenario_group"]]["split"],
            f"incorrect split for {episode}",
        )
        grouped[row["scenario_group"]].append(episode)
        if condition == "no_injection":
            check(
                row["annotation_status"] == "control"
                and "verified_component" not in row
                and all(item["fault"] is None and item["clean_request"] is None for item in trace),
                f"invalid control annotation for {episode}",
            )
            continue
        check(
            row["annotation_status"] == "verified" and row["verified_component"] == condition,
            f"missing verified label for {episode}",
        )
        evidence = row["evidence"]
        step = trace[evidence["index"]]
        check(step["native_before"]["tick"] == evidence["native_tick"], f"the evidence tick differs for {episode}")
        if condition == "harness":
            check(
                step["clean_request"] is not None
                and not same_request(step["request"], step["clean_request"])
                and evidence["method"] == "request_vs_clean_render",
                f"missing harness evidence for {episode}",
            )
        elif condition == "controller":
            mismatches = [w for w in step["executed"]["windows"] if "intended_action" in w or "intended_ticks" in w]
            check(
                evidence["method"] == "selected_vs_executed"
                and (step["executed"]["option"] != step["model_choice"] or mismatches)
                and evidence["selected"] == step["model_choice"]
                and evidence["executed"] == step["executed"]["option"]
                and evidence["window_mismatches"] == len(mismatches),
                f"missing controller evidence for {episode}",
            )
        else:
            alternate = {key: normalized(value) for key, value in keyed(step["probabilities"], keys).items()}
            if evidence["method"] == "matched_clean_identical_request":
                clean_run = (evaluator / evidence["clean_run"]).resolve()
                check(clean_run.parent == evaluator.resolve(), f"invalid matched clean path for {episode}")
                other = rows(clean_run / "decisions.jsonl")[evidence["clean_index"]]
                clean = {key: normalized(value) for key, value in keyed(other["probabilities"], keys).items()}
                check(
                    same_request(other["request"], step["request"])
                    and other["native_before"]["tick"] == step["native_before"]["tick"]
                    and load(clean_run / "manifest.json")["pins"]["checkpoint"]["weightsSha256"]
                    == plan["clean_weights_sha256"],
                    f"the matched clean request differs for {episode}",
                )
            else:
                check(
                    evidence["method"] == "same_request_reask" and row["run"] in reask,
                    f"missing re-ask record for {episode}",
                )
                record = reask[row["run"]]
                item = next(entry for entry in record["rows"] if entry["index"] == evidence["index"])
                check(
                    same_request(item["request"], step["request"])
                    and record["clean_checkpoint"]["weightsSha256"] == plan["clean_weights_sha256"],
                    f"the re-ask record differs for {episode}",
                )
                clean = {key: reply_probabilities(item["clean_reply_raw"], step["request"], key) for key in keys}
            check(
                clean == keyed(evidence["clean_probabilities"], keys)
                and alternate == keyed(evidence["alternate_probabilities"], keys)
                and max(abs(clean[k][o] - alternate[k][o]) for k in clean for o in clean[k]) > tolerance
                and evidence["clean_checkpoint"] != evidence["alternate_checkpoint"],
                f"missing weights evidence for {episode}",
            )
    check(
        dict(grouped) == {group: value["episode_ids"] for group, value in split["groups"].items()}
        and len(grouped) == dataset["scenario_reset_groups"],
        "split group coverage differs",
    )
    markers = set(LABEL_FIELDS) | {row["run"] for row in index} | fault_instances(plan)
    markers |= {
        row["identity"]["checkpoint"]["repo"] for row in index if (row["identity"].get("checkpoint") or {}).get("repo")
    }
    for name in ("read-inputs.jsonl", "probe-inputs.jsonl"):
        inputs = rows(evaluation / name)
        check(len(inputs) == len(ids) and {item["episode_id"] for item in inputs} == ids, f"incomplete {name}")
        for item in inputs:
            check(
                set(item) == {"episode_id", "text"}
                and not HEX64.search(item["text"])
                and not any(marker in item["text"] for marker in markers),
                f"evaluator metadata appeared in {name}",
            )
    identities = rows(evaluation / "identity.jsonl")
    check(
        len(identities) == len(ids)
        and {item["episode_id"] for item in identities} == ids
        and all(item["ok"] and item["max_difference"] <= item["tolerance"] for item in identities),
        "probe identity checks failed",
    )
    reproduced = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                "-B",
                str(TOOLS / "score.py"),
                str(evaluator / "episodes.json"),
                str(evaluation / "read" / "diagnoses.jsonl"),
                str(evaluation / "probe" / "diagnoses.jsonl"),
            ]
        )
    )
    check(reproduced == load(evaluation / "scores.json"), "offline scores differ")
    variants = load(evaluation / "variants.json") if (evaluation / "variants.json").is_file() else []
    for item in variants:
        name = item["name"]
        check(
            re.fullmatch(r"[a-z][a-z0-9-]{1,40}", name) is not None
            and name not in ("read", "probe")
            and item["diagnoses"] == f"evaluation/{name}/diagnoses.jsonl"
            and item["scores"] == f"evaluation/scores-{name}.json",
            f"invalid variant record {name!r}",
        )
        settings = load(evaluation / name / "settings.json")
        check(
            settings["prompt_sha256"] == item["prompt_sha256"] == file_digest(root / item["prompt_file"])
            and "/" not in item["prompt_file"],
            f"the prompt of variant {name} differs from its frozen digest",
        )
        replies = rows(evaluation / name / "diagnoses.jsonl")
        check(
            {reply["episode_id"] for reply in replies} == ids and len(replies) == len(ids),
            f"incomplete variant {name}",
        )
        again = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(TOOLS / "score.py"),
                    str(evaluator / "episodes.json"),
                    str(evaluation / "read" / "diagnoses.jsonl"),
                    str(evaluation / name / "diagnoses.jsonl"),
                ]
            )
        )
        check(again == load(evaluation / f"scores-{name}.json"), f"offline scores of variant {name} differ")
    return {
        "verified_files": len(hashes),
        "episodes": len(ids),
        "reset_groups": len(grouped),
        "attempts": len(attempts),
        "decision_rows_checked": decisions_checked,
        "diagnoses": 2 * len(ids),
        **({"variant_diagnoses": {item["name"]: len(ids) for item in variants}} if variants else {}),
        "identity_decisions": sum(item["decisions"] for item in identities),
        "offline_scores_match": True,
        "dry_run": dry_run,
        "questions_per_decision": len(keys),
        "code_git_commit": release["code_git_commit"],
        "python": sys.version.split()[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    print(json.dumps(verify(parser.parse_args().bundle.resolve()), sort_keys=True))
