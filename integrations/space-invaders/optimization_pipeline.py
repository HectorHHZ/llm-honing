"""Restartable, evidence-first optimization loop for the Space Invaders lane controller.

This is an integration driver. It does not change Campaign or release formats.
Game execution and model training are explicit commands whose outputs are checked
before a Honing Gate can see them. Run ``--help`` for the phase commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.core import (
    ArchitectureSpec,
    CheckpointRef,
    EvaluationResult,
    HarnessFile,
    HarnessSpec,
    HoneRelease,
    ModelSnapshot,
    UpdateCandidate,
)
from honing.release import Gate
from honing.release.selectors import NoRegressionSelector, WinLossSelector

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CODE_NAMES = ("lane_pilot.py", "lane_controller.py", "solo_intercept.py", "lane_io.py", "random_ab_server.py")
SOURCE_FILES = (
    "integrations/space-invaders/optimization_pipeline.py",
    "integrations/space-invaders/pipeline_game_runner.py",
    "integrations/space-invaders/pipeline_sft_adapter.py",
    "integrations/space-invaders/pipeline_bundle.py",
    "integrations/space-invaders/run_counterfactual_sft_gpu4.sh",
    "integrations/space-invaders/train_rollout_pair_sft.py",
    *(str(path.relative_to(REPOSITORY)) for path in sorted((REPOSITORY / "honing").rglob("*.py"))),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x") as output:
            output.write(data)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError(f"Immutable artifact already differs: {path}") from None


def require_stage(run: Path, stage: str) -> dict[str, Any]:
    manifest = read(run / "manifest.json")
    if manifest["stage"] != stage:
        raise ValueError(f"Expected stage {stage}, found {manifest['stage']}")
    return manifest


def advance(run: Path, manifest: dict[str, Any], stage: str, **fields: Any) -> None:
    updated = {**manifest, **fields, "stage": stage}
    temporary = run / "manifest.json.tmp"
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    temporary.replace(run / "manifest.json")


def code_hashes(folder: Path) -> dict[str, str]:
    return {name: digest(folder / name) for name in CODE_NAMES}


def check_frozen(run: Path, manifest: dict[str, Any]) -> None:
    if code_hashes(run / "incumbent-code") != manifest["code_sha256"]:
        raise ValueError("Frozen incumbent code changed")
    if digest(run / "config.json") != manifest["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    version = read(run / "pipeline-version.json")["source_sha256"]
    if any(digest(REPOSITORY / name) != expected for name, expected in version.items()):
        raise ValueError("Pipeline source changed after freeze; start a new versioned round")


def freeze(config_path: Path, run: Path) -> None:
    config = read(config_path)
    if config.get("schema") != 1 or config.get("game") != "ALE/SpaceInvaders-v5":
        raise ValueError("Expected the Space Invaders pipeline configuration schema 1")
    if run.exists() and (run / "manifest.json").exists():
        raise ValueError("Run already frozen; resume it with the next phase")
    seeds = config["seeds"]
    partitions = [seeds[name] for name in ("development", "validation", "gate")]
    if any(not group or any(type(seed) is not int or seed < 0 for seed in group) for group in partitions):
        raise ValueError("Each seed partition needs nonnegative integer seeds")
    if len({seed for group in partitions for seed in group}) != sum(map(len, partitions)):
        raise ValueError("Development, validation and gate seeds must be disjoint")
    if len(partitions[2]) < 2:
        raise ValueError("Gate requires at least two paired seeds")
    health = read(Path(config["incumbent"]["health"]))
    weights = config["incumbent"]["weights_sha256"]
    if not health.get("ready") or health["metadata"].get("weightsSha256") != weights:
        raise ValueError("Incumbent health and configured weight hash differ")
    for name in CODE_NAMES:
        if not Path(config["code"][name]).is_file():
            raise ValueError(f"Missing incumbent code file: {name}")
    for item in config["review_inputs"]:
        source = Path(item["path"])
        if item["kind"] not in ("code", "trace", "video"):
            raise ValueError("Reviewer may only receive current code, traces and videos")
        if not source.is_file() or any(
            term in source.name.lower() for term in ("candidate", "gate", "proposal", "analysis")
        ):
            raise ValueError(f"Reviewer input is missing or appears to include withheld material: {source}")
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run / "config.json")
    shutil.copy2(config["incumbent"]["health"], run / "incumbent-health.json")
    code = run / "incumbent-code"
    code.mkdir()
    for name in CODE_NAMES:
        shutil.copy2(config["code"][name], code / name)
    review = run / "review-packet"
    review.mkdir()
    allowed: list[dict[str, str]] = []
    for index, item in enumerate(config["review_inputs"]):
        source = Path(item["path"])
        target = review / f"{index:03d}-{source.name}"
        shutil.copy2(source, target)
        allowed.append({"kind": item["kind"], "file": target.name, "sha256": digest(target)})
    write_once(run / "allowed-inputs.json", {"files": allowed})
    version = {name: digest(REPOSITORY / name) for name in SOURCE_FILES}
    write_once(run / "pipeline-version.json", {"schema": 1, "source_sha256": version})
    ledger = Path(config["gate_seed_ledger"])
    ledger.mkdir(parents=True, exist_ok=True)
    claimed: list[Path] = []
    try:
        for seed in seeds["gate"]:
            marker = ledger / f"space-invaders-{seed}.json"
            with marker.open("x") as output:
                json.dump({"run": str(run.resolve()), "seed": seed}, output)
            claimed.append(marker)
    except FileExistsError as error:
        for marker in claimed:
            marker.unlink()
        shutil.rmtree(run)
        raise ValueError(f"Final seed already reserved or used: {error.filename}") from None
    manifest = {
        "schema": 1,
        "stage": "frozen",
        "game": config["game"],
        "config_sha256": digest(run / "config.json"),
        "code_sha256": code_hashes(code),
        "weights_sha256": weights,
        "incumbent_health_sha256": digest(run / "incumbent-health.json"),
        "environment_sha256": encoded_hash(config["environment"]),
        "seeds": seeds,
        "pipeline_version_sha256": digest(run / "pipeline-version.json"),
    }
    write_once(run / "manifest.json", manifest)


REVIEW_PROMPT = """Independently inspect only the files in this packet. They contain the current game
controller, game trajectories, and replay videos. Identify concrete failure modes with file/step
evidence. Do not propose a candidate implementation. Return a JSON object with a `findings` array;
each finding has `failure`, `evidence`, and `confidence`. You have not been given the main agent's
hypotheses, past candidate conclusions, or final gate results. Do not look outside this packet.
"""


def review(run: Path, agent_command: list[str] | None = None) -> None:
    manifest = require_stage(run, "frozen")
    check_frozen(run, manifest)
    packet = run / "review-packet"
    allowed = read(run / "allowed-inputs.json")["files"]
    if any(digest(packet / item["file"]) != item["sha256"] for item in allowed):
        raise ValueError("Review packet changed after freeze")
    (packet / "REVIEW_PROMPT.txt").write_text(REVIEW_PROMPT)
    schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "failure": {"type": "string"},
                        "evidence": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["failure", "evidence", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    (packet / "review-schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    output = run / "independent-review.json"
    command = agent_command or [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(packet / "review-schema.json"),
        "-C",
        str(packet),
        "-o",
        str(output),
        "-",
    ]
    with (run / "review.stdout").open("w") as stdout, (run / "review.stderr").open("w") as stderr:
        subprocess.run(command, input=REVIEW_PROMPT, text=True, cwd=packet, stdout=stdout, stderr=stderr, check=True)
    report = read(output)
    if not isinstance(report.get("findings"), list):
        raise ValueError("Independent review must have a findings array")
    advance(run, manifest, "reviewed", independent_review_sha256=digest(output))


def register(run: Path, proposal_path: Path) -> None:
    manifest = require_stage(run, "reviewed")
    check_frozen(run, manifest)
    proposal = read(proposal_path)
    if proposal.get("bottleneck") not in ("execution_code", "menu_prompt", "model_ranking"):
        raise ValueError("Classify the bottleneck as execution_code, menu_prompt or model_ranking")
    method = proposal.get("method")
    target = proposal.get("primary_target")
    valid = {
        "manual_code": ("execution_code", CODE_NAMES),
        "meta_harness": ("execution_code", CODE_NAMES),
        "gepa": ("menu_prompt", ("lane_pilot.py",)),
        "sft": ("model_ranking", ("weights",)),
    }
    if method not in valid or proposal["bottleneck"] != valid[method][0] or target not in valid[method][1]:
        raise ValueError("Method, bottleneck and single primary target are inconsistent")
    if method == "meta_harness":
        raise ValueError("Meta-Harness has no Space Invaders method adapter; do not label a code edit as Meta-Harness")
    if method == "gepa":
        from honing.harness.methods.gepa.method import _require_gepa

        _require_gepa()
        raise ValueError(
            "This lane controller keeps its question in a program file; expose it as a HarnessFile "
            "of kind instructions before running real GEPA"
        )
    if not proposal.get("hypothesis") or not proposal.get("metrics") or not proposal.get("controls"):
        raise ValueError("Register hypothesis, metrics and control arms before development")
    if "incumbent" not in proposal["controls"]:
        raise ValueError("The paired incumbent control must be preregistered")
    if type(proposal.get("validation_min_mean_gain")) not in (int, float) or proposal["validation_min_mean_gain"] <= 0:
        raise ValueError("Pre-register a positive validation mean gain threshold")
    budget = proposal.get("budget", {})
    if any(
        type(budget.get(key)) is not int or budget[key] < 0
        for key in ("development_games", "validation_games", "gate_games")
    ):
        raise ValueError("Register nonnegative development, validation and gate game budgets")
    if budget["validation_games"] < 2 * len(manifest["seeds"]["validation"]):
        raise ValueError("Validation budget does not cover paired games")
    if budget["gate_games"] < 2 * len(manifest["seeds"]["gate"]):
        raise ValueError("Gate budget does not cover paired games")
    if method == "sft":
        if not {"incumbent", "random", "base", "code"}.issubset(proposal["controls"]):
            raise ValueError("SFT must preregister random, base and pure-code controls")
        if budget["gate_games"] < 5 * len(manifest["seeds"]["gate"]):
            raise ValueError("SFT Gate budget does not cover all five arms")
    gate = proposal.get("gate", {})
    if (
        gate.get("selector") not in ("no_regression", "win_loss")
        or type(gate.get("min_mean_gain")) not in (int, float)
        or gate["min_mean_gain"] <= 0
    ):
        raise ValueError("Register Gate selector and a positive minimum mean gain")
    shutil.copy2(proposal_path, run / "proposal.json")
    advance(run, manifest, "registered", proposal_sha256=digest(run / "proposal.json"))


def develop(run: Path, candidate_code: Path | None = None, method_command: list[str] | None = None) -> None:
    manifest = require_stage(run, "registered")
    check_frozen(run, manifest)
    proposal = read(run / "proposal.json")
    method = proposal["method"]
    if method == "manual_code":
        if candidate_code is None or method_command is not None:
            raise ValueError("manual_code requires a candidate code folder and no method command")
        shutil.copytree(candidate_code, run / "candidate-code", dirs_exist_ok=True)
        result = {"method": "manual_code", "candidate_code": str(run / "candidate-code")}
    else:
        if candidate_code is not None or not method_command:
            raise ValueError(f"{method} requires a real method command and its result file")
        if method == "sft":
            adapter = (
                len(method_command) >= 2 and Path(method_command[1]).resolve() == HERE / "pipeline_sft_adapter.py"
            )
            if not adapter:
                raise ValueError("SFT must use the pinned UCSD pipeline_sft_adapter.py")
        environment = os.environ.copy()
        environment.update({"HONING_PIPELINE_RUN": str(run.resolve()), "HONING_PIPELINE_SPLIT": "development"})
        with (run / "method.stdout").open("w") as stdout, (run / "method.stderr").open("w") as stderr:
            subprocess.run(method_command, env=environment, stdout=stdout, stderr=stderr, check=True)
        result = read(run / "method-result.json")
        if result.get("method") != method:
            raise ValueError("Method result has wrong provenance")
        if (
            type(result.get("development_games_spent")) is not int
            or not 0 <= result["development_games_spent"] <= proposal["budget"]["development_games"]
        ):
            raise ValueError("Method did not account for development game budget")
        if method == "sft":
            if (
                not result.get("training_metrics")
                or not result.get("training_log")
                or not result.get("checkpoint_manifest")
                or not result.get("weights_sha256")
            ):
                raise ValueError("SFT result requires weights, manifest, training metrics and a saved log")
            config = read(run / "config.json")
            if (
                result.get("training_host") != "ds-serv2.ucsd.edu"
                or result.get("gpu_uuid") != config["remote"]["gpu_uuid"]
            ):
                raise ValueError("SFT training host or GPU UUID differs from the pinned UCSD target")
            if not set(result.get("data_seeds", [])) <= set(manifest["seeds"]["development"]):
                raise ValueError("SFT method used seeds outside the development partition")
            if not Path(result["training_log"]).is_file():
                raise ValueError("Training log was not copied into local evidence")
            if not result.get("training_chart") and not result.get("equivalent_metrics_explanation"):
                raise ValueError("Training requires a chart or an explanation of equivalent metrics")
        source = Path(result.get("candidate_code", run / "incumbent-code"))
        shutil.copytree(source, run / "candidate-code", dirs_exist_ok=True)
    before, after = manifest["code_sha256"], code_hashes(run / "candidate-code")
    changed = [name for name in CODE_NAMES if before[name] != after[name]]
    if method == "sft":
        if changed or result.get("weights_sha256") in (None, manifest["weights_sha256"]):
            raise ValueError("SFT must change weights and keep frozen harness code")
    elif changed != [proposal["primary_target"]]:
        raise ValueError(f"Exactly the registered code file must change; observed {changed}")
    write_once(
        run / "development-result.json",
        {
            "method": method,
            "method_result": result,
            "candidate_code_sha256": after,
            "candidate_weights_sha256": result.get("weights_sha256", manifest["weights_sha256"]),
        },
    )
    advance(run, manifest, "developed", development_result_sha256=digest(run / "development-result.json"))


def _trace_rows(path: Path) -> list[dict[str, Any]]:
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return [json.loads(line) for line in source]


def audit_arm(
    summary: Path,
    trace: Path,
    seeds: list[int],
    expected_code: dict[str, str],
    weights: str | None,
    environment_sha256: str,
) -> dict[str, Any]:
    data = read(summary)
    metadata = data["pipeline_metadata"]
    if metadata["code_sha256"] != expected_code or metadata["weights_sha256"] != weights:
        raise ValueError("Game runner used different harness code or model weights")
    if metadata["environment_sha256"] != environment_sha256:
        raise ValueError("Game environment differs from frozen configuration")
    if weights is not None:
        health_path = summary.parent / "server-health.json"
        health = read(health_path)
        if (
            not health.get("ready")
            or health["metadata"].get("weightsSha256") != weights
            or metadata.get("model_health_sha256") != digest(health_path)
        ):
            raise ValueError("Model health or weight attestation differs from game summary")
        _snapshot(summary.parent / "model-snapshot.json", weights)
    games = data["results"]
    by_seed = {row["seed"]: row for row in games}
    if len(by_seed) != len(games) or set(by_seed) != set(seeds):
        raise ValueError("Game seeds are missing, duplicated, or outside the frozen split")
    if any(not row["finished"] or row["lives"] != 0 for row in games):
        raise ValueError("Gate and validation require natural-end complete games")
    rows = _trace_rows(trace)
    grouped: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for row in rows:
        if row["seed"] not in grouped:
            raise ValueError("Trace contains an unregistered seed")
        grouped[row["seed"]].append(row)
    counts: Counter[str] = Counter()
    causes: Counter[str] = Counter()
    choice_violations: list[dict[str, Any]] = []
    unrequested_switches: list[dict[str, Any]] = []
    waves: dict[int, int] = {}
    decisions: list[dict[str, Any]] = []
    for seed, game in by_seed.items():
        trace_rows = grouped[seed]
        if len(trace_rows) != game["steps"] or sum(row["reward"] for row in trace_rows) != game["reward"]:
            raise ValueError(f"Trace length or reward mismatch for seed {seed}")
        if [row["step"] for row in trace_rows] != list(range(game["steps"])):
            raise ValueError(f"Trace steps are missing, duplicated, or out of order for seed {seed}")
        calls = 0
        previous_target: str | None = None
        for row in trace_rows:
            choice, options = row.get("model_choice"), row.get("lane_options")
            if choice is not None:
                calls += 1
                if choice not in ("A", "B") or not options or len(options) != 2:
                    raise ValueError("Model choice is not from the offered two-plan menu")
                if "locked_bullet_x" not in row or "target_type" not in row:
                    raise ValueError("Trace omits the executed target intent")
                selected = row.get("selected_lane")
                if selected is None or selected["bullet_x"] != options[0 if choice == "A" else 1]["bullet_x"]:
                    choice_violations.append({"seed": seed, "step": row["step"], "raw_choice": choice})
                if selected is not None and row.get("target_type") != selected.get(
                    "target_type", row.get("target_type")
                ):
                    choice_violations.append(
                        {
                            "seed": seed,
                            "step": row["step"],
                            "raw_choice": choice,
                            "reason": "target_type_changed_before_action",
                        }
                    )
                decisions.append(
                    {
                        "seed": seed,
                        "step": row["step"],
                        "options": options,
                        "raw_choice": choice,
                        "selected_candidate": selected,
                        "executed_intent": {
                            "target_type": row.get("target_type"),
                            "bullet_x": row.get("locked_bullet_x"),
                        },
                        "native_action": row["action"],
                    }
                )
            if (
                choice is None
                and previous_target == "alien"
                and row.get("target_type") == "satellite"
                and row.get("alien_count", 0) > 0
                and not row.get("urgent_low_aliens")
            ):
                unrequested_switches.append(
                    {"seed": seed, "step": row["step"], "interpretation": "possible_bonus_preempt_or_expired_target"}
                )
            if row["lives_after"] < row["lives_before"]:
                if row.get("lowest_alien_y") is not None and row["lowest_alien_y"] >= 180:
                    cause = "suspected_landing"
                elif any(
                    abs(bullet["xy"][0] - row["player_x"]) <= 8 and bullet["xy"][1] >= 160
                    for bullet in row.get("bullets") or []
                ):
                    cause = "suspected_projectile"
                else:
                    cause = "unknown"
                causes[cause] += row["lives_before"] - row["lives_after"]
            counts[row["action"]] += 1
            previous_target = row.get("target_type")
        if calls != game["model_calls"]:
            raise ValueError(f"Recorded model calls differ for seed {seed}")
        waves[seed] = max(row["wave"] for row in trace_rows)
    return {
        "games": games,
        "wave_reached": waves,
        "death_cause_estimates": dict(causes),
        "native_actions": dict(counts),
        "decisions": decisions,
        "choice_violations": choice_violations,
        "possible_unrequested_switches": unrequested_switches,
        "trace_sha256": digest(trace),
        "summary_sha256": digest(summary),
    }


def run_arm(
    run: Path,
    split: str,
    arm: str,
    runner: list[str],
    seeds: list[int],
    code: Path,
    code_sha: dict[str, str],
    weights: str | None,
) -> dict[str, Any]:
    target = run / split / arm
    target.mkdir(parents=True, exist_ok=True)
    summary, trace = target / "summary.json", target / "trace.jsonl.gz"
    if not summary.exists() or not trace.exists():
        variables = {
            "run": str(run.resolve()),
            "arm": arm,
            "split": split,
            "seeds": ",".join(map(str, seeds)),
            "code_dir": str(code.resolve()),
            "summary": str(summary.resolve()),
            "trace": str(trace.resolve()),
            "weights_sha256": weights or "",
        }
        command = [part.format_map(variables) for part in runner]
        with (target / "runner.stdout").open("w") as stdout, (target / "runner.stderr").open("w") as stderr:
            subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    return audit_arm(summary, trace, seeds, code_sha, weights, read(run / "manifest.json")["environment_sha256"])


def validation(run: Path) -> None:
    manifest = require_stage(run, "developed")
    check_frozen(run, manifest)
    config = read(run / "config.json")
    proposal = read(run / "proposal.json")
    development = read(run / "development-result.json")
    seeds = manifest["seeds"]["validation"]
    runner = config["runner_command"]
    arms = {}
    for arm, code, sha, weights in (
        ("incumbent", run / "incumbent-code", manifest["code_sha256"], manifest["weights_sha256"]),
        (
            "candidate",
            run / "candidate-code",
            development["candidate_code_sha256"],
            development["candidate_weights_sha256"],
        ),
    ):
        arms[arm] = run_arm(run, "validation", arm, runner, seeds, code, sha, weights)
    intent_valid = (
        not arms["candidate"]["choice_violations"] and not arms["candidate"]["possible_unrequested_switches"]
    )
    incumbent = {row["seed"]: row["reward"] for row in arms["incumbent"]["games"]}
    candidate = {row["seed"]: row["reward"] for row in arms["candidate"]["games"]}
    gain = statistics.mean(candidate[seed] - incumbent[seed] for seed in seeds)
    selected = intent_valid and gain >= proposal["validation_min_mean_gain"]
    reasons = []
    if not intent_valid:
        reasons.append("candidate overwrote or preempted a model intent")
    if gain < proposal["validation_min_mean_gain"]:
        reasons.append("mean score gain below preregistered validation threshold")
    result = {
        "seeds": seeds,
        "mean_difference": gain,
        "passed": selected,
        "reasons": reasons,
        "arms": {arm: {key: value for key, value in data.items() if key != "decisions"} for arm, data in arms.items()},
    }
    write_once(run / "validation-result.json", result)
    advance(run, manifest, "validated", validation_result_sha256=digest(run / "validation-result.json"))


def _snapshot(path: Path, checksum: str) -> ModelSnapshot:
    data = read(path)
    if data["weights_sha256"] != checksum:
        raise ValueError("Remote NanoJev snapshot hash differs from loaded model health")
    shape = data["architecture"]
    architecture = ArchitectureSpec.uniform_layers(
        num_layers=shape["num_layers"],
        hidden_size=shape["hidden_size"],
        attention_heads=shape["attention_heads"],
        ffn_size=shape["ffn_size"],
    )
    return ModelSnapshot(
        CheckpointRef(data["checkpoint"], "hf_safetensors", checksum), architecture, data["parameters"]
    )


def _harness(code: Path, environment: dict[str, Any]) -> HarnessSpec:
    files = [(name, HarnessFile("program", (code / name).read_text())) for name in CODE_NAMES]
    files.append(("settings.json", HarnessFile("settings", json.dumps(environment, sort_keys=True))))
    return HarnessSpec("space_invaders_lane_controller", files=tuple(files))


def _interval(differences: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(20_000))
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def gate(run: Path) -> None:
    manifest = require_stage(run, "validated")
    check_frozen(run, manifest)
    validation_result = read(run / "validation-result.json")
    if not validation_result["passed"]:
        raise ValueError("Validation failed; the untouched gate split must not be run")
    config, proposal, development = (
        read(run / name) for name in ("config.json", "proposal.json", "development-result.json")
    )
    seeds, runner = manifest["seeds"]["gate"], config["runner_command"]
    arms = {}
    specifications = [
        ("incumbent", run / "incumbent-code", manifest["code_sha256"], manifest["weights_sha256"]),
        (
            "candidate",
            run / "candidate-code",
            development["candidate_code_sha256"],
            development["candidate_weights_sha256"],
        ),
    ]
    if proposal["method"] == "sft":
        specifications.extend(
            [
                ("random", run / "incumbent-code", manifest["code_sha256"], None),
                ("base", run / "incumbent-code", manifest["code_sha256"], config["base_model"]["weights_sha256"]),
                ("code", run / "incumbent-code", manifest["code_sha256"], None),
            ]
        )
    for arm, code, sha, weights in specifications:
        arms[arm] = run_arm(run, "gate", arm, runner, seeds, code, sha, weights)
    intent_valid = (
        not arms["candidate"]["choice_violations"] and not arms["candidate"]["possible_unrequested_switches"]
    )
    scores = {arm: {row["seed"]: row["reward"] for row in audit["games"]} for arm, audit in arms.items()}
    differences = [scores["candidate"][seed] - scores["incumbent"][seed] for seed in seeds]
    incumbent_snapshot = _snapshot(run / "gate" / "incumbent" / "model-snapshot.json", manifest["weights_sha256"])
    candidate_snapshot = _snapshot(
        run / "gate" / "candidate" / "model-snapshot.json", development["candidate_weights_sha256"]
    )
    initial = UpdateCandidate(
        "init", None, incumbent_snapshot, harness=_harness(run / "incumbent-code", config["environment"])
    )
    incumbent = HoneRelease("space-invaders-frozen-incumbent", initial, None)
    candidate = UpdateCandidate(
        "train" if proposal["method"] == "sft" else "harness",
        incumbent.release_id,
        candidate_snapshot,
        harness=_harness(run / "candidate-code", config["environment"]),
    )
    evaluation = EvaluationResult(
        task_ids=tuple(f"seed-{seed}" for seed in seeds),
        candidate_scores=tuple(scores["candidate"][seed] / 1000 for seed in seeds),
        incumbent_scores=tuple(scores["incumbent"][seed] / 1000 for seed in seeds),
        evaluated_variant="master",
    )

    class RecordedEvaluator:
        def evaluate(self, proposal: UpdateCandidate, release: HoneRelease) -> EvaluationResult:
            return evaluation

    rule = proposal["gate"]
    selector = (
        NoRegressionSelector(rule.get("max_regression", 0))
        if rule["selector"] == "no_regression"
        else WinLossSelector(rule.get("min_win_margin", 0))
    )
    _, decision = Gate(RecordedEvaluator(), selector).judge(candidate, incumbent)
    interval = _interval(differences, config["bootstrap_seed"])
    passed = (
        intent_valid
        and decision.selected
        and statistics.mean(differences) >= rule["min_mean_gain"]
        and (not rule.get("require_positive_ci_low", False) or interval[0] > 0)
    )
    reasons = []
    if not intent_valid:
        reasons.append("candidate overwrote or preempted a model intent")
    if not decision.selected:
        reasons.append("Honing Gate rejected the candidate")
    if statistics.mean(differences) < rule["min_mean_gain"]:
        reasons.append("mean score gain below preregistered Gate threshold")
    if rule.get("require_positive_ci_low", False) and interval[0] <= 0:
        reasons.append("paired bootstrap interval includes zero")
    result = {
        "seeds": seeds,
        "per_seed": [
            {
                "seed": seed,
                **{arm: values[seed] for arm, values in scores.items()},
                "candidate_minus_incumbent": (scores["candidate"][seed] - scores["incumbent"][seed]),
                "outcome": (
                    "win"
                    if scores["candidate"][seed] > scores["incumbent"][seed]
                    else "loss" if scores["candidate"][seed] < scores["incumbent"][seed] else "tie"
                ),
            }
            for seed in seeds
        ],
        "mean_scores": {arm: statistics.mean(values.values()) for arm, values in scores.items()},
        "arm_metrics": {
            arm: {
                "mean_model_calls": statistics.mean(game["model_calls"] for game in audit["games"]),
                "executed_model_choices": len(audit["decisions"]),
                "mean_wave_reached": statistics.mean(audit["wave_reached"].values()),
                "death_cause_estimates": audit["death_cause_estimates"],
                "choice_violations": len(audit["choice_violations"]),
                "possible_unrequested_switches": len(audit["possible_unrequested_switches"]),
            }
            for arm, audit in arms.items()
        },
        "mean_difference": statistics.mean(differences),
        "bootstrap_ci95": interval,
        "wins": sum(value > 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "honing_gate": {
            "selected": decision.selected,
            "policy": decision.policy,
            "reason": decision.reason,
            "metrics": dict(decision.metrics),
        },
        "passed": passed,
        "reasons": reasons,
        "arms": {arm: {key: value for key, value in data.items() if key != "games"} for arm, data in arms.items()},
    }
    write_once(run / "gate-result.json", result)
    if passed:
        write_once(
            run / "promoted-incumbent.json",
            {
                "weights_sha256": development["candidate_weights_sha256"],
                "code_sha256": development["candidate_code_sha256"],
                "checkpoint": candidate_snapshot.checkpoint.location,
                "gate_result_sha256": digest(run / "gate-result.json"),
            },
        )
    advance(run, manifest, "gated", gate_result_sha256=digest(run / "gate-result.json"), promoted=passed)


def next_round(previous: Path, config_path: Path, run: Path) -> None:
    previous_manifest = read(previous / "manifest.json")
    if previous_manifest["stage"] not in ("gated", "validated"):
        raise ValueError("Previous round must have a Gate verdict or failed validation")
    if previous_manifest["stage"] == "validated" and read(previous / "validation-result.json")["passed"]:
        raise ValueError("Passed validation must go through the final Gate before another round")
    check_frozen(previous, previous_manifest)
    old_config = read(previous / "config.json")
    new_config = read(config_path)
    if new_config["gate_seed_ledger"] != old_config["gate_seed_ledger"]:
        raise ValueError("The gate seed ledger must carry across rounds")
    if previous_manifest.get("promoted", False):
        expected = read(previous / "promoted-incumbent.json")
    else:
        expected = {
            "weights_sha256": previous_manifest["weights_sha256"],
            "code_sha256": previous_manifest["code_sha256"],
        }
    if new_config["incumbent"]["weights_sha256"] != expected["weights_sha256"]:
        raise ValueError("Next round did not carry forward the accepted incumbent weights")
    new_code = {name: digest(Path(new_config["code"][name])) for name in CODE_NAMES}
    if new_code != expected["code_sha256"]:
        raise ValueError("Next round did not carry forward the accepted incumbent harness")
    freeze(config_path, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--config", type=Path, required=True)
    for name in ("freeze", "review", "register", "develop", "validate", "gate", "next"):
        phase = freeze_parser if name == "freeze" else sub.add_parser(name)
        phase.add_argument("--run", type=Path, required=True)
        if name == "next":
            phase.add_argument("--previous", type=Path, required=True)
            phase.add_argument("--config", type=Path, required=True)
        elif name == "review":
            phase.add_argument("--agent-command", nargs="+")
        elif name == "register":
            phase.add_argument("--proposal", type=Path, required=True)
        elif name == "develop":
            phase.add_argument("--candidate-code", type=Path)
            phase.add_argument("--method-command", nargs="+")
            phase.add_argument("--method-command-file", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.config, args.run)
    elif args.phase == "review":
        review(args.run, args.agent_command)
    elif args.phase == "register":
        register(args.run, args.proposal)
    elif args.phase == "develop":
        if args.method_command and args.method_command_file:
            parser.error("Use one method command source")
        command = json.loads(args.method_command_file.read_text()) if args.method_command_file else args.method_command
        if command is not None and (
            not isinstance(command, list) or not all(isinstance(item, str) for item in command)
        ):
            parser.error("Method command file must hold a JSON array of strings")
        develop(args.run, args.candidate_code, command)
    elif args.phase == "validate":
        validation(args.run)
    elif args.phase == "gate":
        gate(args.run)
    else:
        next_round(args.previous, args.config, args.run)


if __name__ == "__main__":
    main()
