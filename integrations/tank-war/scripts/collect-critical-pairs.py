"""Collect disjoint paired Tank War games with one predeclared critical intervention.

All complete losses are retained. A trigger is not assumed to cause a loss.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_reply(reply: dict) -> dict:
    """Keep only native choices/probabilities; latency metadata may differ."""
    return {
        "answers": {
            head: {"choice": answer["choice"], "probabilities": answer["probabilities"]}
            for head, answer in reply.get("answers", {}).items()
        },
        "branches": {key: policy_reply(value) for key, value in reply.get("branches", {}).items()},
    }


def first_clean_fire(run: Path) -> int | None:
    states, executions = rows(run / "states.jsonl"), rows(run / "execution.jsonl")
    if len(states) != len(executions):
        raise ValueError("clean state and execution counts differ")
    for index, (state, action) in enumerate(zip(states, executions, strict=True)):
        tanks = state["raw_state"]["tanks"]
        a, b = tanks[state["player"]], tanks[1 - state["player"]]
        distance = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
        shot = any(
            motion["shot"]["can_fire"]
            and motion["shot"]["enemy_hit_ms"] is not None
            and motion["shot"]["self_hit_ms"] is None
            for motion in state["request"]["motions"]
        )
        if (
            distance <= 400
            and shot
            and action["selected_action"]["weapon"] == "fire"
            and action["acknowledged_command"]["controls"]["fire"] is True
        ):
            return index
    return None


def compare(clean: Path, fault: Path, layer: str) -> dict:
    a, b = (json.loads((directory / "result.json").read_text()) for directory in (clean, fault))
    manifest_a, manifest_b = (json.loads((directory / "manifest.json").read_text()) for directory in (clean, fault))
    states_a, states_b = rows(clean / "states.jsonl"), rows(fault / "states.jsonl")
    calls_a, calls_b = rows(clean / "model_calls.jsonl"), rows(fault / "model_calls.jsonl")
    exec_a, exec_b = rows(clean / "execution.jsonl"), rows(fault / "execution.jsonl")
    event_path = fault / "private_critical_event.jsonl"
    events = rows(event_path) if event_path.exists() else []
    if len(events) > 1:
        raise ValueError("more than one critical intervention")
    output = {
        "seed": a["seed"],
        "seat": a["seat"],
        "layer": layer,
        "clean_outcome": a["outcome"][a["seat"]],
        "fault_outcome": b["outcome"][b["seat"]],
        "fault_loss": b["outcome"][b["seat"]] == -1,
        "triggered": bool(events),
        "clean_run": str(clean),
        "fault_run": str(fault),
        "clean_result_sha256": sha(clean / "result.json"),
        "fault_result_sha256": sha(fault / "result.json"),
    }
    if not events:
        output["no_trigger_equal"] = [x["state_hash"] for x in states_a] == [
            x["state_hash"] for x in states_b
        ] and a == b
        return output
    event = events[0]
    index = event["decision_index"]
    if min(len(states_a), len(states_b), len(calls_a), len(calls_b), len(exec_a), len(exec_b)) <= index:
        raise ValueError("incomplete selected decision")
    output.update(
        decision_index=index,
        state_hash=event["state_hash"],
        candidate=event["candidate"],
        instance=event["instance"],
        same_prefix=[x["state_hash"] for x in states_a[: index + 1]]
        == [x["state_hash"] for x in states_b[: index + 1]],
        same_request=calls_a[index]["request"] == calls_b[index]["request"],
        same_reply=policy_reply(calls_a[index]["reply"]) == policy_reply(calls_b[index]["reply"]),
        selected_action_changed=exec_a[index]["selected_action"] != exec_b[index]["selected_action"],
        command_changed=exec_a[index]["acknowledged_command"]["controls"]
        != exec_b[index]["acknowledged_command"]["controls"],
        execution_changed=exec_a[index]["executed_action"] != exec_b[index]["executed_action"],
        clean_checkpoint_sha256=calls_a[index].get("checkpoint_sha256") or manifest_a["model"]["weights_sha256"],
        fault_checkpoint_sha256=calls_b[index].get("checkpoint_sha256") or manifest_b["model"]["weights_sha256"],
        outcome_improved_without_fault=a["outcome"][a["seat"]] > b["outcome"][b["seat"]],
    )
    output["layer_isolated"] = bool(
        output["same_prefix"]
        and (
            (
                layer == "harness"
                and not output["same_request"]
                and output["clean_checkpoint_sha256"] == output["fault_checkpoint_sha256"]
            )
            or (
                layer == "weights"
                and output["same_request"]
                and output["clean_checkpoint_sha256"] != output["fault_checkpoint_sha256"]
            )
            or (
                layer == "controller"
                and output["same_request"]
                and output["same_reply"]
                and output["clean_checkpoint_sha256"] == output["fault_checkpoint_sha256"]
            )
        )
    )
    output["local_action_changed"] = bool(
        output["selected_action_changed"] or output["command_changed"] or output["execution_changed"]
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=80)
    parser.add_argument("--seeds-per-layer", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--clean-url", action="append", required=True)
    parser.add_argument("--weights-url", required=True)
    parser.add_argument("--harness-hold", type=Path, required=True)
    parser.add_argument("--harness-turn", type=Path, required=True)
    parser.add_argument("--harness-settings", type=Path)
    parser.add_argument("--select-clean-fire", action="store_true")
    parser.add_argument("--layers", default="harness,weights,controller")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    if len(args.clean_url) != 2 or args.workers < 1:
        parser.error("two clean URLs and positive workers are required")
    plan = json.loads(args.seed_plan.read_text())
    layers = args.layers.split(",")
    if (
        not layers
        or len(set(layers)) != len(layers)
        or any(layer not in {"harness", "weights", "controller"} for layer in layers)
    ):
        parser.error("--layers must be distinct harness, weights, or controller names")
    if args.select_clean_fire and (layers != ["harness"] or args.harness_settings):
        parser.error("--select-clean-fire is only for the question-only harness arm")
    total = len(layers) * args.seeds_per_layer
    seeds = plan["groups"]["final_test"][args.start_index : args.start_index + total]
    if len(seeds) != total or len(set(seeds)) != total:
        parser.error("frozen seed range is incomplete or duplicated")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer_index, layer in enumerate(layers):
        for within, seed in enumerate(
            seeds[layer_index * args.seeds_per_layer : (layer_index + 1) * args.seeds_per_layer]
        ):
            entries.extend(
                [
                    {
                        "layer": layer,
                        "seed": seed,
                        "seat": seat,
                        "seed_index": args.start_index + layer_index * args.seeds_per_layer + within,
                        "split": "blind_test" if within < args.seeds_per_layer // 2 else "development",
                        "trigger": (
                            "incoming_shell"
                            if layer != "controller"
                            and within % 3 == 2
                            and not (layer == "harness" and (args.harness_settings or args.select_clean_fire))
                            else "aim_fire"
                        ),
                    }
                    for seat in ("blue", "red")
                ]
            )
    frozen = args.out / "freeze.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["entries"] != entries:
            raise ValueError("frozen sample plan changed")
    else:
        if args.select_clean_fire:
            version = "v3-harness-at-clean-fire"
        elif args.harness_settings:
            version = "v2-critical-harness-settings"
        else:
            version = "v1-critical-window-20261115"
        frozen.write_text(
            json.dumps(
                {
                    "version": version,
                    "entries": entries,
                    "rule": (
                        "first pre-intervention aim/fire or incoming-shell state; "
                        "controller additionally requires selected permitted fire"
                    ),
                    "loss_policy": "retain every complete fault loss, regardless of trigger or local effect",
                },
                indent=2,
            )
            + "\n"
        )
    configs = args.out / "configs"
    configs.mkdir(exist_ok=True)
    journal = args.out / "attempts.jsonl"
    lock = threading.Lock()

    def collect(entry: dict) -> dict:
        layer, seed, seat = entry["layer"], entry["seed"], entry["seat"]
        name = f"s{seed}-{seat}"
        family = args.out / layer / name
        family.mkdir(parents=True, exist_ok=True)
        config = {
            "layer": layer,
            "kind": (
                ("one_turn_settings" if args.harness_settings else "one_turn_questions")
                if layer == "harness"
                else ("one_turn_checkpoint" if layer == "weights" else "drop_fire_after_selection")
            ),
            "instance": (
                "always-hold-at-clean-fire-v1"
                if layer == "harness" and args.select_clean_fire
                else (
                    "point-blank-v1"
                    if layer == "harness" and args.harness_settings
                    else (
                        "always-hold-v1"
                        if layer == "harness" and entry["trigger"] == "aim_fire"
                        else (
                            "reverse-turn-v1"
                            if layer == "harness"
                            else "reverse-strength-3p0-v1" if layer == "weights" else "drop-fire-v1"
                        )
                    )
                )
            ),
            "trigger": entry["trigger"],
        }
        if layer == "harness":
            config["alternate_harness_dir"] = str(
                args.harness_settings
                if args.harness_settings
                else args.harness_hold if entry["trigger"] == "aim_fire" else args.harness_turn
            )
        if layer == "weights":
            config["alternate_decision_server"] = args.weights_url
        config_path = configs / (
            f"{layer}-{seed}-{seat}.json" if args.select_clean_fire else f"{layer}-{entry['trigger']}.json"
        )
        if not args.select_clean_fire:
            with lock:
                if config_path.exists() and json.loads(config_path.read_text()) != config:
                    raise ValueError("fault configuration changed")
                if not config_path.exists():
                    config_path.write_text(json.dumps(config, indent=2) + "\n")
        result = {**entry, "clean_run": str(family / "clean"), "fault_run": str(family / "fault")}
        for arm in ("clean", "fault"):
            out = family / arm
            if arm == "fault" and args.select_clean_fire:
                index = first_clean_fire(family / "clean")
                config["decision_index"] = index if index is not None else len(rows(family / "clean" / "states.jsonl"))
                result["preselected_clean_fire_index"] = index
                with lock:
                    if config_path.exists() and json.loads(config_path.read_text()) != config:
                        raise ValueError("frozen per-game intervention changed")
                    if not config_path.exists():
                        config_path.write_text(json.dumps(config, indent=2) + "\n")
            if (out / "result.json").exists() and not (out / "error.json").exists():
                result[f"{arm}_reused"] = True
                continue
            if out.exists():
                raise ValueError(f"incomplete previous run: {out}")
            url = args.clean_url[seed % len(args.clean_url)]
            command = [
                str(args.node),
                str(args.repo / "integrations/tank-war/scripts/match.mjs"),
                f"--seed={seed}",
                f"--seat={seat}",
                f"--decision-server={url}",
                f"--out={out}",
                "--seconds=30",
                "--interval-ms=200",
                "--opponent-interval-ms=300",
            ]
            if arm == "fault":
                command.append(f"--critical-fault-file={config_path}")
            started = time.monotonic()
            run = subprocess.run(command, cwd=args.repo, text=True, capture_output=True, timeout=900)
            result[f"{arm}_seconds"] = round(time.monotonic() - started, 2)
            result[f"{arm}_returncode"] = run.returncode
            if run.returncode:
                result[f"{arm}_stderr"] = run.stderr[-2000:]
                break
        if (family / "clean" / "result.json").exists() and (family / "fault" / "result.json").exists():
            evidence = compare(family / "clean", family / "fault", layer)
            (family / "pair_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
            result.update(evidence=evidence)
        with lock, journal.open("a") as output:
            output.write(json.dumps(result) + "\n")
        return result

    pending = iter(entries)
    active: set[concurrent.futures.Future] = set()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while active or time.time() < args.deadline_unix:
            while len(active) < args.workers and time.time() < args.deadline_unix:
                try:
                    entry = next(pending)
                except StopIteration:
                    break
                active.add(pool.submit(collect, entry))
            if not active:
                break
            done, active = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                completed += 1
                try:
                    result = future.result()
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "layer": result["layer"],
                                "seed": result["seed"],
                                "seat": result["seat"],
                                "loss": result.get("evidence", {}).get("fault_loss"),
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    with lock, journal.open("a") as output:
                        output.write(json.dumps({"error": repr(error), "completed": completed}) + "\n")
                    print(json.dumps({"completed": completed, "error": repr(error)}), flush=True)
        # Running children finish; the deadline stops new work without killing unrelated services.
    print(
        json.dumps({"completed_pairs": completed, "planned_pairs": len(entries), "deadline_unix": args.deadline_unix})
    )


if __name__ == "__main__":
    main()
