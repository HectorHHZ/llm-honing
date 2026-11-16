"""Run one frozen, blinded Sol-high diagnosis per Tank War case and mode."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

AUTH = Path("/home/xinkaiz/.codex/auth.json")
BIN_DIR = Path("/home/xinkaiz/.codex/packages/standalone/current/bin")
FORBIDDEN = ("inject", "fault", "clean", "gold", "manifest", "source_run", "seed")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usage_and_tools(path: Path) -> tuple[dict | None, bool]:
    if not path.exists():
        return None, False
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    completed = [event for event in events if event.get("type") == "turn.completed"]
    tool = any(
        event.get("type", "").startswith("item.") and event.get("item", {}).get("type") != "agent_message"
        for event in events
    )
    if len(completed) != 1:
        return None, tool
    record = completed[0]["usage"]
    return {key: int(record[key]) for key in ("input_tokens", "output_tokens")}, tool


def one(case: dict, mode: str, out: Path, prompt: str) -> dict:
    episode = case["episode_id"]
    target = out / mode / episode
    if target.exists():
        raise FileExistsError(f"prior diagnosis exists: {target}")
    target.mkdir()
    replay_path = out / "inputs" / f"{episode}.json"
    replay = replay_path.read_text()
    if sha(replay_path) != case["input_sha256"]:
        raise ValueError("frozen replay changed")
    checks = "null"
    if mode == "probe":
        probe_path = out / "probes" / episode / "probes.json"
        probe = read(probe_path)
        if probe["probe_version"] != "tank183-critical-all-decisions-v1" or [
            row["decision_index"] for row in probe["indices"]
        ] != list(range(case["decisions"])):
            raise ValueError("controlled checks do not cover every decision")
        checks = probe_path.read_text()
    payload = prompt + replay + "\nCONTROLLED REPEAT JSON:\n" + checks
    if (
        len(payload) > 1_500_000
        or any(term in payload.lower() for term in FORBIDDEN)
        or any(value in payload for value in (str(case["seed"]), case["source_run"]))
    ):
        raise ValueError("blind input reveals construction or exceeds limit")
    command = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/home",
        "--dir",
        "/home/sandbox",
        "--dir",
        "/home/sandbox/.codex",
        "--dir",
        "/run",
        "--dir",
        "/run/systemd",
        "--ro-bind",
        "/run/systemd/resolve",
        "/run/systemd/resolve",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc",
        "/etc",
        "--ro-bind",
        str(BIN_DIR),
        "/app",
        "--ro-bind",
        str(AUTH),
        "/home/sandbox/.codex/auth.json",
        "--ro-bind",
        str(out / "schema.json"),
        "/schema.json",
        "--bind",
        str(target),
        "/work",
        "--setenv",
        "HOME",
        "/home/sandbox",
        "--setenv",
        "CODEX_HOME",
        "/home/sandbox/.codex",
        "--chdir",
        "/work",
        "/app/codex",
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        "/schema.json",
        "--output-last-message",
        "/work/answer.json",
        "-m",
        "gpt-6-sol",
        "-c",
        'model_reasoning_effort="high"',
        "-",
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=1200, check=False)
        (target / "events.jsonl").write_text(result.stdout)
        (target / "stderr.txt").write_text(result.stderr)
        attempt = {"exit_code": result.returncode}
    except subprocess.TimeoutExpired as error:
        attempt = {"error": repr(error)}
    attempt["elapsed_seconds"] = round(time.monotonic() - started, 3)
    usage, tool = usage_and_tools(target / "events.jsonl")
    attempt.update(usage=usage, tool_call=tool)
    if attempt.get("exit_code") == 0:
        try:
            answer = read(target / "answer.json")
            index = answer["decision_index"]
            if index is not None and (type(index) is not int or not 0 <= index < case["decisions"]):
                raise ValueError("decision index outside episode")
            if (
                answer["cause"] not in {"harness", "weights", "controller", "uncertain"}
                or answer["impact_on_loss"] not in {"likely", "unclear", "unlikely"}
                or not 0 <= answer["confidence"] <= 1
            ):
                raise ValueError("invalid answer")
        except (OSError, ValueError, KeyError, TypeError) as error:
            attempt["output_validation_error"] = repr(error)
    attempt.update(
        episode_id=episode,
        mode=mode,
        model="gpt-6-sol",
        effort="high",
        input_sha256=sha(replay_path),
        checks_sha256=hashlib.sha256(checks.encode()).hexdigest(),
        prompt_sha256=sha(out / "prompt.txt"),
        schema_sha256=sha(out / "schema.json"),
        command=command,
    )
    (target / "attempt.json").write_text(json.dumps(attempt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "episode_id": episode,
                "mode": mode,
                "exit_code": attempt.get("exit_code"),
                "seconds": attempt["elapsed_seconds"],
                "usage": usage,
                "invalid": attempt.get("output_validation_error"),
                "tool_call": tool,
            }
        ),
        flush=True,
    )
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("read", "probe"), required=True)
    args = parser.parse_args()
    out = args.out
    freeze, protocol = read(out / "sample-freeze.json"), read(out / "protocol.json")
    if (
        freeze["status"] != "frozen_before_model_answers"
        or protocol["model"] != "gpt-6-sol"
        or [
            sha(out / name) == freeze[f"{name.split('.')[0]}_sha256"]
            for name in ("protocol.json", "prompt.txt", "schema.json")
        ]
        != [True, True, True]
    ):
        raise ValueError("frozen protocol changed")
    if sha(out / "collection-cutoff.json") != freeze["collection_cutoff_sha256"]:
        raise ValueError("collection cutoff changed after blind freeze")
    replays = read(out / "replays" / "summary.json")
    if replays["cases"] != len(freeze["cases"]) or replays["valid"] != replays["cases"]:
        raise ValueError("full replay gate incomplete")
    if args.mode == "probe" and read(out / "probes" / "summary.json")["status"] != "complete":
        raise ValueError("probe coverage incomplete")
    version = subprocess.run(
        [str(BIN_DIR / "codex"), "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    if version != "codex-cli 0.158.0":
        raise ValueError(f"CLI version changed: {version}")
    destination = out / args.mode
    destination.mkdir(exist_ok=False)
    (destination / "config.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model": "gpt-6-sol",
                "effort": "high",
                "cli_version": version,
                "cases": len(freeze["cases"]),
                "parallel": 2,
                "sample_freeze_sha256": sha(out / "sample-freeze.json"),
            },
            indent=2,
        )
        + "\n"
    )
    attempts = []
    prompt = (out / "prompt.txt").read_text()
    for offset in range(0, len(freeze["cases"]), 2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            batch = list(
                pool.map(lambda case: one(case, args.mode, out, prompt), freeze["cases"][offset : offset + 2])
            )
        attempts.extend(batch)
        if any(
            x.get("exit_code") != 0 or x.get("output_validation_error") or x.get("tool_call") or x.get("usage") is None
            for x in batch
        ):
            break
    summary = {
        "status": "complete" if len(attempts) == len(freeze["cases"]) else "stopped_invalid",
        "mode": args.mode,
        "completed": len(attempts),
        "expected": len(freeze["cases"]),
        "input_tokens": sum((x.get("usage") or {}).get("input_tokens", 0) for x in attempts),
        "output_tokens": sum((x.get("usage") or {}).get("output_tokens", 0) for x in attempts),
        "elapsed_seconds_sum": sum(x["elapsed_seconds"] for x in attempts),
        "invalid": [
            x["episode_id"]
            for x in attempts
            if x.get("exit_code") != 0
            or x.get("output_validation_error")
            or x.get("tool_call")
            or x.get("usage") is None
        ],
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
