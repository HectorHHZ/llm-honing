"""Run a sealed episode-only Tank War diagnosis on previously frozen games."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def usage_and_tools(path: Path) -> tuple[dict | None, bool]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    completed = [item for item in events if item.get("type") == "turn.completed"]
    tool_call = any(
        item.get("type", "").startswith("item.") and item.get("item", {}).get("type") != "agent_message"
        for item in events
    )
    if len(completed) != 1:
        return None, tool_call
    usage = completed[0]["usage"]
    return {key: int(usage[key]) for key in ("input_tokens", "output_tokens")}, tool_call


def one(case: dict, mode: str, source: Path, out: Path, prompt: str) -> dict:
    episode_id = case["episode_id"]
    target = out / mode / episode_id
    target.mkdir(parents=True, exist_ok=False)
    input_path = source / "inputs" / f"{episode_id}.json"
    if sha(input_path) != case["input_sha256"]:
        raise ValueError(f"source replay changed: {episode_id}")
    checks = "null"
    if mode in ("probe", "probe+fewshot"):
        checks_path = source / "probes" / episode_id / "probes.json"
        if sha(checks_path) != case["probes_sha256"]:
            raise ValueError(f"source probes changed: {episode_id}")
        checks = checks_path.read_text()
    payload = prompt + input_path.read_text() + "\nCONTROLLED REPEAT JSON:\n" + checks
    if len(payload) > 1_500_000 or any(term in payload.lower() for term in FORBIDDEN):
        raise ValueError(f"leaking or oversized diagnostic payload: {episode_id}")
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
        read(out / "freeze.json")["model"],
        "-c",
        f'model_reasoning_effort="{read(out / "freeze.json")["effort"]}"',
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
    usage, tool_call = (
        usage_and_tools(target / "events.jsonl") if (target / "events.jsonl").exists() else (None, False)
    )
    attempt.update(usage=usage, tool_call=tool_call)
    if attempt.get("exit_code") == 0:
        try:
            answer = read(target / "answer.json")
            if (
                set(answer) != {"cause", "confidence", "explanation"}
                or answer["cause"] not in {"harness", "weights", "controller", "uncertain"}
                or not 0 <= answer["confidence"] <= 1
                or not isinstance(answer["explanation"], str)
            ):
                raise ValueError("invalid episode-only answer")
        except (OSError, ValueError, KeyError, TypeError) as error:
            attempt["output_validation_error"] = repr(error)
    attempt.update(
        episode_id=episode_id,
        mode=mode,
        model=read(out / "freeze.json")["model"],
        effort=read(out / "freeze.json")["effort"],
        input_sha256=sha(input_path),
        checks_sha256=hashlib.sha256(checks.encode()).hexdigest(),
        prompt_sha256=sha(out / "prompt.txt"),
        schema_sha256=sha(out / "schema.json"),
        command=command,
    )
    (target / "attempt.json").write_text(json.dumps(attempt, indent=2) + "\n")
    print(json.dumps({"episode_id": episode_id, "mode": mode, **attempt}), flush=True)
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("read", "probe", "read+fewshot", "probe+fewshot"), required=True)
    args = parser.parse_args()
    source, out = args.source_blind, args.out
    freeze = read(out / "freeze.json")
    if args.mode not in freeze["modes"]:
        raise ValueError(f"mode not in frozen experiment: {args.mode}")
    if freeze["status"] != "frozen_before_episode_only_answers" or any(
        sha(out / filename) != freeze[f"{filename.split('.')[0]}_sha256"]
        for filename in ("protocol.json", "prompt.txt", "schema.json")
    ):
        raise ValueError("episode-only freeze changed")
    protocol = read(out / "protocol.json")
    if freeze["model"] not in {"gpt-6-sol", "gpt-6-astra"} or freeze["effort"] not in {"medium", "high"}:
        raise ValueError("unsupported frozen model or effort")
    if (freeze["model"], freeze["effort"]) != (protocol["model"], protocol["effort"]):
        raise ValueError("frozen model or effort differs from protocol")
    if sha(source / "sample-freeze.json") != freeze["source_sample_freeze_sha256"]:
        raise ValueError("source sample changed")
    if read(source / "replays" / "summary.json")["valid"] != len(freeze["cases"]):
        raise ValueError("source replay gate incomplete")
    if read(source / "probes" / "summary.json")["status"] != "complete":
        raise ValueError("source probe gate incomplete")
    version = subprocess.run([str(BIN_DIR / "codex"), "--version"], text=True, capture_output=True, check=True)
    if version.stdout.strip() != "codex-cli 0.158.0":
        raise ValueError("CLI version changed")
    destination = out / args.mode
    destination.mkdir(exist_ok=False)
    prompt = (out / "prompt.txt").read_text()
    attempts = []
    input_cap = freeze.get("maximum_input_tokens_per_mode")
    output_cap = freeze.get("maximum_output_tokens_per_mode")
    for offset in range(0, len(freeze["cases"]), 2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            batch = list(
                pool.map(lambda case: one(case, args.mode, source, out, prompt), freeze["cases"][offset : offset + 2])
            )
        attempts.extend(batch)
        if any(
            item.get("exit_code") != 0
            or item.get("output_validation_error")
            or item.get("tool_call")
            or item.get("usage") is None
            for item in batch
        ):
            break
        if (input_cap is not None and sum(item["usage"]["input_tokens"] for item in attempts) > input_cap) or (
            output_cap is not None and sum(item["usage"]["output_tokens"] for item in attempts) > output_cap
        ):
            break
    input_used = sum((item.get("usage") or {}).get("input_tokens", 0) for item in attempts)
    output_used = sum((item.get("usage") or {}).get("output_tokens", 0) for item in attempts)
    over_budget = (input_cap is not None and input_used > input_cap) or (
        output_cap is not None and output_used > output_cap
    )
    if over_budget:
        status = "stopped_budget"
    elif len(attempts) == len(freeze["cases"]):
        status = "complete"
    else:
        status = "stopped_invalid"
    summary = {
        "status": status,
        "mode": args.mode,
        "completed": len(attempts),
        "expected": len(freeze["cases"]),
        "input_tokens": input_used,
        "output_tokens": output_used,
        "elapsed_seconds_sum": sum(item["elapsed_seconds"] for item in attempts),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
