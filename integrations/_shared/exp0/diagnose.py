"""Run one fresh, tool-disabled diagnosis per episode and mode (``read`` / ``probe``); Griddly v0 ``diagnose_codex.py``.

  python3 diagnose.py READ_INPUTS.jsonl OUT_DIR --mode read  --backend codex|stub [--model gpt-6.2-sol]
  python3 diagnose.py READ_INPUTS.jsonl OUT_DIR --mode probe --probes PROBE_INPUTS.jsonl --backend codex|stub

Every episode and mode starts a new Codex CLI session in an empty temporary directory, ephemeral, with the
user config and rules ignored, a read-only sandbox, web search off and every tool feature disabled; the
prompt is ``diagnosis_prompt.txt`` + the episode payload on stdin, the reply must match
``diagnosis_schema.json``. Each attempt's raw stdout/stderr, return code and wall time are saved as
``<episode>-attempt-<ns>.json``; an episode gets at most two attempts (the second only after an invalid
reply; ``--max-attempts 1`` for a call-capped smoke), after which it is recorded as ``{"status": "invalid"}``
(scored as incorrect). The settings
(model, CLI version, prompt/schema/input digests, command) are frozen in ``settings.json``; a resumed run
must match them. Reported token usage is kept; dollar cost is ``null`` (unknown) unless the CLI reports it.

``--backend codex`` makes real model calls. It refuses to run unless ``EXP0_DIAGNOSIS_APPROVED=yes`` is set
in the environment, which is to be set only once the diagnosis model and its subscription are approved.
``--backend stub`` runs ``tests/stub_codex.py`` through the same subprocess path (no network) to test the
runner end to end.

``--prompt FILE`` (added 2026-11-26 for the pre-registered probe variant of the VVVVVV and full-HK v0 releases;
``diagnosis_prompt_probe_variant.txt``) replaces ``diagnosis_prompt.txt``; its digest is ``prompt_sha256`` and its
file name ``prompt_file`` in ``settings.json``. Without the option the settings are exactly as before.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import COMPONENTS, canonical

HERE = Path(__file__).resolve().parent
PROMPT = HERE / "diagnosis_prompt.txt"
SCHEMA = HERE / "diagnosis_schema.json"
# 2026-11-26: resolved at run time (no absolute local path in the code): $EXP0_CODEX_BIN, else `codex` on PATH, else
# ~/.local/bin/codex (where it was before)
REAL_CODEX = os.environ.get("EXP0_CODEX_BIN") or shutil.which("codex") or str(Path.home() / ".local" / "bin" / "codex")
STUB_CODEX = HERE / "tests" / "stub_codex.py"
DEFAULT_MODEL = "gpt-6.2-sol"
NEUTRAL_ID = re.compile(r"[a-z]{1,3}[0-9]{3}")
# Griddly v0 disabled these features (CLI 0.171.0-alpha.13).
GRIDDLY_DISABLED = ("shell_tool", "apps", "plugins", "hooks", "multi_agent", "multi_agent_v2", "skill_search")
# The local CLI 0.167.0 also enables these tool-providing features by default (``codex features list``,
# 2026-11-23); they are disabled too so that the session stays tool-free. Unknown names are not passed.
EXTRA_DISABLED = (
    "unified_exec",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "view_image",
    "in_app_browser",
    "sleep_tool",
    "tool_suggest",
    "goals",
)
APPROVAL_VARIABLE = "EXP0_DIAGNOSIS_APPROVED"


def inputs(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if set(row) != {"episode_id", "text"} or not isinstance(row["text"], str):
            raise ValueError("input rows must contain only episode_id and text")
        episode_id = row["episode_id"]
        if not NEUTRAL_ID.fullmatch(episode_id) or episode_id in rows:
            raise ValueError("duplicate or non-neutral episode ID")
        rows[episode_id] = row["text"]
    if not rows:
        raise ValueError("no diagnosis inputs")
    return rows


def executable(backend: str, codex_bin: str | None) -> list[str]:
    if backend == "stub":
        return [sys.executable, "-I", "-B", str(Path(codex_bin) if codex_bin else STUB_CODEX)]
    if os.environ.get(APPROVAL_VARIABLE) != "yes":
        raise SystemExit(
            f"refusing real diagnosis calls: set {APPROVAL_VARIABLE}=yes only once the diagnosis model "
            "and its subscription are approved"
        )
    return [codex_bin or REAL_CODEX]


def available_features(base: list[str]) -> set[str] | None:
    try:
        listing = subprocess.run([*base, "features", "list"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listing.returncode:
        return None
    return {line.split()[0] for line in listing.stdout.splitlines() if line.strip()}


def command(base: list[str], model: str, directory: Path, features: set[str] | None) -> list[str]:
    disabled = [
        name
        for name in (*GRIDDLY_DISABLED, *EXTRA_DISABLED)
        if name in GRIDDLY_DISABLED or features is None or name in features
    ]
    flags = [part for name in disabled for part in ("--disable", name)]
    return [
        *base,
        "--ask-for-approval",
        "never",
        "exec",
        "--model",
        model,
        "--cd",
        str(directory),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        *flags,
        "-c",
        'web_search="disabled"',
        "-c",
        "project_doc_max_bytes=0",
        "--json",
        "--output-schema",
        str(SCHEMA),
        "-",
    ]


def decode(stdout: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    items = [event["item"] for event in events if event["type"].startswith("item.")]
    if any(item["type"] not in {"agent_message", "reasoning"} for item in items):
        raise RuntimeError("Codex emitted a non-diagnostic item in the isolated run")
    messages = [
        event["item"]["text"]
        for event in events
        if event["type"] == "item.completed" and event["item"]["type"] == "agent_message"
    ]
    turns = [event for event in events if event["type"] == "turn.completed"]
    if len(turns) != 1 or len(messages) != 1:
        raise ValueError("expected one complete turn and one diagnosis message")
    result = json.loads(messages[0])
    if (
        not isinstance(result, dict)
        or set(result) != {"cause", "confidence", "explanation"}
        or result["cause"] not in (*COMPONENTS, "uncertain")
        or type(result["confidence"]) not in (float, int)
        or not 0 <= result["confidence"] <= 1
        or not isinstance(result["explanation"], str)
        or not result["explanation"].strip()
    ):
        raise ValueError("Codex diagnosis does not match the schema")
    return result, turns[0].get("usage")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> None:
    traces = inputs(args.inputs)
    probes = inputs(args.probes) if args.probes else {}
    if (args.mode == "probe" and set(probes) != set(traces)) or (args.mode == "read" and probes):
        raise ValueError("only probe mode receives probe results and its episode IDs must match")
    prompt_path = getattr(args, "prompt", None) or PROMPT
    base = executable(args.backend, args.codex_bin)
    login = subprocess.run([*base, "login", "status"], capture_output=True, check=False, timeout=30)
    if login.returncode:
        raise RuntimeError("Codex CLI login is required before evaluation")
    cli_version = subprocess.check_output([*base, "--version"], encoding="utf-8", timeout=30).strip()
    features = available_features(base)
    template = command(base, args.model, Path("<fresh-temporary-directory>"), features)
    settings = {
        "mode": args.mode,
        "model": args.model,
        "backend": args.backend,
        "cli_version": cli_version,
        "timeout_seconds": args.timeout,
        "inputs_sha256": file_digest(args.inputs),
        "probe_inputs_sha256": file_digest(args.probes) if args.probes else None,
        "prompt_sha256": file_digest(prompt_path),
        "schema_sha256": file_digest(SCHEMA),
        "command": [part for part in template[len(base) :] if part != str(SCHEMA)],
        "max_attempts": args.max_attempts,
    }
    if prompt_path != PROMPT:
        settings["prompt_file"] = Path(prompt_path).name
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / "settings.json"
    if config.exists() and json.loads(config.read_text(encoding="utf-8")) != settings:
        raise ValueError("resume requires the same diagnosis settings")
    config.write_text(canonical(settings) + "\n", encoding="utf-8")
    output = args.out / "diagnoses.jsonl"
    complete = (
        {json.loads(line)["episode_id"] for line in output.read_text(encoding="utf-8").splitlines()}
        if output.exists()
        else set()
    )
    instruction = Path(prompt_path).read_text(encoding="utf-8").strip()
    with output.open("a", encoding="utf-8") as stream:
        for episode_id, trace in traces.items():
            if episode_id in complete:
                continue
            payload: dict[str, Any] = {"episode_id": episode_id, "trace": trace}
            if probes:
                payload["probe_results"] = probes[episode_id]
            prompt = instruction + "\n\n" + canonical(payload)
            for attempt in range(1, args.max_attempts + 1):
                started = time.perf_counter()
                attempt_file = args.out / f"{episode_id}-attempt-{time.time_ns()}.json"
                with tempfile.TemporaryDirectory(prefix="exp0-diagnosis-") as directory:
                    try:
                        process = subprocess.run(
                            command(base, args.model, Path(directory), features),
                            cwd=directory,
                            input=prompt,
                            capture_output=True,
                            encoding="utf-8",
                            errors="replace",
                            check=False,
                            timeout=args.timeout,
                        )
                    except subprocess.TimeoutExpired as error:
                        attempt_file.write_text(
                            canonical(
                                {
                                    "status": "timeout",
                                    "wall_seconds": time.perf_counter() - started,
                                    "stdout": (
                                        (error.stdout or b"").decode("utf-8", "replace")
                                        if isinstance(error.stdout, bytes)
                                        else (error.stdout or "")
                                    ),
                                    "stderr": (
                                        (error.stderr or b"").decode("utf-8", "replace")
                                        if isinstance(error.stderr, bytes)
                                        else (error.stderr or "")
                                    ),
                                }
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        raise RuntimeError(f"the CLI timed out for {episode_id}; see {attempt_file.name}") from None
                elapsed = time.perf_counter() - started
                attempt_file.write_text(
                    canonical(
                        {
                            "returncode": process.returncode,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                            "wall_seconds": elapsed,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                if process.returncode:
                    raise RuntimeError(f"the CLI failed for {episode_id}; see {attempt_file.name}")
                try:
                    result, usage = decode(process.stdout)
                except ValueError:
                    if attempt < args.max_attempts:
                        continue
                    result, usage = {"status": "invalid"}, None
                record = {
                    "episode_id": episode_id,
                    "mode": args.mode,
                    "model": args.model,
                    "attempts": attempt,
                    "wall_seconds": elapsed,
                    "usage": usage,
                    "cost_usd": None,
                    **result,
                }
                stream.write(canonical(record) + "\n")
                stream.flush()
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mode", choices=("read", "probe"), required=True)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--backend", choices=("codex", "stub"), required=True)
    parser.add_argument("--codex-bin", help="path of the CLI (codex) or of a stub script (stub)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, help="another frozen prompt file (the pre-registered probe variant)")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        choices=(1, 2),
        help="2 as Griddly (a second attempt only after an invalid reply); 1 for call-capped smokes",
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    run(args)


if __name__ == "__main__":
    main()
