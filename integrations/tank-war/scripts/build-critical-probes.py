"""Build neutral, every-decision reference checks for frozen blind Tank War cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: dict, out: Path, repo: Path, node: Path, url: str, harness_repo: Path | None) -> dict:
    episode = case["episode_id"]
    target = out / "probes" / episode
    target.mkdir(parents=True, exist_ok=True)
    source = Path(case["source_run"])
    if harness_repo and any(part.startswith("harness-selected-fire-") for part in source.parts):
        repo = harness_repo
    manifest = read(source / "manifest.json")
    expected_legacy_differences = (
        {"scripts/match.mjs", "scripts/controller-faults.mjs"} if "formal-faults-final-v2" in source.parts else set()
    )
    mismatches = []
    for name, expected in manifest["sources"].items():
        if sha(repo / "integrations/tank-war" / name) != expected:
            if name not in expected_legacy_differences:
                raise ValueError(f"probe source differs from recorded game: {episode}/{name}")
            mismatches.append(name)
    collected = []
    for start in range(0, case["decisions"], 11):
        end = min(start + 11, case["decisions"])
        part = target / f"part-{start:03d}.json"
        if not part.exists():
            command = [
                str(node),
                str(repo / "integrations/tank-war/scripts/probe-critical-window.mjs"),
                f"--run={source}",
                f"--out={part}",
                f"--start={start}",
                f"--end-exclusive={end}",
                f"--reference-decision-server={url}",
            ]
            started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=900, check=False)
                attempt = {
                    "returncode": result.returncode,
                    "stderr": result.stderr[-3000:],
                    "stdout": result.stdout[-1000:],
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                attempt = {"error": repr(error)}
            attempt.update(command=command, elapsed_seconds=round(time.monotonic() - started, 2))
            with (target / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(attempt) + "\n")
            if attempt.get("returncode") != 0:
                raise RuntimeError(f"probe failed: {episode}/{start}")
        data = read(part)
        if (
            data["probe_version"] != "tank183-critical-all-decisions-v1"
            or data["identity_tolerance"] != 0.01
            or [row["decision_index"] for row in data["indices"]] != list(range(start, end))
        ):
            raise ValueError(f"probe chunk changed: {episode}/{start}")
        collected.extend(data["indices"])
    if [row["decision_index"] for row in collected] != list(range(case["decisions"])):
        raise ValueError(f"incomplete probe coverage: {episode}")
    public = {"probe_version": "tank183-critical-all-decisions-v1", "identity_tolerance": 0.01, "indices": collected}
    (target / "probes.json").write_text(json.dumps(public, separators=(",", ":")))
    record = {
        "episode_id": episode,
        "decisions": len(collected),
        "probe_sha256": sha(target / "probes.json"),
        "legacy_probe_source_differences": mismatches,
    }
    print(json.dumps(record), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--harness-repo", type=Path)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--clean-url", action="append", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if freeze["status"] != "frozen_before_model_answers" or any(
        sha(args.out / name) != freeze[f"{name.split('.')[0]}_sha256"]
        for name in ("protocol.json", "prompt.txt", "schema.json")
    ):
        raise ValueError("blind freeze or prompt changed")
    if (args.out / "probes" / "summary.json").exists():
        raise FileExistsError("probes already completed")
    (args.out / "probes").mkdir(exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_case,
                case,
                args.out,
                args.repo,
                args.node,
                args.clean_url[index % len(args.clean_url)],
                args.harness_repo,
            )
            for index, case in enumerate(freeze["cases"])
        ]
        records.extend(future.result() for future in concurrent.futures.as_completed(futures))
    summary = {
        "status": "complete",
        "cases": len(records),
        "decisions": sum(row["decisions"] for row in records),
        "records": records,
        "sample_freeze_sha256": sha(args.out / "sample-freeze.json"),
    }
    (args.out / "probes" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
