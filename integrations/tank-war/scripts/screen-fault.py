"""Run one Tank War fault candidate on the fixed paired validation seed set.

This preserves every attempt, refuses incomplete prior runs, and requires the
independent clean replay gate before spending calls on candidate screening.
Run candidates sequentially; the clean arm is reused only with the same model
endpoint and build hashes, which the paired validator checks at completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from exp0_protocol import validate_fault


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_gate(path: Path, identity_seeds: list[int]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    expected = {(seed, seat) for seed in identity_seeds for seat in ("blue", "red")}
    observed = {(row.get("seed"), row.get("seat")) for row in rows}
    if len(rows) != 128 or observed != expected or not all(row.get("passed") is True for row in rows):
        raise ValueError("64-seed dual-seat independent replay gate is incomplete or failed")


def _completed(run: Path) -> bool:
    if not run.exists():
        return False
    if (run / "error.json").exists() or not (run / "manifest.json").exists() or not (run / "result.json").exists():
        raise ValueError(f"incomplete or failed prior run: {run}")
    return True


def run_grid(
    plan: dict[str, Any],
    gate_log: Path,
    candidate_file: Path,
    out: Path,
    url: str,
    node: Path,
) -> dict[str, Any]:
    require_gate(gate_log, plan["groups"]["identity"])
    candidate = _read_json(candidate_file)
    layer = candidate.get("layer")
    instance = candidate.get("instance")
    if (
        layer not in {"harness", "controller"}
        or not isinstance(instance, str)
        or not instance.replace("-", "").replace("_", "").isalnum()
    ):
        raise ValueError("candidate must be a named Tank War harness or controller fault")
    if layer == "harness" and not (candidate_file.parent / "index.json").exists():
        raise ValueError("harness candidate is missing index.json")
    seeds = plan["groups"]["fault_validation"]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise ValueError("validation requires exactly 128 distinct seed blocks")
    digest = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    record = {"candidate": candidate, "candidate_sha256": digest, "url": url, "seed_plan_version": plan["version"]}
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "candidate.json"
    if record_path.exists() and _read_json(record_path) != record:
        raise ValueError("screening output already belongs to a different candidate or endpoint")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clean_root = out.parent / "clean"
    fault_root = out / "fault"
    clean_root.mkdir(parents=True, exist_ok=True)
    fault_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("match.mjs")
    clean_config = {
        "url": url,
        "seed_plan_version": plan["version"],
        "node": str(node),
        "match_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    clean_config_path = clean_root / "screen-config.json"
    if clean_config_path.exists() and _read_json(clean_config_path) != clean_config:
        raise ValueError("shared clean arm belongs to a different model endpoint, build, or seed plan")
    clean_config_path.write_text(json.dumps(clean_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log = out / "attempts.jsonl"
    for seed in seeds:
        for seat in ("blue", "red"):
            for arm, root in (("clean", clean_root), ("fault", fault_root)):
                run = root / f"s{seed}-{seat}"
                if _completed(run):
                    continue
                command = [
                    os.fspath(node),
                    os.fspath(script),
                    f"--seed={seed}",
                    f"--seat={seat}",
                    f"--decision-server={url}",
                    f"--out={run}",
                    "--seconds=30",
                    "--interval-ms=200",
                    "--opponent-interval-ms=300",
                ]
                if arm == "fault":
                    if layer == "harness":
                        command.append(f"--harness-dir={candidate_file.parent}")
                    else:
                        command.append(f"--controller-fault-file={candidate_file}")
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
                    attempt = {
                        "arm": arm,
                        "seed": seed,
                        "seat": seat,
                        "run_dir": str(run),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except (OSError, subprocess.TimeoutExpired) as error:
                    attempt = {"arm": arm, "seed": seed, "seat": seat, "run_dir": str(run), "error": repr(error)}
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(attempt) + "\n")
                if attempt.get("returncode") != 0:
                    raise RuntimeError(f"match failed; see {log} and {run}")
    result = validate_fault(clean_root, fault_root, seeds, layer)
    result["candidate"] = candidate
    result["candidate_sha256"] = digest
    (out / "screen.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--gate-log", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    args = parser.parse_args()
    result = run_grid(_read_json(args.seed_plan), args.gate_log, args.candidate, args.out, args.url, args.node)
    print(json.dumps({"out": str(args.out / "screen.json"), "retained": result["retained"]}))


if __name__ == "__main__":
    main()
