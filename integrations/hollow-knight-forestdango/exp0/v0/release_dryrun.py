"""Release-chain DRY RUN on a small full-HK collection (smoke / pilot data) with the fake (stub) diagnoser.

  python3 -B release_dryrun.py ROOT [--values card_values.json] [--skip-services]

ROOT is a collection root whose collection.json is a dry-run plan (contract 0.2.0 `dry_run: true`, e.g.
per_condition 1). Steps, each recorded in ROOT/dryrun/steps.jsonl with its command, exit code and seconds:
  1 validate_episode.py                       every attempt valid / incomplete / invalid
  2 probes.py selfcheck (HKKeysAdapter)       every logged request re-rendered from its saved state, byte for byte
  3 audit_set.py reask (8811, real calls)     weights evidence by same-request re-ask (<= 4 calls per weights attempt)
  4 audit_set.py audit                         evidence, selection (per_condition per condition), neutral ids
  5 diagnosis_inputs.py                        label-free read inputs
  6 probes.py capture x2 (8811, 8812) + compare   real probe calls on the first 4 decisions, identity check
  7 diagnose.py read / probe --backend stub    the FAKE diagnoser (no model, no network)
  8 score.py, fill_cards.py                    offline scores; cards filled with dry-run values
  9 bundle.py release --code-revision 0*40     the code is uncommitted local work: the revision is a placeholder
 10 tools/verify_release.py in a fresh venv without packages
 11 upload_release.py check                    must REFUSE the dry-run bundle
Nothing is uploaded, no Codex call is made, no GitHub action. With --skip-services steps 3 and 6 are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = Path(__file__).resolve().parents[3] / "_shared" / "exp0"
ADAPTER = f"{HERE / 'hk_adapter.py'}:HKKeysAdapter"
CLEAN, ALTERNATE = "http://127.0.0.1:8811", "http://127.0.0.1:8812"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--values", type=Path, default=HERE / "card_values_dryrun.json")
    ap.add_argument("--skip-services", action="store_true")
    a = ap.parse_args()
    root = a.root.resolve()
    plan = json.loads((root / "collection.json").read_text())
    if not plan.get("dry_run"):
        raise SystemExit("release_dryrun.py runs only on a dry-run plan")
    work = root / "dryrun"
    if work.exists():
        raise SystemExit(f"{work} exists; a dry run writes into a fresh directory")
    work.mkdir()
    py = sys.executable
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("HF_TOKEN_FILE", None)

    def step(name: str, cmd: list[str], check: bool = True, expect_fail: bool = False) -> subprocess.CompletedProcess:
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (work / f"{name}.out").write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""))
        row = {
            "step": name,
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.monotonic() - t0, 1),
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "stdout_tail": proc.stdout[-600:],
            "stderr_tail": proc.stderr[-600:],
        }
        with (work / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"{name}: exit {proc.returncode} ({row['seconds']} s) {proc.stdout.strip().splitlines()[-1][:200] if proc.stdout.strip() else ''}",
            flush=True,
        )
        if expect_fail:
            if proc.returncode == 0:
                raise SystemExit(f"{name} was expected to fail")
        elif check and proc.returncode:
            raise SystemExit(f"{name} failed: {proc.stderr[-800:]}")
        return proc

    step("01_validate", [py, "-B", str(COMMON / "validate_episode.py"), str(root)], check=False)
    step("02_selfcheck", [py, "-B", str(COMMON / "probes.py"), "selfcheck", str(root), "--adapter", ADAPTER])
    reask = []
    if not a.skip_services:
        step(
            "03_reask",
            [
                py,
                "-B",
                str(COMMON / "audit_set.py"),
                "reask",
                str(root),
                str(root / "weights-reask.jsonl"),
                "--endpoint",
                CLEAN,
            ],
        )
        reask = ["--reask", str(root / "weights-reask.jsonl")]
    step(
        "04_audit",
        [
            py,
            "-B",
            str(COMMON / "audit_set.py"),
            "audit",
            str(root),
            str(root / "audit"),
            *reask,
            "--adapter",
            ADAPTER,
        ],
    )
    index = root / "audit" / "episodes.json"
    step(
        "05_read_inputs",
        [py, "-B", str(COMMON / "diagnosis_inputs.py"), str(root), str(index), str(root / "read-inputs.jsonl")],
    )
    if not a.skip_services:
        step(
            "06a_capture_reference",
            [
                py,
                "-B",
                str(COMMON / "probes.py"),
                "capture",
                str(root),
                str(index),
                str(root / "probe-reference.jsonl"),
                "--endpoint",
                CLEAN,
                "--checkpoint-role",
                "clean",
                "--adapter",
                ADAPTER,
            ],
        )
        step(
            "06b_capture_alternate",
            [
                py,
                "-B",
                str(COMMON / "probes.py"),
                "capture",
                str(root),
                str(index),
                str(root / "probe-alternate.jsonl"),
                "--endpoint",
                ALTERNATE,
                "--checkpoint-role",
                "alternate",
                "--adapter",
                ADAPTER,
            ],
        )
        step(
            "06c_compare",
            [
                py,
                "-B",
                str(COMMON / "probes.py"),
                "compare",
                str(root),
                str(index),
                str(root / "probes"),
                "--reference-capture",
                str(root / "probe-reference.jsonl"),
                "--alternate-capture",
                str(root / "probe-alternate.jsonl"),
                "--adapter",
                ADAPTER,
            ],
        )
    for mode, extra in (("read", []), ("probe", ["--probes", str(root / "probes" / "probe-inputs.jsonl")])):
        step(
            f"07_diagnose_{mode}_STUB",
            [
                py,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(root / f"diagnosis-{mode}"),
                "--mode",
                mode,
                *extra,
                "--backend",
                "stub",
            ],
        )
    scores = step(
        "08a_score",
        [
            py,
            "-I",
            "-B",
            str(COMMON / "score.py"),
            str(index),
            str(root / "diagnosis-read" / "diagnoses.jsonl"),
            str(root / "diagnosis-probe" / "diagnoses.jsonl"),
        ],
    )
    (work / "scores.json").write_text(scores.stdout)
    step(
        "08b_cards",
        [
            py,
            "-B",
            str(COMMON / "fill_cards.py"),
            str(root),
            str(root / "audit"),
            str(work / "cards"),
            "--values",
            str(a.values),
            "--scores",
            str(work / "scores.json"),
        ],
    )
    out = work / "release" / plan["dataset_dir"].rstrip("/")
    step(
        "09_bundle_release",
        [
            py,
            "-B",
            str(COMMON / "bundle.py"),
            "release",
            str(root),
            str(root / "audit"),
            str(out),
            "--cards",
            str(work / "cards"),
            "--code-revision",
            "0" * 40,
            "--source",
            f"hk_v9={HERE}",
        ],
    )
    venv = work / "fresh-venv"
    step("10a_fresh_venv", [py, "-m", "venv", "--without-pip", str(venv)])
    listing = step(
        "10b_fresh_venv_packages",
        [
            str(venv / "bin" / "python"),
            "-I",
            "-c",
            "import sys, pkgutil; print(sorted(m.name for m in pkgutil.iter_modules() "
            "if 'site-packages' in str(getattr(m.module_finder, 'path', ''))))",
        ],
    )
    verified = step(
        "10c_verify_release",
        [str(venv / "bin" / "python"), "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)],
    )
    receipt = {
        **json.loads(verified.stdout),
        "fresh_venv_python": str(venv / "bin" / "python"),
        "fresh_venv_site_packages": listing.stdout.strip(),
        "diagnoser": "STUB (tests/stub_codex.py): no model",
        "code_revision": "placeholder 0*40 (uncommitted local code)",
        "uploaded": False,
    }
    (work / "verify-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    step(
        "11_upload_check_must_refuse",
        [py, "-B", str(COMMON / "upload_release.py"), "check", str(out)],
        expect_fail=True,
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
