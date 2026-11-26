"""Full Hollow Knight (ForestDango) Exp 0 v0: the release chain after the collection, as one documented driver over the
common tools (2026-11-26, Exp 0 v0 HK collection; the VVVVVV driver integrations/vvvvvv/exp0/v0/release_v0.py is the model).

  python3 -B release_v0.py prepare  ROOT [--skip-services] [--displays 370-373] [--parallel 4]
      validate -> probes selfcheck -> native replay of every attempt on the deterministic build (at most --parallel
      games, displays from --displays) -> weights re-ask (8811) -> audit (uses the native replays) -> read inputs ->
      label-isolation check -> probe captures (8811 clean, 8812 alternate) -> probe compare (identity check).
  python3 -B release_v0.py diagnose ROOT --backend stub|codex [--only read,probe,probe-variant]
      read and probe (the 80 primary calls) first, then the pre-registered probe variant (40 calls). A real run
      (`codex`) needs EXP0_DIAGNOSIS_APPROVED=yes (set by the caller after the user's approval).
  python3 -B release_v0.py cards    ROOT [--out DIR]
      fill the release documents (cards/*.md) from the files: plan, pins, attempts, audit index, checks, probes and,
      when the diagnosis has run, the saved replies' scores; the diagnosis is marked pending otherwise.
      Default output: the experiment's package/ directory next to ROOT.
  python3 -B release_v0.py bundle   ROOT --out RELEASE_ROOT --code-revision SHA40 [--cards DIR]
      scores (primary + variant) -> cards -> bundle.py release (text/JSON only: --exclude-extra; the variant) ->
      placeholders for local paths in records (privacy-check.json) -> privacy gate (no absolute path under the data,
      home or Users roots and no local account name anywhere in the bundle, code included) -> tools/verify_release.py
      in a fresh venv without packages -> privacy gate -> upload_release.py check (no token, no network; a dry-run plan
      must be refused). Nothing is uploaded and nothing is committed.
  python3 -B release_v0.py dataset  ROOT --out DIR
      the dataset-only bundle (no evaluation: bundle.py prepare, text/JSON only) -> bundle.py verify -> privacy gate.
  python3 -B release_v0.py resume   ROOT --out RELEASE_ROOT --code-revision SHA40
      ONE command for after the Codex usage limit: checks that the frozen prompts are unchanged and that `prepare` has
      run, then diagnose --backend codex (80 primary calls, then 40 variant calls) -> bundle (scores, cards, release
      bundle, local-path scrub and check, fresh-venv verify, upload check), in this order, stopping at the first
      failure. Needs EXP0_DIAGNOSIS_APPROVED=yes. Rerunning continues where it stopped (diagnose.py skips answered
      episodes; failed attempts are kept).

Re-bundle with the real code revision (after the code is committed; no model call):
  python3 -B integrations/hollow-knight-forestdango/exp0/v0/release_v0.py bundle <ROOT> --out <new release root> --code-revision $(git rev-parse HEAD)

Every step is appended to ROOT/release-steps.jsonl (command, exit code, seconds, output tails); full outputs go to
ROOT/release-steps/<step>.out. Standard library only (the re-render in selfcheck / probes needs Laya's tokenizer:
$HONING_LAYA_TOKENIZER, see runtime.py).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[2] / "_shared" / "exp0"
ISOLATION = (
    HERE.parents[2] / "_shared" / "exp0" / "isolation_check.py"
)  # game-agnostic (reads the plan and the inputs)
ADAPTER = f"{HERE / 'hk_adapter.py'}:HKKeysAdapter"
CLEAN, ALTERNATE = "http://127.0.0.1:8811", "http://127.0.0.1:8812"
VARIANT_PROMPT = COMMON / "diagnosis_prompt_probe_variant.txt"
PROMPTS = {
    "diagnosis_prompt.txt": "12ad488dbea2d7c11c0b40cf2f086d450088547b2148da5399fe0a2aced6d06b",
    "diagnosis_prompt_probe_variant.txt": "da848a845aa7fc24512a1ad7b2d0405efaf7090059e9decabe3a1c3052e60dd6",
}
TEMPLATES = HERE / "cards"
# the collector code a release snapshots (never the game build, the game source or the Demo session's reference copies)
SOURCES = {
    name: HERE / name
    for name in (
        "collect.py",
        "to_contract.py",
        "faults.py",
        "state_codec.py",
        "starts.py",
        "runtime.py",
        "hk_adapter.py",
        "replay_check.py",
        "replay_env.py",
        "native_replay.py",
        "offline_faults.py",
        "verify_starts.py",
        "pins.py",
        "release_jobs.py",
        "release_v0.py",
        "batch.py",
        "pins.json",
        "plan.json",
        "core",
        "starts",
        "cards",
    )
}
SOURCES["harness-v12_standing"] = HERE / "harness" / "v12_standing"
SOURCES["harness-v12_standing.SOURCE.json"] = HERE / "harness" / "v12_standing.SOURCE.json"
SOURCES["isolation_check.py"] = ISOLATION
COMPONENTS = ("harness", "weights", "controller")
CONDITIONS = ("no_injection", "harness", "weights", "controller")
PREFIX = {"no_injection": "CONTROL", "harness": "HARNESS", "weights": "WEIGHTS", "controller": "CONTROLLER"}
PENDING = (
    "Pending: the Codex CLI subscription used for the diagnosis hit its usage limit on 2026-11-26 (the VVVVVV "
    "collection's development call at 01:12-01:14 PST was answered with \"You've hit your usage limit ... try again "
    'at Nov 28th, 2026 11:57 PM"); no Codex call was made for this game. The inputs, probes, prompts and settings '
    "are frozen, and `release_v0.py resume` runs the 80 primary calls, the 40 variant calls and everything after them."
)
PLACEHOLDER_REVISION = "0" * 40
sys.path.insert(0, str(COMMON))


class Steps:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dir = root / "release-steps"
        self.dir.mkdir(exist_ok=True)
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        self.env.pop("HF_TOKEN_FILE", None)

    def run(
        self,
        name: str,
        cmd: list[str],
        *,
        check: bool = True,
        expect_fail: bool = False,
        env: dict[str, str] | None = None,
        bare_env: bool = False,
    ) -> subprocess.CompletedProcess:
        started = time.monotonic()
        # bare_env: a fresh environment (no API key, no token variable, no Python path), as upload_release.py's check
        environment = (
            {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C.UTF-8"}
            if bare_env
            else {**self.env, **(env or {})}
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, env=environment)
        (self.dir / f"{name}.out").write_text(
            proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
        )
        row = {
            "step": name,
            "cmd": cmd,
            "returncode": proc.returncode,
            "seconds": round(time.monotonic() - started, 1),
            "time_unix": time.time(),
            "stdout_tail": proc.stdout[-800:],
            "stderr_tail": proc.stderr[-800:],
        }
        with (self.root / "release-steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        last = proc.stdout.strip().splitlines()[-1][:240] if proc.stdout.strip() else ""
        print(f"{name}: exit {proc.returncode} ({row['seconds']} s) {last}", flush=True)
        if expect_fail:
            if proc.returncode == 0:
                raise SystemExit(f"{name} was expected to fail")
        elif check and proc.returncode:
            raise SystemExit(f"{name} failed: {proc.stderr[-1200:]}")
        return proc

    def note(self, row: dict) -> None:
        with (self.root / "release-steps.jsonl").open("a") as stream:
            stream.write(json.dumps({**row, "time_unix": time.time()}) + "\n")


def attempts_of(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and (p / "extra" / "manifest.json").exists() and (p / "manifest.json").exists()
    )


# --------------------------------------------------------------------------------------------- prepare / diagnose
def native_replays(root: Path, steps: Steps, displays: list[int], parallel: int) -> None:
    """native_replay.py on every attempt without a report (ROOT/native_replay/<run>.json); replays in ROOT/../replays/."""
    py = sys.executable
    (root / "native_replay").mkdir(exist_ok=True)
    work = root.parent / "replays" / root.name
    work.mkdir(parents=True, exist_ok=True)
    todo = [r for r in attempts_of(root) if not (root / "native_replay" / f"{r.name}.json").exists()]
    free = list(displays[:parallel])
    started = time.monotonic()
    results = []

    def one(run: Path, display: int) -> dict:
        out = work / run.name
        if out.exists():
            out = work / f"{run.name}-{int(time.time())}"
        proc = subprocess.run(
            [
                py,
                "-B",
                str(HERE / "native_replay.py"),
                str(run),
                "--out",
                str(out),
                "--display",
                str(display),
                "--report",
                str(root / "native_replay" / f"{run.name}.json"),
            ],
            capture_output=True,
            text=True,
            env=steps.env,
        )
        return {
            "run": run.name,
            "returncode": proc.returncode,
            "display": display,
            "tail": proc.stdout.strip()[-300:],
            "stderr_tail": proc.stderr.strip()[-300:],
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(free))) as pool:
        pending: dict = {}
        queue = list(todo)
        while queue or pending:
            while queue and free:
                pending[pool.submit(one, queue.pop(0), free.pop(0))] = None
            done, _ = concurrent.futures.wait(list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                res = fut.result()
                free.append(res["display"])
                results.append(res)
                print(json.dumps({k: res[k] for k in ("run", "returncode")}), flush=True)
    reports = [
        json.loads((root / "native_replay" / f"{r.name}.json").read_text())
        for r in attempts_of(root)
        if (root / "native_replay" / f"{r.name}.json").exists()
    ]
    summary = {
        "step": "03_native_replay",
        "replayed_now": len(results),
        "failed_now": [r["run"] for r in results if r["returncode"]],
        "reports": len(reports),
        "ok": sum(bool(r.get("ok")) for r in reports),
        "seconds": round(time.monotonic() - started, 1),
    }
    steps.note(summary)
    (steps.dir / "03_native_replay.out").write_text(json.dumps({**summary, "results": results}, indent=1) + "\n")
    print("03_native_replay:", json.dumps(summary), flush=True)
    if summary["failed_now"] or summary["ok"] != len(attempts_of(root)):
        raise SystemExit(f"native replay failed or incomplete: {summary}")


def prepare(root: Path, skip_services: bool, displays: list[int], parallel: int) -> None:
    py, steps = sys.executable, Steps(root)
    steps.run("01_validate", [py, "-B", str(COMMON / "validate_episode.py"), str(root)])
    steps.run("02_selfcheck", [py, "-B", str(COMMON / "probes.py"), "selfcheck", str(root), "--adapter", ADAPTER])
    native_replays(root, steps, displays, parallel)
    reask = []
    if not skip_services:
        steps.run(
            "04_reask",
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
    steps.run(
        "05_audit",
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
    steps.run(
        "06_read_inputs",
        [py, "-B", str(COMMON / "diagnosis_inputs.py"), str(root), str(index), str(root / "read-inputs.jsonl")],
    )
    if skip_services:
        return
    # the two captures use different services (8811, 8812) and run side by side
    captures = [
        (
            "07a_capture_reference",
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
        ),
        (
            "07b_capture_alternate",
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
        ),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for fut in [pool.submit(steps.run, name, cmd) for name, cmd in captures]:
            fut.result()
    steps.run(
        "07c_compare",
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
    steps.run("08_isolation_check", [py, "-B", str(ISOLATION), str(root), "--out", str(root / "isolation-check.json")])


def diagnose(root: Path, backend: str, only: list[str]) -> None:
    py, steps = sys.executable, Steps(root)
    runs = {
        "read": ["--mode", "read"],
        "probe": ["--mode", "probe", "--probes", str(root / "probes" / "probe-inputs.jsonl")],
        "probe-variant": [
            "--mode",
            "probe",
            "--probes",
            str(root / "probes" / "probe-inputs.jsonl"),
            "--prompt",
            str(VARIANT_PROMPT),
        ],
    }
    expected = len(json.loads((root / "audit" / "episodes.json").read_text()))
    for name in ("read", "probe", "probe-variant"):  # the 80 primary calls come first
        if name in only:
            steps.run(
                f"09_diagnose_{name}_{backend}",
                [
                    py,
                    "-B",
                    str(COMMON / "diagnose.py"),
                    str(root / "read-inputs.jsonl"),
                    str(root / f"diagnosis-{name}"),
                    *runs[name],
                    "--backend",
                    backend,
                ],
            )
            if not _complete(root, name, expected):
                raise SystemExit(
                    f"diagnosis-{name}: not every one of the {expected} episodes has a diagnosis; run the "
                    "same command again to continue (answered episodes are skipped)"
                )


# ------------------------------------------------------------------------------------------------------- cards
def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete(root: Path, name: str, episodes: int) -> bool:
    path = root / f"diagnosis-{name}" / "diagnoses.jsonl"
    return path.is_file() and len({row["episode_id"] for row in _rows(path)}) == episodes


def _pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else "n/a"


def _interval(value: dict) -> str:
    interval = value.get("interval_95")
    if not interval:
        return "n/a (fewer than two units)"
    return f"{interval[0] * 100:+.1f} to {interval[1] * 100:+.1f} percentage points over {value['groups']} units"


def _mode_line(name: str, part: dict) -> str:
    matrix = part["confusion_matrix"]
    per = "; ".join(
        f"{label}: {matrix[label][label]}/10 correct"
        + "".join(f", {count} {cause}" for cause, count in matrix[label].items() if cause != label and count)
        for label in COMPONENTS
    )
    controls = ", ".join(f"{count} {cause}" for cause, count in part["control_causes"].items() if count)
    return f"`{name}` — {per}. Controls: {controls}."


def _score_values(scores: dict, variant: dict | None, settings: dict) -> dict[str, str]:
    values: dict[str, str] = {}
    read, probe = scores["read"], scores["probe"]
    fp = {mode: sum(scores[mode]["control_causes"][c] for c in COMPONENTS) for mode in ("read", "probe")}
    usage = {mode: scores[mode]["reported_usage_totals"] for mode in ("read", "probe")}
    attempts = {mode: scores[mode]["attempts_total"] for mode in ("read", "probe")}
    values["DIAGNOSIS_INTRO"] = (
        f"Codex `{scores['model']}` evaluated the same 40 native episodes in both modes through `{settings['cli_version']}`. "
        "Each diagnosis used a fresh, tool-disabled session. The provider's immutable model revision and per-call dollar "
        f"charges were unavailable. The two modes used {attempts['read']} and {attempts['probe']} attempts for 40 episodes "
        "each (a second attempt only after an invalid reply); failed CLI attempts are kept in each mode's `attempts.json`."
    )
    rows = [
        ("Correct on verified interventions", f"{read['correct']}/30", f"{probe['correct']}/30"),
        ("Macro-F1 on verified interventions", f"{read['macro_f1']:.3f}", f"{probe['macro_f1']:.3f}"),
        (
            "Abstentions on verified interventions",
            f"{sum(r['uncertain'] for r in read['confusion_matrix'].values())}/30",
            f"{sum(r['uncertain'] for r in probe['confusion_matrix'].values())}/30",
        ),
        (
            "Invalid replies on verified interventions",
            f"{sum(r['invalid'] for r in read['confusion_matrix'].values())}/30",
            f"{sum(r['invalid'] for r in probe['confusion_matrix'].values())}/30",
        ),
        ("False positives on no-injection controls (any asserted component)", f"{fp['read']}/10", f"{fp['probe']}/10"),
        (
            "Reported input tokens",
            f"{usage['read'].get('input_tokens', 0):,}",
            f"{usage['probe'].get('input_tokens', 0):,}",
        ),
        (
            "Reported output tokens",
            f"{usage['read'].get('output_tokens', 0):,}",
            f"{usage['probe'].get('output_tokens', 0):,}",
        ),
    ]
    values["RESULTS_TABLE"] = "| Metric | Read | Probe |\n| --- | ---: | ---: |\n" + "\n".join(
        f"| {a} | {b} | {c} |" for a, b, c in rows
    )
    values["PER_CLASS_NARRATIVE"] = (
        _mode_line("read", read)
        + "\n\n"
        + _mode_line("probe", probe)
        + "\n\nA control false positive is any component asserted on a no-injection episode "
        "(#185: an asserted component fault without supporting audit evidence)."
    )
    paired, base = scores["paired_probe_minus_read"], scores["baselines"]
    values["PAIRED_AND_BASELINES"] = (
        f"The paired accuracy difference (probe minus read) on the 30 verified cases was {paired['estimate'] * 100:+.1f} "
        f"percentage points; the descriptive 95% bootstrap interval spans "
        f"{_interval(paired['scenario_reset_bootstrap'])} (reset groups) and {_interval(paired['fault_instance_bootstrap'])} "
        "(fault instances, one per component). The groups are starts on one route through one area, so these intervals "
        f"describe this sample only. Baselines on the verified cases: uniform random accuracy {base['uniform_expected_accuracy']:.3f} "
        f"(macro-F1 {base['uniform_expected_macro_f1']:.3f}), majority {base['majority_accuracy']:.3f}, each constant "
        "component 0.333 accuracy with 10/10 control false positives."
    )
    values["USAGE_SUMMARY"] = (
        f"`read` {usage['read'].get('input_tokens', 0):,} input and "
        f"{usage['read'].get('output_tokens', 0):,} output tokens; `probe` "
        f"{usage['probe'].get('input_tokens', 0):,} input and {usage['probe'].get('output_tokens', 0):,} "
        "output tokens; per-call dollars unavailable"
    )
    values["ATTRIBUTION_SUMMARY"] = (
        f"`read` {read['correct']}/30 and `probe` {probe['correct']}/30 on verified interventions"
    )
    values["CONTROLS_SUMMARY"] = (
        f"`read` {fp['read']}/10 and `probe` {fp['probe']}/10 false positives (any asserted component)"
    )
    if variant:
        vp = variant["probe"]
        vfp = sum(vp["control_causes"][c] for c in COMPONENTS)
        values["VARIANT_RESULTS"] = (
            "`probe` mode once more on the same 40 episodes with `diagnosis_prompt_probe_variant.txt` (the primary prompt "
            "plus one sentence: the reference-versus-alternate difference is not evidence by itself; only a logged "
            "probability that fails to match the reference replay and matches the alternate is). Scored with the same "
            f"`score.py` against the same `read` replies: {vp['correct']}/30 correct, macro-F1 {vp['macro_f1']:.3f}, "
            f"{sum(r['uncertain'] for r in vp['confusion_matrix'].values())}/30 abstentions, {vfp}/10 control false "
            f"positives; paired variant minus read {variant['paired_probe_minus_read']['estimate'] * 100:+.1f} percentage "
            "points. "
            + _mode_line("probe-variant", vp)
            + " This variant was pre-registered before any call; it is not "
            "the primary endpoint."
        )
        values["VARIANT_SUMMARY"] = (
            f"{vp['correct']}/30 correct, {vfp}/10 control false positives (reported separately)"
        )
    return values


def _model_rows(root: Path, run: str) -> list[dict]:
    return [r for r in _rows(root / run / "decisions.jsonl") if r["kind"] == "model"]


def _essential_diff(row: dict) -> tuple[list[str], list[str]]:
    """Sentences of the request's state lines that the clean render has and the logged request lacks, and back. The
    "On screen:" prefix of a line is set apart, so that removing the line's first sentence does not count as a change
    of the next one."""

    def sentences(request: dict) -> list[str]:
        out = []
        for i, line in enumerate(request["parts"]["essential"]):
            head = "On screen:" if line.startswith("On screen:") else ""
            body = line[len(head) :].strip()
            out += [f"{i}|{part.strip()}" for part in re.split(r"(?<=\.)\s+", body) if part.strip()]
        return out

    now, clean = sentences(row["request"]), sentences(row["clean_request"])
    return [s.split("|", 1)[1] for s in clean if s not in now], [s.split("|", 1)[1] for s in now if s not in clean]


def _sentence_kind(sentence: str) -> str:
    """'Ground', 'Ledges near you', 'Ceiling', 'Enemy', 'No enemy', ... (the words before a colon near the start, else
    the first word)."""
    if ":" in sentence[:25]:
        return sentence.split(":", 1)[0]
    words = sentence.split()
    return " ".join(words[:2]) if words and words[0] == "No" else (words[0] if words else "")


def card_values(root: Path) -> dict[str, str]:
    """Every number of the cards, recomputed from the collection root's files (and the scores when they exist)."""
    import release_metadata

    plan = json.loads((root / "collection.json").read_text())
    hk_plan = json.loads((HERE / "plan.json").read_text())
    index = json.loads((root / "audit" / "episodes.json").read_text())
    attempts = _rows(root / "audit" / "attempts.jsonl")
    per = {c: [row for row in index if row["condition"] == c] for c in CONDITIONS}
    results = {row["run"]: json.loads((root / row["run"] / "result.json").read_text()) for row in index}
    manifest = json.loads((root / index[0]["run"] / "manifest.json").read_text())
    task = hk_plan["tasks"][hk_plan["collection"]["task"]]
    values: dict[str, str] = {
        "PLAN_SHA": _sha(root / "collection.json"),
        "HK_PLAN_SHA": _sha(HERE / "plan.json"),
        "DECISION_CAP": str(task["caps"]["decisions"]),
        "FRAME_CAP": str(task["caps"]["frames"]),
        "ATTEMPT_CAP": str(plan["attempt_cap_per_slot"]),
        "TOLERANCE": str(plan["weight_difference_tolerance"]),
        "PROMPT_SHA": _sha(COMMON / "diagnosis_prompt.txt"),
        "VARIANT_PROMPT_SHA": _sha(VARIANT_PROMPT),
        "CODE_REVISION": "pending (filled by `release_v0.py bundle --code-revision`)",
        "HARNESS_FINGERPRINT": (
            manifest["pins"]["harness"].get("fingerprint", "")
            if isinstance(manifest["pins"].get("harness"), dict)
            else ""
        ),
    }
    values["FROZEN_AT"] = hk_plan.get("frozen_at", "")
    pins = json.loads((HERE / "pins.json").read_text())
    files = pins["game"]["files"]
    values["BUILD_SHA"] = files["HollowKnightDemo.x86_64"]["sha256"]
    values["CSHARP_SHA"] = files["HollowKnightDemo_Data/Managed/Assembly-CSharp.dll"]["sha256"]
    values["BRIDGE_VERSION"] = str(pins["game"].get("bridge_version"))
    values["TOKENIZER_SHA"] = hk_plan["release"]["tokenizer"]["sha256"]
    values["HARNESS_FINGERPRINT"] = pins["code"]["harness"]["fingerprint"]
    # groups: start, waypoint, clean outcome
    group_rows = []
    clean_by_group = {row["scenario_group"]: row for row in per["no_injection"]}
    for g in plan["groups"]:
        if g["scenario"].get("reserve"):
            continue
        w = g["scenario"]["waypoint"]
        clean = clean_by_group.get(g["id"])
        res = results.get(clean["run"]) if clean else None
        outcome = (
            "—"
            if res is None
            else (
                f"{res['status']} at decision {res['model_decisions']}"
                if res["status"] == "win"
                else f"{res['status']} ({res['reason']})"
            )
        )
        start = hk_plan["starts"][g["scenario"]["start"]]
        group_rows.append(
            f"| `{g['id']}` | {g['split']} | {start.get('steps', 0)} | ({w['place'][0]:.2f}, {w['place'][1]:.2f}), "
            f"{w['distance_from_start']} units, height {w['height_change']:+.2f} | {outcome} |"
        )
    values["GROUP_TABLE"] = (
        "| Group | Split | Prefix steps | Waypoint (place, distance, height change) | Clean outcome |\n"
        "| --- | --- | ---: | --- | --- |\n" + "\n".join(group_rows)
    )
    reserves = [g["id"] for g in plan["groups"] if g["scenario"].get("reserve")]
    values["RESERVE_GROUPS"] = ", ".join(f"`{g}`" for g in reserves)
    for condition, prefix in PREFIX.items():
        values[f"{prefix}_DECISIONS"] = f"{sum(results[row['run']]['model_decisions'] for row in per[condition]):,}"
        deaths = sum(1 for row in per[condition] if results[row["run"]]["status"] == "loss")
        values[f"{prefix}_DEATHS"] = str(deaths)
        values[f"{prefix}_AUTO"] = str(sum(results[row["run"]].get("auto_rows", 0) for row in per[condition]))
    values["AUTO_ROWS"] = f"{sum(results[row['run']].get('auto_rows', 0) for row in index):,}"
    initial = {group["id"] for group in plan["groups"][: plan["initial_groups"]]}
    reserve_attempts = [row for row in attempts if row["scenario_group"] not in initial]
    excluded = [row for row in attempts if row["annotation_status"] == "excluded"]
    values["RESERVE_NOTE"] = (
        "no reserve group was collected."
        if not reserve_attempts
        else f"reserve attempts: {', '.join(sorted(row['run'] for row in reserve_attempts))}."
    )
    values["COLLECTION_STATUS"] = (
        f"{len(attempts)} formal attempts, {len(excluded)} excluded, "
        f"{sum(row['selected_for_scoring'] for row in attempts)} selected; " + values["RESERVE_NOTE"]
    )
    # harness: what the fault removed
    harness = []
    removed_kinds, added_kinds, option_changes = Counter(), Counter(), 0
    for row in per["harness"]:
        model = _model_rows(root, row["run"])
        changed = [r for r in model if r.get("fault") and r["fault"].get("changed")]
        for r in changed:
            gone, new = _essential_diff(r)
            removed_kinds.update(_sentence_kind(s) for s in gone)
            added_kinds.update(_sentence_kind(s) for s in new)
            if r["request"]["questions"] != r["clean_request"]["questions"]:
                option_changes += 1
        harness.append((row["evidence"]["model_decision_number"], len(changed), len(model)))
    first = Counter(item[0] for item in harness)
    values["HARNESS_EVIDENCE_SUMMARY"] = (
        f"in the {len(harness)} harness episodes the first changed request is at model decision "
        f"{', '.join(f'{k} ({v} episodes)' for k, v in sorted(first.items()))}; "
        f"{sum(i[1] for i in harness):,} of their {sum(i[2] for i in harness):,} model requests differ from the clean render. "
        f"Sentences of the state text left out of the actual requests, by kind: "
        f"{', '.join(f'“{k}” {v}' for k, v in removed_kinds.most_common())}"
        + (
            f"; sentences only in the actual requests: {', '.join(f'“{k}” {v}' for k, v in added_kinds.most_common())}"
            if added_kinds
            else "; no sentence was added"
        )
        + f"; the questions, the option order and the option texts differ in {option_changes} requests."
    )
    # weights
    weights = [row["evidence"] for row in per["weights"]]
    positions = Counter(item["model_decision_number"] for item in weights)
    methods = Counter(item["method"] for item in weights)
    dps = []
    top_changed = 0
    for item in weights:
        c, a = item["clean_probabilities"], item["alternate_probabilities"]
        dps.append(max(abs(c[q][o] - a[q][o]) for q in c for o in c[q]))
        top_changed += any(max(c[q], key=c[q].get) != max(a[q], key=a[q].get) for q in c)
    values["WEIGHTS_TOP_CHANGED"] = str(top_changed)
    values["WEIGHTS_EVIDENCE_POSITIONS"] = "evidence at model decision " + ", ".join(
        f"{k} in {v} of {len(weights)} weights episodes" for k, v in sorted(positions.items())
    )
    values["WEIGHTS_EVIDENCE_SUMMARY"] = (
        f"{', '.join(f'{v} by `{k}`' for k, v in methods.items())}; {values['WEIGHTS_EVIDENCE_POSITIONS']}; the largest "
        f"option probability difference at the evidence row is {min(dps):.4f}–{max(dps):.4f} (over both questions), and "  # noqa: RUF001
        f"the most probable option of at least one question changed in {top_changed} of {len(weights)}."
    )
    reask_rows = _rows(root / "weights-reask.jsonl") if (root / "weights-reask.jsonl").is_file() else []
    if reask_rows:
        values["WEIGHTS_EVIDENCE_SUMMARY"] += (
            f" The re-ask record holds {sum(len(r['rows']) for r in reask_rows)} clean-service replies to the evidence "
            f"requests of {len(reask_rows)} weights attempts."
        )
    # controller
    controller = []
    for row in per["controller"]:
        model = _model_rows(root, row["run"])
        swapped = [r for r in model if r["executed"]["option"]["move"] != r["model_choice"]["move"]]
        directional = [r for r in model if any(k in r["model_choice"]["move"] for k in ("left", "right"))]
        controller.append((row["evidence"]["model_decision_number"], len(swapped), len(directional), len(model)))
    cfirst = Counter(item[0] for item in controller)
    values["CONTROLLER_EVIDENCE_SUMMARY"] = (
        f"the first swapped decision is model decision {', '.join(f'{k} ({v} episodes)' for k, v in sorted(cfirst.items()))}; "
        f"{sum(i[1] for i in controller):,} of the {sum(i[2] for i in controller):,} directional selections "
        f"({sum(i[3] for i in controller):,} model decisions) were executed with the other direction; every changed window "
        "keeps the selected keys as `intended_action`."
    )
    # outcomes against the paired clean episode
    clean_status = {row["scenario_group"]: results[row["run"]]["status"] for row in per["no_injection"]}
    pairing = []
    for condition in COMPONENTS:
        same = sum(
            1 for row in per[condition] if clean_status.get(row["scenario_group"]) == results[row["run"]]["status"]
        )
        pairing.append(f"{condition} {same}/{len(per[condition])}")
    values["OUTCOME_PAIRING_SUMMARY"] = (
        "Same native status as the paired clean episode of the group: " + ", ".join(pairing) + "."
    )
    clean_choices = {
        row["scenario_group"]: [r["model_choice"] for r in _model_rows(root, row["run"])]
        for row in per["no_injection"]
    }
    follow = []
    for condition in COMPONENTS:
        firsts, identical = [], 0
        for row in per[condition]:
            mine = [r["model_choice"] for r in _model_rows(root, row["run"])]
            other = clean_choices.get(row["scenario_group"])
            if other is None:
                continue
            n = next(
                (i for i, (a, b) in enumerate(zip(mine, other)) if a != b), min(len(mine), len(other))  # noqa: B905
            )
            if n == len(mine) == len(other):
                identical += 1
            else:
                firsts.append(n)
        firsts.sort()
        span = (
            f"{firsts[0]}"
            if firsts and firsts[0] == firsts[-1]
            else f"{firsts[0]}–{firsts[-1]} (median {firsts[len(firsts) // 2]})" if firsts else ""  # noqa: RUF001
        )
        follow.append(
            f"{condition}: {identical} of {len(per[condition])} identical to the end"
            + (f", the others first differ at model decision {span}" if firsts else "")
        )
    values["FOLLOW_SUMMARY"] = (
        "Selections compared with the paired clean episode's, decision by decision (same start): "
        + "; ".join(follow)
        + "."
    )
    # pilots against the release's clean controls (determinism across sessions and code changes)
    pilots = root.parent / "pilots"
    replica = []
    for row in per["no_injection"]:
        start = next(g["scenario"]["start"] for g in plan["groups"] if g["id"] == row["scenario_group"])
        pdir = pilots / f"pilot-{start}" / "extra" / "session" / "requests"
        mine = root / row["run"] / "extra" / "session" / "requests"
        if pdir.is_dir() and mine.is_dir():
            files = sorted(mine.iterdir())
            replica.append(
                sum((pdir / f.name).is_file() and (pdir / f.name).read_bytes() == f.read_bytes() for f in files)
                == len(files)
            )
    values["PILOT_REPLICA"] = (
        (
            f"{sum(replica)} of {len(replica)} clean controls sent exactly the requests of their group's "
            "development pilot (same bytes, decision by decision, up to the control's end)"
        )
        if replica
        else "not checked"
    )
    # checks
    steps = root / "release-steps"
    stdout = lambda name: (steps / f"{name}.out").read_text().split("\n--- stderr ---\n")[0].strip()  # noqa: E731
    selfcheck = json.loads(stdout("02_selfcheck").splitlines()[-1])
    validation = stdout("01_validate").splitlines()[-1]
    reports = [json.loads(p.read_text()) for p in sorted((root / "native_replay").glob("*.json"))]
    n_ok = sum(bool(r.get("ok")) for r in reports)
    totals = Counter()
    names = {
        "debug_state_per_window": "debug",
        "visible_observation_per_window": "visible",
        "state_per_frame": "frames",
    }
    for r in reports:
        for item in r.get("results", []):
            part = names.get(item.get("what"))
            if part:
                totals[f"{part}_equal"] += item.get("identical") or 0
                totals[f"{part}_total"] += item.get("rows_attempt") or 0
                if part == "frames":
                    totals["ngo"] += (
                        len(item.get("object_count_nGo_differs_in_frames") or [])
                        if isinstance(item.get("object_count_nGo_differs_in_frames"), list)
                        else (item.get("object_count_nGo_differs_in_frames") or 0)
                    )
    values["CHECKS_SUMMARY"] = (
        f"Validator over all attempts: `{validation}`. Probe self-check: {selfcheck['decisions']:,} logged requests of "
        f"{selfcheck['runs']} attempts re-rendered byte for byte from their saved states, {selfcheck['mismatches']} mismatches. "
        f"Native replay on the deterministic build (a new game process fed only the attempt's recorded native input, "
        f"from the boot on): {n_ok} of {len(reports)} attempts identical in every debug state per window "
        f"({totals['debug_equal']:,}/{totals['debug_total']:,}), every visible observation ({totals['visible_equal']:,}/"
        f"{totals['visible_total']:,}) and every per-frame state record ({totals['frames_equal']:,}/{totals['frames_total']:,}; "
        f"the object count `nGo` left out, it differed in {totals['ngo']} frames). {values['PILOT_REPLICA']}."
    )
    isolation = root / "isolation-check.json"
    if isolation.is_file():
        iso = json.loads(isolation.read_text())
        uniform = sorted(
            {f"`{m['marker']}` ({m['file']}, all {m['episodes']} episodes)" for m in iso.get("uniform_markers", [])}
        )
        values["ISOLATION_SUMMARY"] = (
            f"{'passed' if iso.get('ok') else 'FAILED'}: {len(iso.get('forbidden_hits', []))} "
            "forbidden markers (fault family and instance names with '-', '_' and ' ' equal, run "
            "and group-condition names, checkpoint repos, revisions and SHA-256s, any 64-hex "
            "digest, label field names); uniform markers: " + ("; ".join(uniform) or "none")
        )
    else:
        values["ISOLATION_SUMMARY"] = "not run"
    identity = _rows(root / "probes" / "identity.jsonl")
    decided = sum(item["decisions"] for item in identity)
    worst = max(item["max_difference"] for item in identity)
    values["IDENTITY_SUMMARY"] = (
        f"{sum(item['ok'] for item in identity)} of {len(identity)} episodes passed "
        f"({decided} decisions, {sum(item['questions'] for item in identity)} question records; "
        f"largest difference {worst:.4f}, tolerance 0.01, tie 0.01)."
    )
    values["IDENTITY_SHORT"] = f"{decided} decisions, largest difference {worst:.4f}"
    summary = release_metadata.weight_probe_summary(
        root, index, plan["weight_difference_tolerance"], plan["question_keys"]
    )
    by = summary["by_collection_condition"]
    values["WEIGHT_PROBE_SUMMARY"] = (
        "; ".join(
            f"{condition}: {item['differences_above_tolerance']}/{item['decisions']} probed decisions above 0.01, mean "
            f"largest option difference {item['mean_max_option_difference']:.3f}, top option changed in "
            f"{item['most_probable_option_changed']}"
            for condition, item in by.items()
        )
        + "."
    )
    values["KNOWN_LIMITS"] = "Known limits and what was not done are listed in [TODO.md](TODO.md)."
    # diagnosis
    settings_path = root / "diagnosis-read" / "settings.json"
    if (
        _complete(root, "read", len(index))
        and _complete(root, "probe", len(index))
        and (root / "scores.json").is_file()
    ):
        settings = json.loads(settings_path.read_text())
        scores = json.loads((root / "scores.json").read_text())
        variant = (
            json.loads((root / "scores-probe-variant.json").read_text())
            if _complete(root, "probe-variant", len(index)) and (root / "scores-probe-variant.json").is_file()
            else None
        )
        values.update(_score_values(scores, variant, settings))
        values["CLI_VERSION"] = f"`{settings['cli_version']}`"
        values["DIAGNOSIS_STATUS"] = "Completed; results in [RESULTS.md](RESULTS.md)."
        values["DIAGNOSIS_BOX"] = "x"
        values["NOT_DONE_EXTRA"] = "" if variant else "The variant run is incomplete."
        if not variant:
            values["VARIANT_RESULTS"] = "Pending: the variant run has not completed (see `release-steps.jsonl`)."
            values["VARIANT_SUMMARY"] = "pending"
    else:
        values.update(
            {
                "CLI_VERSION": "`codex-cli 0.167.0` (the version the VVVVVV collection checked on 2026-11-26)",
                "DIAGNOSIS_STATUS": PENDING,
                "DIAGNOSIS_BOX": " ",
                "DIAGNOSIS_INTRO": "**The diagnosis has not run yet.** " + PENDING,
                "RESULTS_TABLE": "| Metric | Read | Probe |\n| --- | ---: | ---: |\n| Correct on verified interventions | pending | pending |\n"
                "| Macro-F1 on verified interventions | pending | pending |\n"
                "| False positives on no-injection controls (any asserted component) | pending | pending |",
                "PER_CLASS_NARRATIVE": "Per-class results, confusion matrices and calibration: pending (no scored call yet).",
                "PAIRED_AND_BASELINES": (
                    "Baselines that need no diagnosis (balanced 10/10/10 verified cases): uniform random "
                    "accuracy 0.333, majority 0.333, each constant component 0.333 accuracy with 10/10 "
                    "control false positives. The paired probe-minus-read difference is pending."
                ),
                "VARIANT_RESULTS": "Pending, for the same reason; it runs after the 80 primary calls.",
                "VARIANT_SUMMARY": "pending (Codex usage limit)",
                "USAGE_SUMMARY": "pending",
                "ATTRIBUTION_SUMMARY": "pending (Codex usage limit)",
                "CONTROLS_SUMMARY": "pending",
                "NOT_DONE_EXTRA": "The real diagnosis is pending (Codex usage limit).",
            }
        )
    return values


def cards(root: Path, out: Path, code_revision: str | None = None, steps: Steps | None = None) -> Path:
    values = card_values(root)
    if code_revision:
        values["CODE_REVISION"] = f"`{code_revision}`" + (
            " (placeholder: the code is not committed yet)" if code_revision == PLACEHOLDER_REVISION else ""
        )
    values_file = root / "card-values.json"
    values_file.write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    scores = ["--scores", str(root / "scores.json")] if (root / "scores.json").is_file() else []
    (steps or Steps(root)).run(
        "10_cards",
        [
            sys.executable,
            "-B",
            str(COMMON / "fill_cards.py"),
            str(root),
            str(root / "audit"),
            str(out),
            "--values",
            str(values_file),
            *scores,
            "--templates",
            str(TEMPLATES),
        ],
    )
    return out


# ----------------------------------------------------------------------------------------------------- privacy
# 2026-11-26 (main session): nothing that is published (package docs, adapter, release bundle) may hold an absolute
# local path or the local account name. Code has none (run-time resolution: runtime.py); records lose the services'
# local `path` field in bundle.py; any other local path or the account name found in a released record is replaced by
# a documented placeholder below, and the check stops the chain before the fresh verify and the upload check if
# anything remains anywhere (code included).
LOCAL_PATH = re.compile(r"(?<![\w.<])(?:/data|/home|/Users)/[A-Za-z0-9_.~-]+")
CODE_DIRS = ("collector-source", "common-source", "tools", "schemas")
PLACEHOLDERS = {
    "<workspace>": "the local checkout of this repository",
    "<home>": "the local home directory",
    "<conda-root>": "the local conda installation",
    "<user>": "the local account name",
    "<local-path>": "any other absolute local path",
}


def _accounts() -> set[str]:
    names = {Path.home().name, getpass.getuser()}
    return {name for name in names if name and len(name) >= 3}


def _roots() -> list[tuple[str, str]]:
    """Local path prefixes (both the literal and the resolved form) with their placeholders, longest first."""
    pairs: dict[str, str] = {}
    for path, placeholder in (
        (HERE.parents[3], "<workspace>"),
        (Path.home(), "<home>"),
        (Path(sys.base_prefix), "<conda-root>"),
    ):
        for form in {str(path), str(path.resolve())}:
            pairs.setdefault(form, placeholder)
    return sorted(pairs.items(), key=lambda item: -len(item[0]))


def privacy_findings(out: Path) -> list[str]:
    """Every absolute local path or local account name in the given tree (code included)."""
    accounts = [re.compile(re.escape(name), re.IGNORECASE) for name in _accounts()]
    found = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [m.group(0) for m in LOCAL_PATH.finditer(text)] + [
            m.group(0) for rx in accounts for m in rx.finditer(text)
        ]
        if hits:
            found.append(f"{path.relative_to(out).as_posix()}: {len(hits)} (first: {hits[0][:40]!r})")
    return found


def sanitize_records(out: Path) -> dict:
    """Replace local paths and the account name in released RECORDS (not code) by placeholders; rewrite sha256.json."""
    import bundle as common_bundle

    accounts = [re.compile(re.escape(name), re.IGNORECASE) for name in _accounts()]
    changed: dict[str, int] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        relative = path.relative_to(out).as_posix()
        if relative.split("/")[0] in CODE_DIRS or relative == "sha256.json":
            continue
        text = path.read_text(encoding="utf-8")
        new, count = text, 0
        for prefix, placeholder in _roots():
            count += new.count(prefix)
            new = new.replace(prefix, placeholder)
        for rx in accounts:
            new, n = rx.subn("<user>", new)
            count += n
        new, n = LOCAL_PATH.subn("<local-path>", new)
        count += n
        if count:
            path.write_text(new, encoding="utf-8")
            changed[relative] = count
    report = {
        "schema": "honing-exp0-privacy-check-v1",
        "placeholders": PLACEHOLDERS,
        "rule": (
            "no absolute local path and no local account name in any released file; the decision services' "
            "local `path` fields are removed by bundle.py; other occurrences in records are replaced by the "
            "placeholders; code holds none (paths are resolved at run time)"
        ),
        "records_changed": changed,
    }
    (out / "privacy-check.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    hashes = common_bundle.file_hashes(out)
    (out / "sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def code_privacy_findings() -> list[str]:
    """Local paths or the account name in the collector code the bundle copies (bundle.py skips __pycache__)."""
    accounts = [re.compile(re.escape(name), re.IGNORECASE) for name in _accounts()]
    found = []
    for name, source in SOURCES.items():
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        for path in files:
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = [m.group(0) for m in LOCAL_PATH.finditer(text)] + [
                m.group(0) for rx in accounts for m in rx.finditer(text)
            ]
            if hits:
                found.append(f"{name}/{path.name}: {len(hits)} (first: {hits[0][:40]!r})")
    return found


def privacy_gate(out: Path, steps: Steps, name: str) -> None:
    findings = privacy_findings(out)
    steps.note({"step": name, "findings": findings, "accounts_checked": len(_accounts())})
    print(f"{name}: {len(findings)} files with a local path or the account name", flush=True)
    if findings:
        raise SystemExit(
            f"{name}: local paths or the account name remain, stopping before the upload check: {findings[:5]}"
        )


# ------------------------------------------------------------------------------------------------------ bundle
def bundle(root: Path, release_root: Path, code_revision: str, cards_dir: Path | None) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", code_revision):
        raise SystemExit("--code-revision must be a 40-hex commit SHA")
    py, steps = sys.executable, Steps(root)
    plan = json.loads((root / "collection.json").read_text())
    if release_root.exists() and any(release_root.iterdir()):
        raise SystemExit(f"{release_root} is not empty; a release is built into a new directory")
    # the code that goes into the bundle holds no local path (checked before anything is built)
    code_findings = code_privacy_findings()
    steps.note({"step": "11a_code_privacy", "findings": code_findings})
    if code_findings:
        raise SystemExit(f"local paths in the collector code: {code_findings[:5]}")
    index = root / "audit" / "episodes.json"
    scores = steps.run(
        "11b_score",
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
    (root / "scores.json").write_text(scores.stdout)
    variant = root / "diagnosis-probe-variant"
    variant_flags = []
    if (variant / "diagnoses.jsonl").is_file():
        again = steps.run(
            "11c_score_variant",
            [
                py,
                "-I",
                "-B",
                str(COMMON / "score.py"),
                str(index),
                str(root / "diagnosis-read" / "diagnoses.jsonl"),
                str(variant / "diagnoses.jsonl"),
            ],
        )
        (root / "scores-probe-variant.json").write_text(again.stdout)
        variant_flags = ["--variant", f"probe-variant={variant}"]
    # the package documents next to the collection root are regenerated with this revision (the PR's docs)
    cards_dir = cards_dir or cards(root, root.parent / "package", code_revision, steps)
    out = release_root / plan["dataset_dir"].rstrip("/")
    sources = [part for name, path in SOURCES.items() for part in ("--source", f"hk-{name}={path}")]
    steps.run(
        "12_bundle_release",
        [
            py,
            "-B",
            str(COMMON / "bundle.py"),
            "release",
            str(root),
            str(root / "audit"),
            str(out),
            "--cards",
            str(cards_dir),
            "--code-revision",
            code_revision,
            "--exclude-extra",
            *sources,
            *variant_flags,
        ],
    )
    sanitized = sanitize_records(out)
    print(
        f"12b_sanitize_records: {sum(sanitized['records_changed'].values())} replacements in "
        f"{len(sanitized['records_changed'])} record files",
        flush=True,
    )
    steps.note({"step": "12b_sanitize_records", "records_changed": sanitized["records_changed"]})
    privacy_gate(out, steps, "12c_privacy_gate")
    venv = release_root.parent / f"{release_root.name}-fresh-venv"
    steps.run("13a_fresh_venv", [py, "-m", "venv", "--without-pip", str(venv)])
    listing = steps.run(
        "13b_fresh_venv_packages",
        [
            str(venv / "bin" / "python"),
            "-I",
            "-c",
            "import sys, pkgutil; print(sorted(m.name for m in pkgutil.iter_modules() "
            "if 'site-packages' in str(getattr(m.module_finder, 'path', ''))))",
        ],
    )
    verified = steps.run(
        "13c_verify_release",
        [str(venv / "bin" / "python"), "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)],
        bare_env=True,
    )
    receipt = {
        **json.loads(verified.stdout),
        "fresh_venv_python_version": subprocess.check_output([str(venv / "bin" / "python"), "-V"], text=True).strip(),
        "fresh_venv_site_packages": listing.stdout.strip(),
        "code_revision": code_revision,
        "code_revision_is_placeholder": code_revision == PLACEHOLDER_REVISION,
        "uploaded": False,
    }
    (release_root.parent / f"{release_root.name}-verify-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    privacy_gate(out, steps, "13d_privacy_gate")
    steps.run(
        "14_upload_check",
        [
            py,
            "-B",
            str(COMMON / "upload_release.py"),
            "check",
            str(out),
            "--work",
            str(release_root.parent / f"{release_root.name}-upload-check"),
        ],
        expect_fail=bool(plan.get("dry_run")),
        check=not plan.get("dry_run"),
    )
    print(json.dumps(receipt, indent=1))


def dataset_bundle(root: Path, out_root: Path) -> None:
    """The dataset-only bundle (no evaluation; bundle.py prepare, text/JSON only) + bundle.py verify + privacy scan."""
    py, steps = sys.executable, Steps(root)
    plan = json.loads((root / "collection.json").read_text())
    out = out_root / plan["dataset_dir"].rstrip("/")
    if out.exists():
        raise SystemExit(f"{out} exists; a bundle is built into a new directory")
    sources = [part for name, path in SOURCES.items() for part in ("--source", f"hk-{name}={path}")]
    steps.run(
        "D1_dataset_bundle",
        [
            py,
            "-B",
            str(COMMON / "bundle.py"),
            "prepare",
            str(root),
            str(root / "audit"),
            str(out),
            "--exclude-extra",
            *sources,
        ],
    )
    steps.run("D2_dataset_verify", [py, "-B", str(COMMON / "bundle.py"), "verify", str(out)])
    privacy_gate(out, steps, "D3_dataset_privacy_gate")


def resume_checks(root: Path) -> None:
    for name, digest in PROMPTS.items():
        if _sha(COMMON / name) != digest:
            raise SystemExit(f"{name} is not the frozen prompt (sha256 {digest[:12]}...)")
    for need in (
        "audit/episodes.json",
        "read-inputs.jsonl",
        "probes/probe-inputs.jsonl",
        "probes/identity.jsonl",
        "isolation-check.json",
    ):
        if not (root / need).exists():
            raise SystemExit(f"{need} is missing: run `release_v0.py prepare` first")
    identity = _rows(root / "probes" / "identity.jsonl")
    if not identity or not all(item["ok"] for item in identity):
        raise SystemExit("the probe identity check did not pass")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--skip-services", action="store_true")
    p.add_argument("--displays", default="370-373")
    p.add_argument("--parallel", type=int, default=4)
    d = sub.add_parser("diagnose")
    d.add_argument("root", type=Path)
    d.add_argument("--backend", choices=("stub", "codex"), required=True)
    d.add_argument("--only", default="read,probe,probe-variant")
    c = sub.add_parser("cards")
    c.add_argument("root", type=Path)
    c.add_argument("--out", type=Path)
    ds = sub.add_parser("dataset")
    ds.add_argument("root", type=Path)
    ds.add_argument(
        "--out", type=Path, required=True, help="a new directory; the bundle is <out>/hollow-knight-forestdango/v0"
    )
    for name in ("bundle", "resume"):
        b = sub.add_parser(name)
        b.add_argument("root", type=Path)
        b.add_argument(
            "--out",
            type=Path,
            required=True,
            help="a new release root; the bundle is <out>/hollow-knight-forestdango/v0",
        )
        b.add_argument("--code-revision", required=True)
        if name == "bundle":
            b.add_argument("--cards", type=Path, help="filled cards to use instead of regenerating them")
    a = ap.parse_args()
    root = a.root.resolve()
    if a.command == "prepare":
        lo, hi = (int(v) for v in a.displays.split("-"))
        if lo < 350 or hi > 379:
            raise SystemExit("displays must be within :350-:379")
        prepare(root, a.skip_services, list(range(lo, hi + 1)), a.parallel)
    elif a.command == "diagnose":
        if a.backend == "codex" and os.environ.get("EXP0_DIAGNOSIS_APPROVED") != "yes":
            raise SystemExit("a real diagnosis needs EXP0_DIAGNOSIS_APPROVED=yes (the user's approval)")
        diagnose(root, a.backend, a.only.split(","))
    elif a.command == "cards":
        print(cards(root, (a.out or root.parent / "package").resolve()))
    elif a.command == "dataset":
        dataset_bundle(root, a.out.resolve())
    elif a.command == "bundle":
        bundle(root, a.out.resolve(), a.code_revision, a.cards.resolve() if a.cards else None)
    else:
        if os.environ.get("EXP0_DIAGNOSIS_APPROVED") != "yes":
            raise SystemExit("resume makes real Codex calls: set EXP0_DIAGNOSIS_APPROVED=yes (the user's approval)")
        if not re.fullmatch(r"[0-9a-f]{40}", a.code_revision):
            raise SystemExit("--code-revision must be a 40-hex commit SHA")
        resume_checks(root)
        diagnose(root, "codex", ["read", "probe", "probe-variant"])
        bundle(root, a.out.resolve(), a.code_revision, None)


if __name__ == "__main__":
    main()
