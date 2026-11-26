"""VVVVVV Exp 0 v0: the release chain after the collection, as one documented driver over the common tools.

  python3 -B release_v0.py prepare  ROOT [--skip-services]
      validate -> probes selfcheck -> weights re-ask (8813) -> audit (native replay of every attempt) -> read
      inputs -> probe captures (8813 clean, 8814 alternate) -> probe compare (identity check).
  python3 -B release_v0.py diagnose ROOT --backend stub|codex [--only read,probe,probe-variant]
      read and probe (the 80 primary calls) first, then the pre-registered probe variant (40 calls). A real run
      (`codex`) needs EXP0_DIAGNOSIS_APPROVED=yes (set by the caller after the user's approval).
  python3 -B release_v0.py cards    ROOT [--out DIR]
      fill the release documents (cards/*.md) from the files: plan, pins, attempts, audit index, checks, probes and,
      when the diagnosis has run, the saved replies' scores; the diagnosis is marked pending otherwise.
      Default output: the experiment's package/ directory next to ROOT.
  python3 -B release_v0.py bundle   ROOT --out RELEASE_ROOT --code-revision SHA40 [--cards DIR]
      scores (primary + variant) -> cards (regenerated into ROOT/../package unless --cards is given) ->
      bundle.py release (text/JSON only: --exclude-extra; the variant) -> placeholders for local paths in records
      (privacy-check.json) -> privacy gate (no absolute path under the data, home or Users roots and no local account
      name anywhere in the bundle) ->
      tools/verify_release.py in a fresh venv without packages -> upload_release.py check (no token, no network;
      a dry-run plan must be refused). Nothing is uploaded and nothing is committed.
  python3 -B release_v0.py resume   ROOT --out RELEASE_ROOT --code-revision SHA40
      ONE command for after the Codex usage limit: diagnose --backend codex (80 primary calls, then 40 variant
      calls) -> bundle (scores, cards, release bundle, fresh-venv verify, upload check), in this order, stopping at
      the first failure. Needs EXP0_DIAGNOSIS_APPROVED=yes. Rerunning continues where it stopped (diagnose.py skips
      answered episodes; failed attempts are kept).

Re-bundle with the real code revision (after the code is committed; no model call):
  python3 -B integrations/vvvvvv/exp0/v0/release_v0.py bundle <ROOT> --out <new release root> --code-revision $(git rev-parse HEAD)

Every step is appended to ROOT/release-steps.jsonl (command, exit code, seconds, output tails); full outputs go to
ROOT/release-steps/<step>.out. Standard library only.
"""

from __future__ import annotations

import argparse
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
ADAPTER = f"{HERE / 'adapter.py'}:ADAPTER"
CLEAN, ALTERNATE = "http://127.0.0.1:8813", "http://127.0.0.1:8814"
VARIANT_PROMPT = COMMON / "diagnosis_prompt_probe_variant.txt"
TEMPLATES = HERE / "cards"
# the collector code a release snapshots (never the game source or build)
SOURCES = {
    "vvvvvv-collector": HERE / "exp0_vvvvvv.py",
    "vvvvvv-adapter": HERE / "adapter.py",
    "vvvvvv-release-driver": HERE / "release_v0.py",
    "vvvvvv-fault-checks": HERE / "fault_checks.py",
    "vvvvvv-card-templates": HERE / "cards",
    "vvvvvv-pins": HERE / "pins.json",
    "vvvvvv-plans": HERE / "plans",
    "vvvvvv-integration": HERE.parents[1],
    "vvvvvv-shared-receiver": HERE.parents[2] / "_shared" / "decision.py",
    "smallgames-procs": HERE / "procs.py",
    "smallgames-collect-set": HERE / "collect_set.py",
    "smallgames-cut-video": HERE / "cut_video.py",
}
COMPONENTS = ("harness", "weights", "controller")
CONDITIONS = ("no_injection", "harness", "weights", "controller")
PREFIX = {"no_injection": "CONTROL", "harness": "HARNESS", "weights": "WEIGHTS", "controller": "CONTROLLER"}
PENDING = (
    "Pending: the Codex CLI subscription used for the diagnosis answered the first call of 2026-11-26 01:12 PST, a "
    "development check outside the scored set, with \"You've hit your usage limit ... try again at Nov 28th, 2026 "
    '11:57 PM"; no scored diagnosis call has been made. The inputs, probes, prompts and settings are frozen, and '
    "`release_v0.py resume` runs the 80 primary calls, the 40 variant calls and everything after them."
)
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


# --------------------------------------------------------------------------------------------- prepare / diagnose
def prepare(root: Path, skip_services: bool) -> None:
    py, steps = sys.executable, Steps(root)
    steps.run("01_validate", [py, "-B", str(COMMON / "validate_episode.py"), str(root)])
    steps.run("02_selfcheck", [py, "-B", str(COMMON / "probes.py"), "selfcheck", str(root), "--adapter", ADAPTER])
    reask = []
    if not skip_services:
        steps.run(
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
    replays = root.parent / "replays" / f"{root.name}-audit"
    replays.mkdir(parents=True, exist_ok=True)
    steps.run(
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
        env={"EXP0_REPLAY_WORK_ROOT": str(replays)},
    )
    index = root / "audit" / "episodes.json"
    steps.run(
        "05_read_inputs",
        [py, "-B", str(COMMON / "diagnosis_inputs.py"), str(root), str(index), str(root / "read-inputs.jsonl")],
    )
    if skip_services:
        return
    steps.run(
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
    steps.run(
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
    steps.run(
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
    for name in ("read", "probe", "probe-variant"):  # the 80 primary calls come first
        if name in only:
            steps.run(
                f"07_diagnose_{name}_{backend}",
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
        "(fault instances, one per component). The groups are policy seeds of one layout, so these intervals describe this "
        f"sample only. Baselines on the verified cases: uniform random accuracy {base['uniform_expected_accuracy']:.3f} "
        f"(macro-F1 {base['uniform_expected_macro_f1']:.3f}), majority {base['majority_accuracy']:.3f}, each constant "
        "component 0.333 accuracy with 10/10 control false positives."
    )
    values["READ_CORRECT"], values["PROBE_CORRECT"] = str(read["correct"]), str(probe["correct"])
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
            + " This variant was pre-registered in the brief before any call; "
            "it is not the primary endpoint."
        )
        values["VARIANT_SUMMARY"] = (
            f"{vp['correct']}/30 correct, {vfp}/10 control false positives (reported separately)"
        )
    return values


def card_values(root: Path) -> dict[str, str]:
    """Every number of the cards, recomputed from the collection root's files (and the scores when they exist)."""
    import release_metadata

    plan = json.loads((root / "collection.json").read_text())
    index = json.loads((root / "audit" / "episodes.json").read_text())
    attempts = _rows(root / "audit" / "attempts.jsonl")
    per = {c: [row for row in index if row["condition"] == c] for c in CONDITIONS}
    results = {row["run"]: json.loads((root / row["run"] / "result.json").read_text()) for row in index}
    manifest = json.loads((root / index[0]["run"] / "manifest.json").read_text())
    values: dict[str, str] = {
        "BINARY_SHA": manifest["pins"]["game_build"]["sha256"],
        "ASSETS_SHA": manifest["pins"]["game_build"]["assets_sha256"],
        "PLAN_SHA": _sha(root / "collection.json"),
        "DECISION_CAP": str(plan["groups"][0]["scenario"]["caps"]["decisions"]),
        "ATTEMPT_CAP": str(plan["attempt_cap_per_slot"]),
        "TOLERANCE": str(plan["weight_difference_tolerance"]),
        "PROMPT_SHA": _sha(COMMON / "diagnosis_prompt.txt"),
        "VARIANT_PROMPT_SHA": _sha(VARIANT_PROMPT),
        "CODE_REVISION": "pending (filled by `release_v0.py bundle --code-revision`)",
    }
    for condition, prefix in PREFIX.items():
        values[f"{prefix}_DEATHS"] = str(sum(results[row["run"]]["deaths"] for row in per[condition]))
        values[f"{prefix}_DECISIONS"] = f"{sum(results[row['run']]['model_decisions'] for row in per[condition]):,}"
    values["AUTO_ROWS"] = f"{sum(results[row['run']]['auto_rows'] for row in index):,}"
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
    # harness
    harness = []
    for row in per["harness"]:
        rows = _rows(root / row["run"] / "decisions.jsonl")
        model = [r for r in rows if r["kind"] == "model"]
        changed = [r for r in model if r["fault"] and r["fault"]["changed"]]
        fields = set()
        for r in changed:
            now, clean = json.loads(r["request"]["parts"]["essential"][1]), json.loads(
                r["clean_request"]["parts"]["essential"][1]
            )
            fields |= {k for k in set(now) | set(clean) if now.get(k) != clean.get(k)}
            if (
                r["request"]["questions"] != r["clean_request"]["questions"]
                or r["request"]["parts"]["essential"][0] != r["clean_request"]["parts"]["essential"][0]
            ):
                fields.add("questions or rule text")
        harness.append((row["evidence"]["model_decision_number"], len(changed), len(model), fields))
    first = Counter(item[0] for item in harness)
    values["HARNESS_EVIDENCE_SUMMARY"] = (
        f"in the {len(harness)} harness episodes the first changed request is at model decision "
        f"{', '.join(f'{k} ({v} episodes)' for k, v in sorted(first.items()))}; "
        f"{sum(i[1] for i in harness):,} of their {sum(i[2] for i in harness):,} model requests differ from the clean render, "
        f"and only in {', '.join(f'`{f}`' for f in sorted(set().union(*(i[3] for i in harness))))}."
    )
    # weights
    weights = [row["evidence"] for row in per["weights"]]
    positions = Counter(item["model_decision_number"] for item in weights)
    methods = Counter(item["method"] for item in weights)
    flip = [(item["clean_probabilities"].get("flip"), item["alternate_probabilities"].get("flip")) for item in weights]
    values["WEIGHTS_TOP_CHANGED"] = str(
        sum(
            max(item["clean_probabilities"], key=item["clean_probabilities"].get)
            != max(item["alternate_probabilities"], key=item["alternate_probabilities"].get)
            for item in weights
        )
    )
    values["WEIGHTS_EVIDENCE_POSITIONS"] = "evidence at model decision " + ", ".join(
        f"{k} in {v} of {len(weights)} weights episodes" for k, v in sorted(positions.items())
    )
    values["WEIGHTS_EVIDENCE_SUMMARY"] = (
        f"{', '.join(f'{v} by `{k}`' for k, v in methods.items())}; at the evidence row p(`flip`) was "
        f"{min(f[0] for f in flip):.4f}–{max(f[0] for f in flip):.4f} under the clean checkpoint and "  # noqa: RUF001
        f"{min(f[1] for f in flip):.4f}–{max(f[1] for f in flip):.4f} under the alternate."  # noqa: RUF001
    )
    evidence_requests = {
        json.dumps(
            _rows(root / row["run"] / "decisions.jsonl")[row["evidence"]["index"]]["request"],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in per["weights"]
    }
    if len(evidence_requests) == 1 and len(weights) > 1:
        values["WEIGHTS_EVIDENCE_SUMMARY"] += (
            f" All {len(weights)} evidence rows are the same request: every reset group starts from the same native state, "
            "so the first request (and its two probability vectors) is identical in every group; the weights evidence "
            "therefore rests on one request."
        )
    reask_rows = _rows(root / "weights-reask.jsonl") if (root / "weights-reask.jsonl").is_file() else []
    if reask_rows:
        values["WEIGHTS_EVIDENCE_SUMMARY"] += (
            f" The re-ask record holds {sum(len(r['rows']) for r in reask_rows)} clean-service replies to the first requests "
            f"of {len(reask_rows)} weights attempts, each differing by "
            f"{min(r['rows'][-1]['max_probability_difference'] for r in reask_rows):.4f}–"  # noqa: RUF001
            f"{max(r['rows'][-1]['max_probability_difference'] for r in reask_rows):.4f} from the logged alternate reply."
        )
    choices = Counter()
    for row in per["weights"]:
        choices.update(r["model_choice"] for r in _rows(root / row["run"] / "decisions.jsonl") if r["kind"] == "model")
    values["WEIGHTS_FLIP_SHARE"] = _pct(choices["flip"], sum(choices.values()))
    # controller
    controller = []
    for row in per["controller"]:
        rows = _rows(root / row["run"] / "decisions.jsonl")
        model = [r for r in rows if r["kind"] == "model"]
        swapped = [r for r in model if r["executed"]["option"] != r["model_choice"]]
        echo = all(r["extra"]["after_observation"]["lastAction"] == r["executed"]["option"] for r in model)
        controller.append((row["evidence"]["model_decision_number"], len(swapped), len(model), echo))
    cfirst = Counter(item[0] for item in controller)
    values["CONTROLLER_EVIDENCE_SUMMARY"] = (
        f"the first swapped decision is model decision {', '.join(f'{k} ({v} episodes)' for k, v in sorted(cfirst.items()))}; "
        f"{sum(i[1] for i in controller):,} of {sum(i[2] for i in controller):,} model decisions were sent mirrored "
        f"(every directional selection); the bridge echoed the sent input at every decision: "
        f"{'yes' if all(i[3] for i in controller) else 'NO'}."
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
    # how long a changed episode's selections follow its clean partner's (same seed, same start)
    clean_choices = {
        row["scenario_group"]: [
            r["model_choice"] for r in _rows(root / row["run"] / "decisions.jsonl") if r["kind"] == "model"
        ]
        for row in per["no_injection"]
    }
    follow = []
    for condition in COMPONENTS:
        firsts, identical = [], 0
        for row in per[condition]:
            mine = [r["model_choice"] for r in _rows(root / row["run"] / "decisions.jsonl") if r["kind"] == "model"]
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
        follow.append(
            f"{condition}: {identical} of {len(per[condition])} identical to the end"
            + (
                f", the others first differ at model decision {firsts[0]}–{firsts[-1]} (median "  # noqa: RUF001
                f"{firsts[len(firsts) // 2]})"
                if firsts
                else ""
            )
        )
    values["FOLLOW_SUMMARY"] = (
        "Selections compared with the paired clean episode's, decision by decision (same seed, "
        "same start): " + "; ".join(follow) + "."
    )
    values["OUTCOME_PAIRING_SUMMARY"] += " " + values["FOLLOW_SUMMARY"]
    # checks
    steps = root / "release-steps"
    stdout = lambda name: (steps / f"{name}.out").read_text().split("\n--- stderr ---\n")[0].strip()  # noqa: E731
    selfcheck = json.loads(stdout("02_selfcheck").splitlines()[-1])
    validation = stdout("01_validate").splitlines()[-1]
    replays = [row["replay"] for row in attempts if row.get("replay")]
    receiver = root.parent / "checks" / "collection_fault_checks.json"
    receiver_text = ""
    if receiver.is_file():
        checked = json.loads(receiver.read_text())
        accepted = sum(item["receiver_accepted"] for item in checked.values())
        total = sum(item["model_rows"] for item in checked.values())
        receiver_text = (
            f" The integration's own receiver accepted {accepted:,} of {total:,} executed replies of all attempts."
        )
    values["CHECKS_SUMMARY"] = (
        f"Validator over all attempts: `{validation}`. Probe self-check: {selfcheck['decisions']:,} logged requests of "
        f"{selfcheck['runs']} attempts re-rendered byte for byte, {selfcheck['mismatches']} mismatches. Native replay "
        f"(fresh game process, logged inputs only): {sum(r['verified'] for r in replays)} of {len(replays)} attempts "
        f"reproduced every logged observation and the outcome ({sum(r.get('native_steps', 0) for r in replays):,} native "
        "windows)." + receiver_text
    )
    identity = _rows(root / "probes" / "identity.jsonl")
    decided = sum(item["decisions"] for item in identity)
    worst = max(item["max_difference"] for item in identity)
    values["IDENTITY_SUMMARY"] = (
        f"{sum(item['ok'] for item in identity)} of {len(identity)} episodes passed "
        f"({decided} decisions; largest difference {worst:.4f}, tolerance 0.01, tie 0.01)."
    )
    values["IDENTITY_SHORT"] = f"{decided} decisions, largest difference {worst:.4f}"
    summary = release_metadata.weight_probe_summary(
        root, index, plan["weight_difference_tolerance"], plan["question_key"]
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
    values["PILOT_EVIDENCE"] = (
        "pilot group `p01` (seed 9001), all four attempts valid: harness — all 150 requests differ from the clean render, "
        "only in `localSolidRows`, from decision 0; weights — decision 0's request identical to the clean partner's, p(`flip`) "
        ".3403 → .9911, the clean service's re-ask equal to the clean partner; controller — all 80 directional selections "
        "sent mirrored from decision 1, the bridge echoed every sent input; native replay 4/4. The clean pilot repeated "
        "phase 1's clean `p01` on all 156 rows."
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
                "CLI_VERSION": "`codex-cli 0.167.0` (version check of 2026-11-26)",
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
            " (placeholder: the code is not committed yet)" if code_revision == "0" * 40 else ""
        )
    values_file = root / "card-values.json"
    values_file.write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    scores = ["--scores", str(root / "scores.json")] if (root / "scores.json").is_file() else []
    (steps or Steps(root)).run(
        "08c_cards",
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
# local path or the local account name. Code has none (run-time resolution: runtime.py, diagnose.py); records lose the
# services' local `path` field in bundle.py; any other local path or the account name found in a released record is
# replaced by a documented placeholder below, and the check stops the chain before the fresh verify and the upload
# check if anything remains anywhere (code included).
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
    """Every absolute local path or local account name in the bundle (code included)."""
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
    sys.path.insert(0, str(COMMON))
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


def privacy_gate(out: Path, steps: Steps, name: str) -> None:
    findings = privacy_findings(out)
    row = {"step": name, "findings": findings, "accounts_checked": len(_accounts()), "time_unix": time.time()}
    with (steps.root / "release-steps.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(f"{name}: {len(findings)} files with a local path or the account name", flush=True)
    if findings:
        raise SystemExit(
            f"{name}: local paths or the account name remain, stopping before the upload check: " f"{findings[:5]}"
        )


# ------------------------------------------------------------------------------------------------------ bundle
def bundle(root: Path, release_root: Path, code_revision: str, cards_dir: Path | None) -> None:
    py, steps = sys.executable, Steps(root)
    plan = json.loads((root / "collection.json").read_text())
    if release_root.exists() and any(release_root.iterdir()):
        raise SystemExit(f"{release_root} is not empty; a release is built into a new directory")
    index = root / "audit" / "episodes.json"
    scores = steps.run(
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
    (root / "scores.json").write_text(scores.stdout)
    variant = root / "diagnosis-probe-variant"
    variant_flags = []
    if (variant / "diagnoses.jsonl").is_file():
        again = steps.run(
            "08b_score_variant",
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
    sources = [part for name, path in SOURCES.items() for part in ("--source", f"{name}={path}")]
    steps.run(
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
        f"09b_sanitize_records: {sum(sanitized['records_changed'].values())} replacements in "
        f"{len(sanitized['records_changed'])} record files",
        flush=True,
    )
    privacy_gate(out, steps, "09c_privacy_gate")
    venv = release_root.parent / f"{release_root.name}-fresh-venv"
    steps.run("10a_fresh_venv", [py, "-m", "venv", "--without-pip", str(venv)])
    listing = steps.run(
        "10b_fresh_venv_packages",
        [
            str(venv / "bin" / "python"),
            "-I",
            "-c",
            "import sys, pkgutil; print(sorted(m.name for m in pkgutil.iter_modules() "
            "if 'site-packages' in str(getattr(m.module_finder, 'path', ''))))",
        ],
    )
    verified = steps.run(
        "10c_verify_release",
        [str(venv / "bin" / "python"), "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)],
        bare_env=True,
    )
    receipt = {
        **json.loads(verified.stdout),
        "fresh_venv_python": str(venv / "bin" / "python"),
        "fresh_venv_python_version": subprocess.check_output([str(venv / "bin" / "python"), "-V"], text=True).strip(),
        "fresh_venv_site_packages": listing.stdout.strip(),
        "code_revision": code_revision,
        "code_revision_is_placeholder": code_revision == "0" * 40,
        "uploaded": False,
    }
    (release_root.parent / f"{release_root.name}-verify-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    privacy_gate(out, steps, "10d_privacy_gate")
    steps.run(
        "11_upload_check",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--skip-services", action="store_true")
    d = sub.add_parser("diagnose")
    d.add_argument("root", type=Path)
    d.add_argument("--backend", choices=("stub", "codex"), required=True)
    d.add_argument("--only", default="read,probe,probe-variant")
    c = sub.add_parser("cards")
    c.add_argument("root", type=Path)
    c.add_argument("--out", type=Path)
    for name in ("bundle", "resume"):
        b = sub.add_parser(name)
        b.add_argument("root", type=Path)
        b.add_argument("--out", type=Path, required=True, help="a new release root; the bundle is <out>/vvvvvv/v0")
        b.add_argument("--code-revision", required=True)
        if name == "bundle":
            b.add_argument("--cards", type=Path, help="filled cards to use instead of regenerating them")
    a = ap.parse_args()
    root = a.root.resolve()
    if a.command == "prepare":
        prepare(root, a.skip_services)
    elif a.command == "diagnose":
        diagnose(root, a.backend, a.only.split(","))
    elif a.command == "cards":
        print(cards(root, (a.out or root.parent / "package").resolve()))
    elif a.command == "bundle":
        bundle(root, a.out.resolve(), a.code_revision, a.cards.resolve() if a.cards else None)
    else:
        if os.environ.get("EXP0_DIAGNOSIS_APPROVED") != "yes":
            raise SystemExit("resume makes real Codex calls: set EXP0_DIAGNOSIS_APPROVED=yes (the user's approval)")
        diagnose(root, "codex", ["read", "probe", "probe-variant"])
        bundle(root, a.out.resolve(), a.code_revision, None)


if __name__ == "__main__":
    main()
