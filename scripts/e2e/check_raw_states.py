"""Check that recorded tank rounds re-render, replay and take teacher labels from their raw states.

Record on a CPU and check (from the repository root):
    python scripts/e2e/check_raw_states.py --record runs/raw-states --seeds 2001,2002,2003,2004

plays every seed from both seats with the tank dry run's stand-in decision model
(``tests/fakes/tank.py``) against the planner, and the planner teacher's round
of each scenario, then checks them. To check rounds recorded elsewhere, such as
a NanoJev round on a GPU, run this on the machine that recorded them, since
replay is bit exact only on the recording platform:
    python scripts/e2e/check_raw_states.py RUN...

Every run is checked four ways. Each decision re-renders under its logged
harness to its logged contextParts. The round replays from its seed with every
decision's state hash as logged, and then its events and result. The planner
teacher labels every question, and on the teacher's own rounds the labels are
its recorded choices. On model states that a teacher round also reached (the
same seat and state hash), the labels match the teacher's choices there. One
JSON summary goes to stdout, with the bytes the raw states add to each run
directory. It exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # the stand-in model lives under tests/

from honing.backends.tank.raw_states import STATES_FILE, TankRawStates, read_labels, read_states  # noqa: E402
from honing.core import ContractError  # noqa: E402

INTEGRATION = ROOT / "integrations/tank-war"
TEACHER = "teacher-demonstration"


def record(out: Path, seeds: Sequence[int], seconds: float) -> list[Path]:
    """The run directories of every seed from both seats, played by the stand-in model and by the teacher."""
    from honing.backends.jev.server import DecisionServer
    from tests.fakes.tank import START_QUALITY, ForecastModel

    server = DecisionServer(("127.0.0.1", 0), ForecastModel(START_QUALITY))
    serving = threading.Thread(target=server.serve, daemon=True)
    serving.start()
    out = out.absolute()  # match.mjs runs in the integration directory
    out.mkdir(parents=True)
    jobs = []
    for seed in seeds:
        for seat in ("blue", "red"):
            for agent in ("model", "teacher"):
                run = out / f"s{seed}-{seat}-{agent}"
                source = "--teacher" if agent == "teacher" else f"--decision-server={server.url}"
                jobs.append((run, [f"--seed={seed}", f"--seat={seat}", f"--seconds={seconds:g}", source]))

    def play(job: tuple[Path, list[str]]) -> Path:
        run, arguments = job
        command = ["node", str(INTEGRATION / "scripts/match.mjs"), *arguments, f"--out={run}"]
        process = subprocess.run(command, cwd=INTEGRATION, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise ContractError(f"{run.name} exited {process.returncode}: {process.stderr.strip()[-500:]}")
        return run

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(play, jobs))
    finally:
        server.stop()
        serving.join()
        server.server_close()


def size(run: Path) -> dict[str, Any]:
    """The bytes the raw states add: states.jsonl and the decision field of each decisions.jsonl row."""
    files = [p for p in run.iterdir() if p.is_file() and p.name != "labels.jsonl"]
    total = sum(p.stat().st_size for p in files)
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line]
    fields = sum(len(f',"decision":{row["decision"]}') for row in rows)
    extra = (run / STATES_FILE).stat().st_size + fields
    return {"run_bytes": total, "extra_bytes": extra, "extra_percent": round(100 * extra / total, 1)}


def _kind(run: Path) -> str:
    return str(json.loads((run / "manifest.json").read_text())["kind"])


def teacher_agreement(runs: Sequence[Path]) -> dict[str, int]:
    """Labels on model states that equal a teacher round's state, against the teacher's recorded choices there.

    A state is the same when its seat and its hash are: both seats start a seed in the same game state.
    """
    teacher: dict[tuple[int, str], dict[str, str]] = {}
    for run in (run for run in runs if _kind(run) == TEACHER):
        _, states = read_states(run)
        rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line]
        for state in states:
            chosen = {k: v for row in rows if row["decision"] == state["decision"] for k, v in row["chosen"].items()}
            teacher[state["player"], state["state_sha256"]] = {**chosen, "asked": state["motion"]}
    shared = compared = matched = 0
    for run in (run for run in runs if _kind(run) != TEACHER):
        _, states = read_states(run)
        _, labels = read_labels(run)
        for state in states:
            choice = teacher.get((state["player"], state["state_sha256"]))
            if choice is None:
                continue
            shared += 1
            for line in (line for line in labels if line["decision"] == state["decision"]):
                for head, label in line["labels"].items():
                    # The teacher chose the weapon for its own motion, so compare where the model asked about it too.
                    if head == "weapon" and state["motion"] != choice["asked"]:
                        continue
                    compared += 1
                    matched += label == choice[head]
    return {"shared_states": shared, "compared": compared, "matched": matched}


def check(runs: Sequence[Path], tool: TankRawStates) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    for trip in tool.round_trip(runs):
        entries[trip["run"]] = {"decisions": trip["decisions"], "rows": trip["rows"]}
    for run in runs:
        entry = entries[str(run)]
        entry["kind"] = _kind(run)
        middle = tool.replay(run, to=entry["decisions"] // 2)
        entry["replayed"] = [middle["checked"], tool.replay(run)["checked"]]
        entry.update(size(run))
    for labels in tool.label(runs):
        entry = entries[labels["run"]]
        entry["questions_labelled"] = labels["questions"]
        if "recorded" in labels:
            entry["teacher_rows_matched"] = labels["recorded"]["matched"]
    every = list(entries.values())
    totals = {
        "runs": len(every),
        "decisions": sum(e["decisions"] for e in every),
        "model_decisions": sum(e["decisions"] for e in every if e["kind"] != TEACHER),
        "rows_round_tripped": sum(e["rows"] for e in every),
        "questions_labelled": sum(e["questions_labelled"] for e in every),
        "extra_bytes_per_round": round(sum(e["extra_bytes"] for e in every) / len(every)),
        "extra_percent_of_run_dirs": round(
            100 * sum(e["extra_bytes"] for e in every) / sum(e["run_bytes"] for e in every), 1
        ),
    }
    return {"runs": entries, "totals": totals, "teacher_agreement": teacher_agreement(runs)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run directories written by match.mjs")
    parser.add_argument("--record", type=Path, help="record rounds into this new directory first")
    parser.add_argument("--seeds", default="2001,2002,2003,2004", help="comma-separated seeds for --record")
    parser.add_argument("--seconds", type=float, default=30.0, help="round length for --record")
    args = parser.parse_args(argv)
    runs = list(args.runs)
    try:
        if args.record:
            runs += record(args.record, [int(s) for s in args.seeds.split(",")], args.seconds)
        if not runs:
            parser.error("name run directories or --record")
        report = check(runs, TankRawStates(INTEGRATION))
    except ContractError as error:
        print(f"raw states do not reproduce: {error}", file=sys.stderr)
        return 1
    agreement = report["teacher_agreement"]
    print(json.dumps(report, indent=2))
    return 0 if agreement["compared"] == agreement["matched"] else 1


if __name__ == "__main__":
    sys.exit(main())
