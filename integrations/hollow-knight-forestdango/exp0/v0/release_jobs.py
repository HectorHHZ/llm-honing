"""Job lists of the formal full-HK Exp 0 v0 collection from the FROZEN plan (2026-11-26, Exp 0 v0 HK collection).

  python3 -B release_jobs.py --root ROOT [--groups g01,g02,...] [--attempt 1] [--displays 350-369]

Writes ROOT/collection.json (to_contract.write_frozen_plan) unless it exists (it must then be byte-identical to a fresh
render) and one job file per decision service: ROOT/jobs/<attempt>_<port>.json. The clean-model conditions
(no_injection, harness, controller) of a group run on one clean service; groups alternate between the plan's two
clean services (8811 / 8820); the weights condition runs on the alternate service (8812). Job names:
<group>-<condition>-a<attempt> (private run ids, never shown to a model). Each job has its own display.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import to_contract

ORDER = ("no_injection", "harness", "controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=HERE / "plan.json")
    ap.add_argument("--groups", default=None, help="comma-separated group ids (default: the initial groups)")
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--displays", default="350-369")
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    if not str(plan.get("status", "")).startswith("FROZEN"):
        raise SystemExit("the plan is not frozen")
    root = a.root.resolve()
    coll = root / "collection.json"
    if coll.exists():
        old = coll.read_bytes()
        tmp = root / ".collection.check"
        tmp.mkdir(exist_ok=True)
        new = to_contract.write_frozen_plan(tmp, a.plan).read_bytes()
        (tmp / "collection.json").unlink()
        tmp.rmdir()
        if old != new:
            raise SystemExit("ROOT/collection.json differs from the frozen plan")
    else:
        to_contract.write_frozen_plan(root, a.plan)
    groups = (
        [g for g in plan["groups"] if not g.get("reserve")]
        if not a.groups
        else [g for g in plan["groups"] if g["id"] in a.groups.split(",")]
    )
    spec = plan["collection"]
    lo, hi = (int(v) for v in a.displays.split("-"))
    pool = list(range(lo, hi + 1))
    clean = [plan["endpoints"]["clean"], plan["endpoints"]["clean_2"]]
    jobs: dict[str, list] = {}
    for i, g in enumerate(groups):
        url = clean[i % 2]
        for condition in ORDER:
            job = {
                "name": f"{g['id']}-{condition}-a{a.attempt}",
                "task": spec["task"],
                "group": g["id"],
                "condition": condition,
                "endpoint": url,
                "label": f"Exp 0 v0 formal attempt {a.attempt}",
            }
            if condition != "no_injection":
                job["instance"] = spec["faults"][condition]
            jobs.setdefault(url.rsplit(":", 1)[1], []).append(job)
        jobs.setdefault(plan["endpoints"]["alternate"].rsplit(":", 1)[1], []).append(
            {
                "name": f"{g['id']}-weights-a{a.attempt}",
                "task": spec["task"],
                "group": g["id"],
                "condition": "weights",
                "instance": spec["faults"]["weights"],
                "endpoint": plan["endpoints"]["alternate"],
                "label": f"Exp 0 v0 formal attempt {a.attempt}",
            }
        )
    (root / "jobs").mkdir(exist_ok=True)
    per = max(1, len(pool) // max(1, len(jobs)))
    for n, (port, js) in enumerate(sorted(jobs.items())):
        mine = pool[n * per : (n + 1) * per]  # each service's jobs run one after another: a few displays each
        for k, job in enumerate(js):
            job["display"] = mine[k % len(mine)]
        path = root / "jobs" / f"a{a.attempt}_{port}.json"
        if path.exists():
            raise SystemExit(f"{path} exists")
        path.write_text(json.dumps(js, indent=1) + "\n")
        print(path, len(js))


if __name__ == "__main__":
    main()
