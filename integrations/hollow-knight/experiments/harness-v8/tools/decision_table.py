# fmt: off
# isort: skip_file
# ruff: noqa
"""Per-decision table of a finished run: what the harness offered, what it withheld and why, what the
decision model picked — and a summary of how many decisions were settled by menu rules vs by the model.

  python3 decision_table.py <run_dir> [--md out.md] [--json out.json]

<run_dir> is a run directory (with session/) or a directory holding the trace files directly, plain or gzipped
(the committed trace under data/eval_laya_v8_L1L2 is read that way).

Definitions used in the summary (stated here so the numbers can be checked):
  * way-on options: every offered behaviour except `retreat_left` and `wait` (those two never lead on);
    options the loop guard has moved to the end ("Tried Nx here ...") do not count as way-on either.
  * settled by the menu: the decision had at most one way-on option on the menu. The model could then
    only take it or stall (step back / wait).
  * model's choice: two or more way-on options were on the menu; the model's pick decided which one.
  * rank: position of the pick in the menu as it was listed to the chooser.
  * harness rank: position of the pick in the harness's own ranking (1 = the harness's first-ranked option).
    The two differ only in runs with --menu-order reversed / shuffled.
Evaluation only: reads decisions.jsonl / skills.jsonl of the run; nothing is fed back to the harness.
"""
import argparse
import collections
import gzip
import json
import pathlib

STALL = {"retreat_left", "wait"}

ap = argparse.ArgumentParser()
ap.add_argument("run", type=pathlib.Path)
ap.add_argument("--md", type=pathlib.Path)
ap.add_argument("--json", type=pathlib.Path)
a = ap.parse_args()
session = a.run / "session" if (a.run / "session").is_dir() else a.run


def rows_of(name: str) -> list:
    plain, packed = session / name, session / (name + ".gz")
    if plain.exists():
        text = plain.read_text()
    elif packed.exists():
        text = gzip.decompress(packed.read_bytes()).decode()
    else:
        return []
    return [json.loads(l) for l in text.splitlines() if l.strip()]


decisions = rows_of("decisions.jsonl")
skills = {k["decision"]: k for k in rows_of("skills.jsonl")}

rows = []
for d in decisions:
    if "choice" not in d:
        continue
    offered = d["offered"]
    guard = d.get("loop_guard") or {}
    noted = {n for n, g in guard.items() if g["mechanism"] == "language"}
    way_on = [o for o in offered if o not in STALL and o not in noted]
    withheld = d.get("withheld") or {}
    harness_order = d.get("harness_order") or offered
    k = skills.get(d["decision"], {})
    ev = k.get("eval_only", {})
    rows.append({
        "decision": d["decision"], "n_offered": len(offered), "offered": offered, "way_on": way_on,
        "withheld": withheld, "choice": d["choice"], "rank": offered.index(d["choice"]) + 1,
        "harness_order": harness_order, "harness_rank": harness_order.index(d["choice"]) + 1,
        "p": d.get("probability"), "p_first": (d.get("probabilities") or {}).get(offered[0]),
        "settled_by": "menu" if len(way_on) <= 1 else "model",
        "pick_kind": "stall" if d["choice"] in STALL else ("noted" if d["choice"] in noted else "way_on"),
        "result": k.get("reason", ""), "windows": k.get("windows"), "x": ev.get("x"), "y": ev.get("y"),
        "hp": ev.get("health"), "scene": ev.get("scene"), "loop_guard": guard,
    })

n = len(rows)
menu_rows = [r for r in rows if r["settled_by"] == "menu"]
model_rows = [r for r in rows if r["settled_by"] == "model"]
rules = collections.Counter()
for r in rows:
    for name, rule in r["withheld"].items():
        rules[rule.split(":")[0]] += 1
summary = {
    "run": str(a.run), "decisions": n,
    "options_offered_mean": round(sum(r["n_offered"] for r in rows) / max(1, n), 2),
    "options_offered_min_max": [min(r["n_offered"] for r in rows), max(r["n_offered"] for r in rows)] if rows else None,
    "decisions_with_something_withheld": sum(1 for r in rows if r["withheld"]),
    "withheld_by_rule": dict(rules.most_common()),
    "settled_by_menu(<=1 way-on option)": len(menu_rows),
    "  of those: model took the way-on option": sum(1 for r in menu_rows if r["pick_kind"] == "way_on"),
    "  of those: model stalled (retreat/wait/noted option)": sum(1 for r in menu_rows if r["pick_kind"] != "way_on"),
    "model_choice(>=2 way-on options)": len(model_rows),
    "  of those: picked the harness's first-ranked option": sum(1 for r in model_rows if r["harness_rank"] == 1),
    "  of those: picked another way-on option": sum(1 for r in model_rows if r["harness_rank"] != 1 and r["pick_kind"] == "way_on"),
    "  of those: stalled": sum(1 for r in model_rows if r["pick_kind"] != "way_on"),
    "menus_listed_in_harness_order": sum(1 for r in rows if r["offered"] == r["harness_order"]),
    "picked_rank_1_overall": sum(1 for r in rows if r["rank"] == 1),
    "rank_histogram": dict(sorted(collections.Counter(r["rank"] for r in rows).items())),
    "picked_harness_rank_1_overall": sum(1 for r in rows if r["harness_rank"] == 1),
    "harness_rank_histogram": dict(sorted(collections.Counter(r["harness_rank"] for r in rows).items())),
    "chosen_probability_mean": round(sum(r["p"] for r in rows if r["p"] is not None) / max(1, sum(1 for r in rows if r["p"] is not None)), 3),
    "loop_guard_language_notes": sum(1 for r in rows for g in r["loop_guard"].values() if g["mechanism"] == "language"),
    "loop_guard_withdrawals": sum(1 for r in rows for g in r["loop_guard"].values() if g["mechanism"] == "withdrawn"),
}

lines = ["| # | offered (as listed) | withheld (rule) | pick | rank as listed | harness rank | p | settled by | result | x |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    wh = "; ".join(f"{k} ({v.split(':')[0]})" for k, v in r["withheld"].items()) or "-"
    p = f"{r['p']:.2f}" if r["p"] is not None else "-"
    x = f"{r['x']:.1f}" if r["x"] is not None else "-"
    lines.append(f"| {r['decision']} | {', '.join(r['offered'])} | {wh} | {r['choice']} | {r['rank']}/{r['n_offered']} | {r['harness_rank']} | {p} | "
                 f"{r['settled_by']} | {r['result'][:70]} | {x} |")
table = "\n".join(lines)
print(table)
print(json.dumps(summary, indent=1))
if a.md:
    a.md.write_text(table + "\n\n```json\n" + json.dumps(summary, indent=1) + "\n```\n")
if a.json:
    a.json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
