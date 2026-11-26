"""Per-attempt fault checks "from decision 1" (no game; the weights re-ask calls the clean service 8813 once per
weights attempt, only with --reask). For a collection root:
  harness:    every model row: request vs clean_request (which fields differ), changed rows / model rows,
              first changed row; the pre-harness public observation keeps the true terrain;
  weights:    decision 0's request is byte-identical to the clean partner's decision 0 (same tick), |dp| between
              them; with --reask: the same request re-asked on the clean service, |dp| vs the logged alternate reply;
  controller: rows whose selection had a direction: selected vs sent, the bridge's echo (lastAction) of the sent
              input; first swapped row;
  all:        the integration's own receiver (checked_action) on every executed model row.
Usage: python3 -B fault_checks.py ROOT [--reask] [--out FILE]"""

import argparse
import json
import sys
from pathlib import Path

GAME = Path(__file__).resolve().parent
sys.path.insert(0, str(GAME.parents[1]))
sys.path.insert(0, str(GAME.parents[2] / "_shared" / "exp0"))
from contract import ServiceClient, guard_response, normalized, same_request
from decision import checked_action, strict_json_loads

MIRROR = {"left": "right", "right": "left", "flip_left": "flip_right", "flip_right": "flip_left"}


def rows_of(run):
    return [json.loads(l) for l in (run / "decisions.jsonl").read_text().splitlines() if l.strip()]  # noqa: E741


def state_of(request):
    return json.loads(request["parts"]["essential"][1])


def dp(a, b):
    a, b = normalized(a), normalized(b)
    return max(abs(a[k] - b[k]) for k in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--reask", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = sorted(p for p in a.root.iterdir() if (p / "manifest.json").is_file())
    by = {}
    for run in runs:
        m = json.loads((run / "manifest.json").read_text())
        by[(m["reset_group"], m["condition"], m["attempt"]["number"])] = run
    report = {}
    clean_client = ServiceClient("http://127.0.0.1:8813") if a.reask else None
    for (group, condition, number), run in sorted(by.items()):  # noqa: B007
        rows = rows_of(run)
        model = [r for r in rows if r["kind"] == "model" and r["executed"] is not None]
        receiver_rejected = []
        for r in model:
            raw = (run / r["response"]["file"]).read_bytes()
            try:
                checked_action(strict_json_loads(raw), r["request"])
            except Exception as error:
                receiver_rejected.append([r["index"], str(error)])
        result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else None
        item = {
            "run": run.name,
            "model_rows": len(model),
            "receiver_accepted": len(model) - len(receiver_rejected),
            "receiver_rejected": receiver_rejected[:3],
            "outcome": (
                None
                if result is None
                else {
                    k: result[k]
                    for k in ("status", "reason", "model_decisions", "auto_rows", "deaths", "trigger_index")
                }
            ),
        }
        if condition == "harness":
            changed = [r for r in model if r["fault"] and r["fault"]["changed"]]
            fields = set()
            for r in changed:
                s1, s0 = state_of(r["request"]), state_of(r["clean_request"])
                fields |= {k for k in set(s1) | set(s0) if s1.get(k) != s0.get(k)}
                if r["request"]["questions"] != r["clean_request"]["questions"]:
                    fields.add("questions")
                if r["request"]["parts"]["essential"][0] != r["clean_request"]["parts"]["essential"][0]:
                    fields.add("rule text")
            first = changed[0] if changed else None
            pub_true = all(
                r["public_observation"]["localSolidRows"] == state_of(r["clean_request"])["localSolidRows"]
                for r in model
                if r["clean_request"]
            )
            item.update(
                harness={
                    "rows_with_fault_record": sum(1 for r in model if r["fault"]),
                    "rows_changed": len(changed),
                    "first_changed_index": first["index"] if first else None,
                    "fields_changed": sorted(fields),
                    "first_changed_rows": (
                        {
                            "request": state_of(first["request"])["localSolidRows"],
                            "clean_request": state_of(first["clean_request"])["localSolidRows"],
                        }
                        if first
                        else None
                    ),
                    "public_observation_keeps_true_terrain": pub_true,
                }
            )
        if condition == "controller":
            swapped = [r for r in model if r["executed"]["option"] != r["model_choice"]]
            directional = [r for r in model if r["model_choice"] in MIRROR]
            echo_ok = all(r["extra"]["after_observation"]["lastAction"] == r["executed"]["option"] for r in model)
            mirror_ok = all(r["executed"]["option"] == MIRROR[r["model_choice"]] for r in swapped)
            item.update(
                controller={
                    "directional_selections": len(directional),
                    "swapped": len(swapped),
                    "every_directional_swapped": len(directional) == len(swapped),
                    "first_swapped_index": swapped[0]["index"] if swapped else None,
                    "first_swap": [swapped[0]["model_choice"], swapped[0]["executed"]["option"]] if swapped else None,
                    "bridge_echo_equals_sent": echo_ok,
                    "swaps_are_left_right_mirror": mirror_ok,
                }
            )
        if condition == "weights":
            partner = by.get((group, "no_injection", 1))
            w0 = model[0]
            entry = {"alternate_first_probabilities": w0["probabilities"]}
            if partner is not None:
                c0 = [r for r in rows_of(partner) if r["kind"] == "model"][0]  # noqa: RUF015
                entry.update(
                    first_request_identical_to_clean_partner=same_request(c0["request"], w0["request"])
                    and c0["native_before"]["tick"] == w0["native_before"]["tick"],
                    clean_partner_first_probabilities=c0["probabilities"],
                    max_dp_vs_clean_partner=dp(c0["probabilities"], w0["probabilities"]),
                )
            if clean_client is not None:
                raw, seconds = clean_client.decide_raw(w0["request"])
                again = guard_response(w0["request"], raw, "action")["probabilities"]
                entry.update(
                    reask_clean_probabilities=again,
                    max_dp_reask=dp(again, w0["probabilities"]),
                    reask_seconds=round(seconds, 3),
                    reask_equals_clean_partner=(again == entry.get("clean_partner_first_probabilities")),
                )
            entry["argmax_alternate"] = max(w0["probabilities"], key=w0["probabilities"].get)
            choices = [r["model_choice"] for r in model]
            entry["model_choices"] = {c: choices.count(c) for c in sorted(set(choices))}
            item.update(weights=entry)
        report[run.name] = item
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
