"""Kev serving on a GPU: Honing's KevDecisionModel against kev.serve, Kev's own server at the pinned commit.

Usage: kev_serving.py <our-gpu> <out-dir> <dtype> <port:name>...

The kev.serve instances must already be running on the given ports (see kev_serving.sh). The requests are the
24 serving parity requests (decision_requests.py) plus 3 with long logs that make Kev trim the state. For each
question it reports whether Honing and each reference chose the same option and the largest probability difference.
Honing packs and trims the state; the references receive the state text Honing sent the model.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

from honing.backends.jev.kev import KevDecisionModel, KevSettings

ROOT = Path(__file__).resolve().parents[1]
gpu, out, dtype = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
refs = [(int(x.split(":")[0]), x.split(":")[1]) for x in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
reqs = requests()
for i in range(3):
    base = reqs[i]
    logs = [
        f"tick {t}: unit {t % 17} moved to cell ({t % 90}, {(t * 7) % 90}); income {t * 3} credits; queue {t % 5}"
        for t in range(900 + 300 * i)
    ]
    memory = [f"memory note {j}" for j in range(20)]
    reqs.append(
        {"id": f"long-{i}", "parts": {**base["parts"], "logs": logs, "memory": memory}, "questions": base["questions"]}
    )


def post(port, state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone", body, {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(request, timeout=600))["answers"]


settings = KevSettings(
    pins_path=ROOT / "pins.json", source_dir=Path(os.environ["KEV_SOURCE_DIR"]), gpu=gpu, dtype=dtype
)
rows, ours_all = [], []
with KevDecisionModel.load(settings) as model:
    for r in reqs:
        ours = model.predict(r["parts"], r["questions"])
        ours_all.append({"id": r["id"], **{k: ours[k] for k in ("answers", "stateText", "tokenAudit", "trimmed")}})
        for port, name in refs:
            theirs = post(port, ours["stateText"], r["questions"])
            for q, a in ours["answers"].items():
                b = theirs[q]
                rows.append(
                    {
                        "request": r["id"],
                        "question": q,
                        "ref": name,
                        "same_choice": a["choice"] == b["choice"],
                        "max_abs": max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]),
                        "ours": a["probabilities"],
                        "theirs": b["probabilities"],
                    }
                )
(out / f"ours-{dtype}.json").write_text(json.dumps(ours_all, indent=1))
(out / f"rows-{dtype}.json").write_text(json.dumps(rows, indent=1))
for _, name in refs:
    sel = [x for x in rows if x["ref"] == name]
    print(
        f"vs kev.serve {name}: {len(sel)} questions, same choice {sum(x['same_choice'] for x in sel)}/{len(sel)}, "
        f"max |dp| {max(x['max_abs'] for x in sel):.4f}, mean |dp| {sum(x['max_abs'] for x in sel) / len(sel):.5f}"
    )
    for x in sel:
        if not x["same_choice"]:
            print("   differs:", x["request"], x["question"], "ours", x["ours"], "theirs", x["theirs"])
trimmed = [o for o in ours_all if o["trimmed"]]
print(
    f"[{dtype}] requests with a trimmed state: {len(trimmed)}; "
    f"entries trimmed: {sum(len(o['trimmed']) for o in trimmed)}"
)
