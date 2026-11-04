"""Nimble-v4-max serving on GPUs: Honing's NimbleDecisionModel against Bespoke's NimbleEngine called directly.

Usage: NIMBLE_PYTHON=... NIMBLE_SOURCE_DIR=... NIMBLE_KIT_DIR=... \\
    "$NIMBLE_PYTHON" nimble_serving.py <our-gpu> <reference-gpu> <out-dir>

Bespoke ships no server for the model, so the reference is the engine its Decision Index run used, loaded in a
second process on its own GPU with that run's settings (shared prefix on, max_tails=48) and its default 32,768-token
limit, and called as the Decision Index kit calls it: engine(state, questions). The requests are the 24 serving
parity requests (decision_requests.py), 3 with long logs that make Honing trim the state to its 8,192-token default,
and 3 that ask yes/no and score questions. Honing packs and trims the state; the reference receives the state text
Honing sent the model. For each question it reports whether both chose the same option and the largest probability
difference.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

from honing.backends.jev.nimble import NimbleDecisionModel, NimbleSettings, resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ["no threat", "low threat", "moderate threat", "high threat", "critical threat"]


def reference_main(snapshot):
    """The reference process: one JSON request per stdin line, the engine's raw response per stdout line."""
    sys.path[:0] = [os.environ["NIMBLE_KIT_DIR"], str(Path(os.environ["NIMBLE_SOURCE_DIR"]) / "code")]
    from contextlib import redirect_stdout

    from nimble.evaluation.decision_index_engine import NimbleEngine

    with redirect_stdout(sys.stderr):
        engine = NimbleEngine(adapter=snapshot, max_tails=48)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        with redirect_stdout(sys.stderr):
            response, _ = engine(request["state"], request["questions"])
        print(json.dumps(response), flush=True)


def ask(reference, state, questions):
    reference.stdin.write(json.dumps({"state": state, "questions": questions}) + "\n")
    reference.stdin.flush()
    return json.loads(reference.stdout.readline())["answers"]


def comparable(question, answer):
    """A distribution to compare: the options' probabilities, or yes and no for a yes/no question."""
    if question["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def main(gpu, reference_gpu, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = NimbleSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ["NIMBLE_SOURCE_DIR"]),
        kit_dir=Path(os.environ["NIMBLE_KIT_DIR"]),
        gpu=gpu,
    )
    reqs = requests()
    for i in range(3):
        base = reqs[i]
        logs = [
            f"tick {t}: unit {t % 17} moved to cell ({t % 90}, {(t * 7) % 90}); income {t * 3} credits; queue {t % 5}"
            for t in range(900 + 300 * i)
        ]
        memory = [f"memory note {j}" for j in range(20)]
        reqs.append(
            {
                "id": f"long-{i}",
                "parts": {**base["parts"], "logs": logs, "memory": memory},
                "questions": base["questions"],
            }
        )
    for i in range(3):
        base = reqs[i]
        reqs.append(
            {
                "id": f"typed-{i}",
                "parts": base["parts"],
                "questions": {
                    "attack": {"type": "noul", "instructions": "Should the army attack now?", "criteria": None},
                    "threat": {"type": "score", "instructions": "How threatened is the base?", "criteria": LEVELS},
                    **base["questions"],
                },
            }
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    snapshot = resolve_checkpoint(settings, json.loads(settings.pins_path.read_text()))["path"]
    log = (out / "reference.log").open("w")
    reference = subprocess.Popen(
        [sys.executable, __file__, "--reference", snapshot],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": reference_gpu},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
    )

    try:
        if json.loads(reference.stdout.readline()) != {"ready": True}:
            raise RuntimeError("the reference engine did not start; see its log")
        rows, ours_all = [], []
        with NimbleDecisionModel.load(settings) as model:
            for r in reqs:
                ours = model.predict(r["parts"], r["questions"])
                ours_all.append(
                    {
                        "id": r["id"],
                        **{k: ours[k] for k in ("answers", "stateText", "tokenAudit", "trimmed", "latencyMs")},
                    }
                )
                theirs = ask(reference, ours["stateText"], r["questions"])
                for q, a in ours["answers"].items():
                    p, t = comparable(r["questions"][q], a), comparable(r["questions"][q], theirs[q])
                    rows.append(
                        {
                            "request": r["id"],
                            "question": q,
                            "type": r["questions"][q]["type"],
                            "same_choice": max(p, key=p.get) == max(t, key=t.get),
                            "max_abs": max(abs(p[k] - t[k]) for k in p),
                            "ours": p,
                            "theirs": t,
                        }
                    )
            metadata = dict(model.metadata)
    finally:
        reference.stdin.close()
        reference.wait()
    (out / "ours.json").write_text(json.dumps({"metadata": metadata, "responses": ours_all}, indent=1))
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    for kind in ("choice", "noul", "score"):
        selected = [x for x in rows if x["type"] == kind]
        if selected:
            print(
                f"nimble-v4-max vs NimbleEngine, {kind}: {len(selected)} questions, "
                f"same choice {sum(x['same_choice'] for x in selected)}/{len(selected)}, "
                f"max |dp| {max(x['max_abs'] for x in selected):.6f}"
            )
    for x in rows:
        if not x["same_choice"]:
            print("   differs:", x["request"], x["question"], "ours", x["ours"], "theirs", x["theirs"])
    trimmed = [o for o in ours_all if o["trimmed"]]
    latencies = sorted(o["latencyMs"] for o in ours_all if not o["trimmed"])
    print(
        f"requests with a trimmed state: {len(trimmed)}; entries trimmed: {sum(len(o['trimmed']) for o in trimmed)}; "
        f"peak prompt {max(a['inputTokens'] for o in ours_all for a in o['tokenAudit'].values())} tokens; "
        f"median latency of untrimmed requests {latencies[len(latencies) // 2]:.1f} ms"
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reference":
        reference_main(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
