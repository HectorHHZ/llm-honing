"""StartLux-Decision serving on GPUs: Honing's StartLuxDecisionModel against startlux_decision.server, StartLux's own
server at the pinned commit.

Usage: STARTLUX_PYTHON=... STARTLUX_SOURCE_DIR=... [STARTLUX_MODEL=startlux-27b] \\
    "$STARTLUX_PYTHON" startlux_serving.py <our-gpu> <reference-gpu> <out-dir>     (default startlux-4b)

The reference server is started on its own GPU from the pinned snapshot, on port 8094 plus its GPU index, with its
defaults (CUDA graphs, its own graph self-test) and without the vision tower, as Honing loads it. The requests are the
24 serving parity requests (decision_requests.py) plus 3 with long logs that make Honing trim the state to its
8,192-token default. Honing packs and trims the state; the reference receives the state text Honing sent the model.
For each question it reports whether both chose the same option and the largest probability difference. The
reference keeps StartLux's shared-prefix path, which Honing turns off (see honing/backends/jev/startlux.py), so
requests with several questions over 4,096 tokens are expected to differ at bf16 precision; they are reported
separately.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

from honing.backends.jev.startlux import StartLuxDecisionModel, StartLuxSettings, resolve_checkpoint

ROOT = Path(__file__).resolve().parents[1]
gpu, reference_gpu, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
PORT = 8094 + int(reference_gpu)
out.mkdir(parents=True, exist_ok=True)
source = Path(os.environ["STARTLUX_SOURCE_DIR"])
settings = StartLuxSettings(
    pins_path=ROOT / "pins.json", source_dir=source, model=os.environ.get("STARTLUX_MODEL", "startlux-4b"), gpu=gpu
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
        {"id": f"long-{i}", "parts": {**base["parts"], "logs": logs, "memory": memory}, "questions": base["questions"]}
    )


def post(state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/systemone", body, {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(request, timeout=600))["answers"]


os.environ["HF_HUB_OFFLINE"] = "1"
snapshot = resolve_checkpoint(settings, json.loads(settings.pins_path.read_text()))["path"]
log = (out / f"reference-{settings.model}.log").open("w")
reference = subprocess.Popen(
    [sys.executable, "-m", "startlux_decision.server", "--model", snapshot, "--port", str(PORT), "--no-images"],
    cwd=source,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": reference_gpu},
    stdout=log,
    stderr=subprocess.STDOUT,
)
try:
    rows, ours_all = [], []
    with StartLuxDecisionModel.load(settings) as model:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                break
            except OSError:
                if reference.poll() is not None:
                    raise RuntimeError("the reference server exited; see its log") from None
                time.sleep(2)
        for r in reqs:
            ours = model.predict(r["parts"], r["questions"])
            ours_all.append(
                {"id": r["id"], **{k: ours[k] for k in ("answers", "stateText", "tokenAudit", "trimmed", "latencyMs")}}
            )
            theirs = post(ours["stateText"], r["questions"])
            for q, a in ours["answers"].items():
                b = theirs[q]
                rows.append(
                    {
                        "request": r["id"],
                        "question": q,
                        "same_choice": a["choice"] == b["choice"],
                        "max_abs": max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]),
                        "ours": a["probabilities"],
                        "theirs": b["probabilities"],
                    }
                )
        metadata = dict(model.metadata)
finally:
    reference.terminate()
    reference.wait()
(out / f"ours-{settings.model}.json").write_text(json.dumps({"metadata": metadata, "responses": ours_all}, indent=1))
(out / f"rows-{settings.model}.json").write_text(json.dumps(rows, indent=1))
shared = {
    o["id"]
    for o in ours_all
    if len(o["tokenAudit"]) > 1 and min(a["inputTokens"] for a in o["tokenAudit"].values()) > 4096
}
for name, selected in (
    ("eager and graph paths", [x for x in rows if x["request"] not in shared]),
    ("shared-prefix requests", [x for x in rows if x["request"] in shared]),
):
    if selected:
        print(
            f"{settings.model} vs startlux_decision.server, {name}: {len(selected)} questions, "
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
    f"median latency of untrimmed requests {latencies[len(latencies) // 2]:.1f} ms; "
    f"graph self-test {metadata['graphSelfTestMaxDifference']:.2e} over {metadata['cudaGraphs']} graphs"
)
