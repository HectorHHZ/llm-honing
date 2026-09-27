"""#52 parity on GPU: #49's original worker vs our stdio worker vs our HTTP server, same requests.

No game needed. Requests vary state length (some force trimming), question count, and option count.
"""

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PY = os.environ.get("LAYA_PYTHON", sys.executable)
PR49 = Path(os.environ["PR49_DIR"])
OURS = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ENV = {
    **os.environ,
    "LAYA_MODEL_TYPE": "typed-decisions",
    "LAYA_SOURCE_DIR": str(PR49 / "deps/laya"),
    "LAYA_MODEL_CONFIG_DIR": str(PR49 / ".cache/models"),
}
BUILDINGS = ["Power plant", "Ore refinery", "Barracks", "War factory", "Radar", "Battle lab", "Wall", "Service depot"]


def requests(n=24, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        essential = [f"Tick {30 * (i + 1)}. Credits {rng.randint(0, 10000)}. Army {rng.randint(0, 20)}."]
        logs = [
            f"t{j}: order {rng.choice(BUILDINGS)} queued and {rng.choice(['placed', 'cancelled', 'delayed'])}"
            for j in range(rng.choice([0, 5, 400]))
        ]
        parts = {
            "essential": essential,
            "details": [f"Visible enemy units: {rng.randint(0, 9)}"],
            "memory": [],
            "logs": logs,
        }
        questions = {}
        for q in range(rng.randint(1, 3)):
            opts = rng.sample(BUILDINGS, rng.randint(2, 8))
            questions[f"q{q}"] = {
                "type": "choice",
                "instructions": f"Which structure should be built next ({q})?",
                "criteria": {f"o{k}": label for k, label in enumerate(opts)},
            }
        out.append({"id": i + 1, "parts": parts, "questions": questions})
    return out


def stdio(worker, gpu, reqs, cwd):
    proc = subprocess.Popen(
        [PY, "-u", str(worker)],
        cwd=cwd,
        env={**ENV, "LAYA_GPU": str(gpu)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=(OUT / f"stdio-gpu{gpu}.err").open("w"),
        text=True,
    )
    ready = json.loads(proc.stdout.readline())
    assert ready.get("ready"), ready
    replies = []
    for r in reqs:
        proc.stdin.write(json.dumps(r) + "\n")
        proc.stdin.flush()
        replies.append(json.loads(proc.stdout.readline()))
    proc.stdin.close()
    proc.wait(timeout=60)
    return ready["metadata"], replies


def http_server(gpu):
    proc = subprocess.Popen(
        [
            PY,
            "-m",
            "honing.backends.jev.server",
            "--pins",
            str(OURS / "integrations/ra2/pins.json"),
            "--model",
            "typed-decisions",
            "--model-config-dir",
            ENV["LAYA_MODEL_CONFIG_DIR"],
            "--laya-source",
            ENV["LAYA_SOURCE_DIR"],
            "--gpu",
            str(gpu),
        ],
        cwd=OURS,
        env={**ENV, "PYTHONPATH": str(OURS)},
        stdout=subprocess.PIPE,
        stderr=(OUT / f"server-gpu{gpu}.err").open("w"),
        text=True,
    )
    ready = json.loads(proc.stdout.readline())
    assert ready.get("ready"), ready
    return proc, ready["url"], ready["metadata"]


def post(url, body):
    req = urllib.request.Request(
        url + "/v1/decide", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


COMPARE = ("answers", "modelResult", "stateText", "tokenAudit", "trimmed")


def diff(a, b, label):
    bad = [(x["id"], k) for x, y in zip(a, b, strict=True) for k in COMPARE if x.get(k) != y.get(k)]
    errors = [x.get("error") for x in a + b if "error" in x]
    verdict = "identical on " + ", ".join(COMPARE) if not bad else f"{len(bad)} differences, first {bad[:5]}"
    print(f"{label}: {len(a)} requests, {verdict}" + (f"; errors {errors[:2]}" if errors else ""))
    return not bad


reqs = requests()
(OUT / "requests.json").write_text(json.dumps(reqs))
print("trimming requests:", sum(len(r["parts"]["logs"]) == 400 for r in reqs))
ref_meta, ref = stdio(PR49 / "integration/laya/worker.py", 3, reqs, PR49)
our_meta, ours = stdio(OURS / "integrations/ra2/integration/laya/worker.py", 4, reqs, OURS / "integrations/ra2")
proc, url, srv_meta = http_server(5)
try:
    seq = [post(url, r) for r in reqs]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        conc = list(pool.map(lambda r: post(url, r), reqs))
    conc_s = time.time() - t0
finally:
    proc.terminate()
    proc.wait(timeout=60)
for name, data in (("reference", ref), ("ours-stdio", ours), ("ours-http", seq), ("ours-http-4x", conc)):
    (OUT / f"{name}.json").write_text(json.dumps(data))
keys = ("revision", "dtype", "torch", "transformers", "layaVersion", "sourceCommit", "maxLen", "headMaxLen")
print("metadata equal:", all(ref_meta.get(k) == m.get(k) for m in (our_meta, srv_meta) for k in keys))
ok = diff(ref, ours, "#49 worker vs our stdio worker")
ok &= diff(ref, seq, "#49 worker vs our HTTP server (sequential)")
ok &= diff(ref, conc, "#49 worker vs our HTTP server (4 concurrent clients)")
trimmed = sum(len(r["trimmed"]) for r in ref)
print(f"4 concurrent clients: {len(conc)} requests in {conc_s:.1f}s; trimmed entries in reference: {trimmed}")
print("PARITY" if ok else "NO PARITY")
