"""DiffusionGemma serving through djev on a GPU: every Honing path against the djev API itself, same requests.

Usage: djev_serving.py <out-dir> <gpu> <small-context-gpu>

Run it with the djev environment's Python (scripts/setup-djev.sh), with DJEV_SOURCE_DIR set and the pinned model in
the Hugging Face cache. It starts a djev deployment through Honing (honing.backends.djev.runtime), which verifies the
environment first, and sends the 24 serving requests (decision_requests.py) plus 3 whose logs exceed djev's
20,000-character state limit:

- the reference is djev's own ``POST /v1/request`` on the state text Honing packed;
- DjevDecisionModel, twice (it must repeat itself);
- Honing's stdio worker (DECISION_FAMILY=djev);
- Honing's HTTP server, sequentially and from 4 concurrent clients (djev runs vLLM batch invariant).

A second deployment with a 2,048-token context checks the other trimming path: djev refuses the long state as
over the context, and DjevDecisionModel removes entries until djev accepts it.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

from honing.backends.djev import DjevClient, DjevDecisionModel, DjevRuntimeSettings, DjevSettings

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PINS = ROOT / "pins.json"
SOURCE = Path(os.environ["DJEV_SOURCE_DIR"])
out, gpu, small_gpu = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
reqs = requests()
for i in range(3):
    base = reqs[i]
    logs = [
        f"tick {t}: unit {t % 17} moved to cell ({t % 90}, {(t * 7) % 90}); queue {t % 5}"
        for t in range(400 + 200 * i)
    ]
    reqs.append({"id": 100 + i, "parts": {**base["parts"], "logs": logs}, "questions": base["questions"]})


def runtime(chosen_gpu, ports, max_model_len=32768):
    return DjevRuntimeSettings(
        pins_path=PINS,
        source_dir=SOURCE,
        python=Path(sys.executable),
        gpu=chosen_gpu,
        api_port=ports[0],
        model_port=ports[1],
        max_model_len=max_model_len,
        log_dir=out / f"logs-{chosen_gpu}",
        cache_dir=Path(os.environ.get("DJEV_CACHE_DIR", out / "cache")),
    )


def answers_of(response):
    return {
        key: {"choice": a["choice"], "probabilities": a["probabilities"]} for key, a in response["answers"].items()
    }


def compare(label, reference, theirs):
    same = sum(answers_of(a) == answers_of(b) for a, b in zip(reference, theirs, strict=True))
    worst = max(
        abs(a["answers"][q]["probabilities"][k] - b["answers"][q]["probabilities"][k])
        for a, b in zip(reference, theirs, strict=True)
        for q in a["answers"]
        for k in a["answers"][q]["probabilities"]
    )
    print(f"djev API vs {label}: {same}/{len(reference)} requests identical; largest |dp| {worst:.2e}", flush=True)


report = {}
with DjevDecisionModel.load(DjevSettings(pins_path=PINS, runtime=runtime(gpu, (8100, 8101)))) as model:
    url = str(model.metadata["url"])
    print("deployment verified and ready:", json.dumps(model.metadata["runtime"]), flush=True)
    first = [model.predict(r["parts"], r["questions"]) for r in reqs]
    second = [model.predict(r["parts"], r["questions"]) for r in reqs]
    client = DjevClient(url)
    direct = [client.request(o["stateText"], r["questions"]) for o, r in zip(first, reqs, strict=True)]
    trimmed = [len(o["trimmed"]) for o in first]
    print(
        f"state trimmed to djev's character limit on {sum(bool(t) for t in trimmed)} requests ({sum(trimmed)} entries)"
    )
    compare("DjevDecisionModel", direct, first)
    compare("DjevDecisionModel, second pass", direct, second)

    worker = subprocess.Popen(
        [sys.executable, str(ROOT / "integration/laya/worker.py")],
        cwd=ROOT,
        env={**os.environ, "DECISION_FAMILY": "djev", "DJEV_URL": url},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert json.loads(worker.stdout.readline())["ready"]
    stdio = []
    for r in reqs:
        worker.stdin.write(json.dumps(r) + "\n")
        worker.stdin.flush()
        stdio.append(json.loads(worker.stdout.readline()))
    worker.stdin.close()
    worker.wait(timeout=60)
    compare("Honing's stdio worker", direct, stdio)

    command = [sys.executable, "-m", "honing.backends.jev.server", "--family", "djev", "--pins", str(PINS)]
    server = subprocess.Popen(
        [*command, "--djev-url", url, "--port", "0"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    server_url = json.loads(server.stdout.readline())["url"]

    def post(request):
        body = json.dumps(request).encode()
        call = urllib.request.Request(server_url + "/v1/decide", body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(call, timeout=600))

    http = [post(r) for r in reqs]
    started = time.time()
    with ThreadPoolExecutor(4) as pool:
        concurrent = list(pool.map(post, reqs))
    concurrent_seconds = time.time() - started
    server.terminate()
    compare("Honing's HTTP server (sequential)", direct, http)
    compare("Honing's HTTP server (4 concurrent clients)", direct, concurrent)
    same_state = all(a["stateText"] == b["stateText"] for a, b in zip(first, http, strict=True))
    print(f"same packed state on every path: {same_state}; 4 concurrent clients: {concurrent_seconds:.1f}s")
    report["latencyMs"] = sorted(o["latencyMs"] for o in second)

with DjevDecisionModel.load(
    DjevSettings(pins_path=PINS, runtime=runtime(small_gpu, (8110, 8111), max_model_len=2048))
) as small:
    long = reqs[1]
    answer = small.predict(long["parts"], long["questions"])
    accepted = DjevClient(str(small.metadata["url"])).request(answer["stateText"], long["questions"])
    print(
        f"2,048-token context: {len(answer['trimmed'])} entries trimmed after djev refused the context; "
        f"{answer['tokenAudit']['q0']['inputTokens']} prompt tokens; same answers as djev on that state: "
        f"{answers_of(answer) == answers_of(accepted)}"
    )

latencies = report["latencyMs"]
print(f"warm latency ms: median {latencies[len(latencies) // 2]:.0f}, max {latencies[-1]:.0f}")
(out / "responses.json").write_text(json.dumps({"direct": direct, "honing": first}, indent=1))
