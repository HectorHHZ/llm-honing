"""StartLux-Decision on the SGLang engine (RFC #103 prototype) against Honing's native StartLux path.

Usage, from the StartLux environment (STARTLUX_PYTHON, STARTLUX_SOURCE_DIR as for startlux_serving.py) and with
SGLANG_PYTHON naming the SGLang environment's interpreter; STARTLUX_MODEL defaults to startlux-4b:

    "$STARTLUX_PYTHON" startlux_sglang.py run <native|sglang|sglang-nocache> <gpu> <out-dir>
    "$STARTLUX_PYTHON" startlux_sglang.py compare <out-dir>
    "$STARTLUX_PYTHON" startlux_sglang.py bench <native|sglang> <gpu> <out-dir> [rounds]

``run`` answers the 24 serving parity requests (decision_requests.py) and 3 with long logs that Honing trims to its
8,192-token default, whose 1 to 3 questions share a long state. The SGLang variants answer every request twice: the
second pass finds the prompts in the radix cache. ``compare`` reports, per question, whether two runs chose the same
option and the largest probability difference: SGLang against native, the radix cache off against on, and the second
pass against the first. ``bench`` starts ``honing.backends.jev.server`` and sends the same requests from 1 and then 2
concurrent clients, ``rounds`` times each (default 3), and reports latency and throughput.
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("STARTLUX_MODEL", "startlux-4b")


def all_requests():
    reqs = requests()
    for i in range(3):
        base = reqs[i]
        logs = [
            f"tick {t}: unit {t % 17} moved to cell ({t % 90}, {(t * 7) % 90}); income {t * 3} credits; queue {t % 5}"
            for t in range(900 + 300 * i)
        ]
        memory = [f"memory note {j}" for j in range(20)]
        parts = {**base["parts"], "logs": logs, "memory": memory}
        reqs.append({"id": f"long-{i}", "parts": parts, "questions": base["questions"]})
    return reqs


def settings_for(variant, gpu):
    from honing.backends.jev.startlux import StartLuxSettings

    engine = "native" if variant == "native" else "sglang"
    return StartLuxSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ["STARTLUX_SOURCE_DIR"]),
        model=MODEL,
        gpu=gpu,
        engine=engine,
        sglang_python=Path(os.environ["SGLANG_PYTHON"]) if engine == "sglang" else None,
        prefix_cache=variant != "sglang-nocache",
    )


def run(variant, gpu, out):
    from honing.backends.jev.startlux import StartLuxDecisionModel

    os.environ.setdefault("SGLANG_LOG", str(out / f"engine-{MODEL}-{variant}.log"))
    reqs = all_requests()
    passes = 1 if variant == "native" else 2
    results = []
    with StartLuxDecisionModel.load(settings_for(variant, gpu)) as model:
        # The model call alone, without Honing's packing, whose trimming re-renders the state for every entry it
        # drops and dominates the long requests.
        timings = []
        decide = model._engine.decide

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return decide(*args, **kwargs)
            finally:
                timings.append((time.perf_counter() - started) * 1000)

        model._engine.decide = timed
        model.predict(reqs[0]["parts"], reqs[0]["questions"])  # warm-up, not recorded
        if variant != "native":  # start from an empty cache, so the first pass is cold
            model._engine.sglang.flush_cache()
        for pass_ in range(passes):
            for r in reqs:
                answer = model.predict(r["parts"], r["questions"])
                results.append(
                    {
                        "pass": pass_,
                        "id": r["id"],
                        "answers": answer["answers"],
                        "inputTokens": {k: a["inputTokens"] for k, a in answer["tokenAudit"].items()},
                        "latencyMs": answer["latencyMs"],
                        "modelMs": timings[-1],
                    }
                )
        metadata = dict(model.metadata)
    (out / f"run-{MODEL}-{variant}.json").write_text(json.dumps({"metadata": metadata, "results": results}, indent=1))
    lat = [x["modelMs"] for x in results if x["pass"] == 0]
    print(f"{variant}: {len(reqs)} requests, model call median {statistics.median(lat):.1f} ms, max {max(lat):.1f} ms")


def _rows(a, b, pass_a=0, pass_b=0):
    """Per question: same choice, largest probability difference."""
    by_id = {(x["id"], x["pass"]): x for x in b["results"]}
    rows = []
    for x in a["results"]:
        if x["pass"] != pass_a:
            continue
        y = by_id[(x["id"], pass_b)]
        for q, p in x["answers"].items():
            r = y["answers"][q]
            rows.append(
                {
                    "request": x["id"],
                    "question": q,
                    "long": max(x["inputTokens"].values()) > 4096,
                    "same": p["choice"] == r["choice"],
                    "max_abs": max(abs(p["probabilities"][k] - r["probabilities"][k]) for k in p["probabilities"]),
                }
            )
    return rows


def compare(out):
    runs = {v: json.loads((out / f"run-{MODEL}-{v}.json").read_text()) for v in ("native", "sglang", "sglang-nocache")}
    pairs = [
        ("sglang (cold) vs native", _rows(runs["sglang"], runs["native"])),
        ("sglang, radix cache off vs on", _rows(runs["sglang-nocache"], runs["sglang"])),
        ("sglang, cached second pass vs first", _rows(runs["sglang"], runs["sglang"], 1, 0)),
        ("sglang without cache, second pass vs first", _rows(runs["sglang-nocache"], runs["sglang-nocache"], 1, 0)),
    ]
    for name, rows in pairs:
        for label, selected in (("all", rows), ("long prompts", [r for r in rows if r["long"]])):
            if selected:
                worst = max(selected, key=lambda r: r["max_abs"])
                print(
                    f"{MODEL} {name}, {label}: {len(selected)} questions, same choice "
                    f"{sum(r['same'] for r in selected)}/{len(selected)}, max |dp| {worst['max_abs']:.6f} "
                    f"(request {worst['request']} {worst['question']}), "
                    f"mean |dp| {statistics.fmean(r['max_abs'] for r in selected):.6f}"
                )
        for r in rows:
            if not r["same"]:
                print("   differs:", r)
    for v, data in runs.items():
        for pass_ in sorted({x["pass"] for x in data["results"]}):
            lat = [x["modelMs"] for x in data["results"] if x["pass"] == pass_ and not str(x["id"]).startswith("long")]
            long = [x["modelMs"] for x in data["results"] if x["pass"] == pass_ and str(x["id"]).startswith("long")]
            print(
                f"{v} pass {pass_}: model call median {statistics.median(lat):.1f} ms, long requests "
                + ", ".join(f"{t:.0f}" for t in long)
                + " ms"
            )


def bench(variant, gpu, out, rounds):
    reqs = requests()  # the 24 untrimmed requests: the long ones measure Honing's packing, not the engine
    command = [
        sys.executable, "-m", "honing.backends.jev.server", "--family", "startlux", "--pins", str(ROOT / "pins.json"),
        "--startlux-source", os.environ["STARTLUX_SOURCE_DIR"], "--model", MODEL, "--gpu", gpu,
        *(("--engine", "sglang", "--sglang-python", os.environ["SGLANG_PYTHON"]) if variant == "sglang" else ()),
    ]  # fmt: skip
    env = {**os.environ, "SGLANG_LOG": str(out / f"bench-engine-{MODEL}-{variant}.log")}
    errors = (out / f"bench-{MODEL}-{variant}.err").open("w")
    server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, env=env, text=True)
    try:
        ready = json.loads(server.stdout.readline())
        url = ready["url"]

        def post(r):
            body = json.dumps({"parts": r["parts"], "questions": r["questions"]}).encode()
            request = urllib.request.Request(f"{url}/v1/decide", body, {"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=600) as response:
                json.load(response)
            return (time.perf_counter() - started) * 1000

        post(reqs[0])  # warm-up
        report = {}

        def client(offset, latencies, lock):
            for _ in range(rounds):
                for i in range(len(reqs)):
                    t = post(reqs[(i + offset) % len(reqs)])
                    with lock:
                        latencies.append(t)

        for clients in (1, 2):
            latencies, lock = [], threading.Lock()
            threads = [threading.Thread(target=client, args=(c * 7, latencies, lock)) for c in range(clients)]
            started = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - started
            latencies.sort()
            report[clients] = {
                "requests": len(latencies),
                "median_ms": statistics.median(latencies),
                "p90_ms": latencies[int(0.9 * (len(latencies) - 1))],
                "requests_per_s": len(latencies) / wall,
            }
            rounded = {k: round(v, 1) for k, v in report[clients].items()}
            print(f"{variant} {MODEL}, {clients} client(s): {json.dumps(rounded)}")
        (out / f"bench-{MODEL}-{variant}.json").write_text(json.dumps(report, indent=1))
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        out = Path(sys.argv[4])
        out.mkdir(parents=True, exist_ok=True)
        run(sys.argv[2], sys.argv[3], out)
    elif mode == "compare":
        compare(Path(sys.argv[2]))
    elif mode == "bench":
        out = Path(sys.argv[4])
        out.mkdir(parents=True, exist_ok=True)
        bench(sys.argv[2], sys.argv[3], out, int(sys.argv[5]) if len(sys.argv) > 5 else 3)
    else:
        raise SystemExit(__doc__)
