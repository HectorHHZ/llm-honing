"""Compare #49 and the extracted controller in a real game with one shared CPU model.

This checks the game/controller boundary. It does not replace the separate GPU
worker parity test. The old stdio client uses a transport bridge to the same
model server as the extracted HTTP client; neither game policy is replaced.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from integrations.ra2.parity.replay_compare import replay_comparison  # noqa: E402

from honing.backends.jev.laya import LayaDecisionModel, LayaSettings, _resolve_model  # noqa: E402
from honing.backends.jev.server import DecisionServer  # noqa: E402

BRIDGE = """#!/usr/bin/env python3
import json, os, sys, urllib.request
url = os.environ["PARITY_DECISION_SERVER"]
with urllib.request.urlopen(url + "/v1/health") as response:
    print(response.read().decode(), flush=True)
for line in sys.stdin:
    request = urllib.request.Request(url + "/v1/decide", data=line.encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        print(response.read().decode(), flush=True)
"""


def normalize(value):
    """Timing is observational; preserve every decision, probability and context field."""
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key != "latencyMs"}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


@contextlib.contextmanager
def serving_cpu_model(model):
    """Run the CPU parity model through the same queue as the decision-server CLI."""
    server = DecisionServer(("127.0.0.1", 0), model)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop()
        thread.join(timeout=10)
        server.server_close()
        if thread.is_alive():
            raise RuntimeError("parity decision server did not stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="#49 examples/ra2 directory")
    parser.add_argument("--source", type=Path, required=True, help="pinned Laya checkout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map", default="mp03t4.map")
    parser.add_argument("--seat", default="red", choices=("red", "blue"))
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    integration = REPO / "integrations/ra2"
    settings = LayaSettings(
        pins_path=integration / "pins.json",
        model_type="typed-decisions",
        model_config_dir=args.reference / ".cache/models",
        source_dir=args.source,
    )
    pins = json.loads(settings.pins_path.read_text())
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pins["layaSource"]["revision"]:
        raise ValueError("Laya source differs from pin")
    record = _resolve_model(settings, pins, revision)
    os.environ.update(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", USE_TF="0")
    sys.path.insert(0, str(args.source))
    import laya
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(1)
    with contextlib.redirect_stdout(sys.stderr):
        agent = laya.load(record["path"], device="cpu")
    metadata = {**record, "sourceCommit": revision, "device": "cpu", "dtype": str(agent.dtype)}
    bridge = args.out / "stdio_http_bridge.py"
    bridge.write_text(BRIDGE)
    bridge.chmod(0o700)
    started = time.monotonic()
    with serving_cpu_model(LayaDecisionModel(agent, metadata, 2048)) as server:
        env = {**os.environ, "LAYA_PYTHON": str(bridge), "PARITY_DECISION_SERVER": server.url}
        for label, root in (("reference", args.reference), ("extracted", integration)):
            command = [
                "node",
                str(root / "integration/laya/run.mjs"),
                f"--out={args.out / label}",
                f"--ticks={args.ticks}",
                f"--seed={args.seed}",
                f"--map={args.map}",
                f"--seat={args.seat}",
            ]
            if label == "extracted":
                command.append(f"--decision-server={server.url}")
            with (args.out / f"{label}.log").open("w") as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({"completed": label, "seconds": time.monotonic() - started}), flush=True)
    report = {
        "device": "cpu",
        "ticks": args.ticks,
        "seed": args.seed,
        "map": args.map,
        "seat": args.seat,
        "model": metadata,
        "elapsedSeconds": time.monotonic() - started,
    }
    compare_runs(args.out, report)


def compare_runs(out, report):
    """Check already recorded matches without playing them again."""
    for filename in ("decisions.jsonl", "actions.jsonl"):
        left, right = [
            normalize([json.loads(line) for line in (out / name / filename).read_text().splitlines()])
            for name in ("reference", "extracted")
        ]
        report[filename] = {"equal": left == right, "referenceRows": len(left), "extractedRows": len(right)}
    results = [json.loads((out / name / "result.json").read_text()) for name in ("reference", "extracted")]
    report["simulationHashes"] = [result["hash"] for result in results]
    report["replay"] = replay_comparison(out / "reference/match.rplx", out / "extracted/match.rplx")
    report["passed"] = (
        all(report[name]["equal"] for name in ("decisions.jsonl", "actions.jsonl"))
        and results[0]["hash"] == results[1]["hash"]
        and report["replay"]["equal"]
    )
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
