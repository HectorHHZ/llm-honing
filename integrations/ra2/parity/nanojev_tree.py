"""NanoJev's token-tree inference against its reference inference on one GPU, with the serving parity requests.

Usage: nanojev_tree.py <out-dir> <gpu> [requests]

Needs NANOJEV_SOURCE_DIR (the NanoJev checkout at the pinned commit) and the pinned model in the Hugging Face cache, as
nanojev_serving.py does. Each mode loads in its own process, one after another on the same GPU, because the GPU lock
refuses a GPU that already has a compute process: the reference in fp32 (the yardstick), the reference in bf16 (what
serves today), then tree and tree-compiled in bf16. Each answers the parity requests (decision_requests.py, 24 unless
given) twice and times the second pass. For each bf16 mode the script reports the largest probability difference
from the fp32 reference, the questions whose choice differs from it, the load time, the per-request latency and the
peak GPU memory PyTorch reserved. Run it with a Python that has NanoJev's requirements and can import Honing (for
example NANOJEV_PYTHON with the repository on PYTHONPATH).
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

ROOT = Path(__file__).resolve().parents[1]
MODES = (("reference", "fp32"), ("reference", "bf16"), ("tree", "bf16"), ("tree-compiled", "bf16"))


def worker(inference, precision, gpu, count, out):
    from honing.backends.jev.nanojev import NanoJevDecisionModel, NanoJevSettings

    settings = NanoJevSettings(
        pins_path=ROOT / "pins.json",
        source_dir=Path(os.environ["NANOJEV_SOURCE_DIR"]),
        gpu=gpu,
        precision=precision,
        inference=inference,
    )
    batch = requests(count)
    started = time.time()
    with NanoJevDecisionModel.load(settings) as model:
        load_seconds = time.time() - started
        for r in batch:
            model.predict(r["parts"], r["questions"])
        answers, latency_ms = [], []
        for r in batch:
            begun = time.perf_counter()
            answers.append(model.predict(r["parts"], r["questions"])["answers"])
            latency_ms.append((time.perf_counter() - begun) * 1000)
        metadata = dict(model.metadata)
        import torch

        reserved_gib = torch.cuda.max_memory_reserved() / 2**30
    record = {
        "load_seconds": load_seconds,
        "latency_ms": latency_ms,
        "reserved_gib": reserved_gib,
        "answers": answers,
        "metadata": metadata,
    }
    Path(out).write_text(json.dumps(record, default=str))


def compare(yardstick, answers):
    differences, differing = [], []
    for i, (a, b) in enumerate(zip(yardstick, answers, strict=True)):
        for q in a:
            pa, pb = a[q]["probabilities"], b[q]["probabilities"]
            differences.append(max(abs(pa[k] - pb[k]) for k in pa))
            if a[q]["choice"] != b[q]["choice"]:
                top = sorted(pa.values(), reverse=True)
                differing.append({"request": i + 1, "question": q, "fp32_top2_gap": round(top[0] - top[1], 6)})
    return max(differences), differing, len(differences)


def main():
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
        return
    out, gpu = Path(sys.argv[1]), sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for inference, precision in MODES:
        path = out / f"{inference}-{precision}.json"
        command = [sys.executable, __file__, "--worker", inference, precision, gpu, str(count), str(path)]
        subprocess.run(command, check=True)
        runs[inference, precision] = json.loads(path.read_text())
    yardstick = runs["reference", "fp32"]["answers"]
    summary = {}
    for (inference, precision), run in runs.items():
        largest, differing, questions = compare(yardstick, run["answers"])
        summary[f"{inference} {precision}"] = {
            "largest_abs_probability_difference_from_fp32": largest,
            "choices_differing_from_fp32": differing,
            "questions": questions,
            "median_latency_ms": statistics.median(run["latency_ms"]),
            "load_seconds": run["load_seconds"],
            "peak_reserved_gib": run["reserved_gib"],
        }
        print(
            f"{inference} {precision}: {questions} questions, largest |dp| from fp32 {largest:.2e}, "
            f"{len(differing)} choices differ, median {statistics.median(run['latency_ms']):.1f} ms per request, "
            f"loaded in {run['load_seconds']:.0f} s, peak {run['reserved_gib']:.1f} GiB reserved"
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
