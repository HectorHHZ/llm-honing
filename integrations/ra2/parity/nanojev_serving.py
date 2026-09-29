"""NanoJev serving on a GPU: #49's native worker, Honing's stdio worker, and Honing's HTTP server, same requests.

Usage: nanojev_serving.py <out-dir> <gpu-49> <gpu-stdio> <gpu-http>

Needs PR49_DIR (#49's examples/ra2 directory), NANOJEV_PYTHON (NanoJev's environment, with Honing installed), and
NANOJEV_SOURCE_DIR (the NanoJev checkout at the pinned commit), with the pinned model in the Hugging Face cache.
#49's worker reads a prepared manifest (a model directory and its file hashes), which this script writes for a plain
copy of the pinned files; its output directory therefore needs about 2.5 GB. The comparison ignores the checkpoint
directory each side reports, since the two read different copies of the same files.
The requests are the 24 serving parity requests (decision_requests.py); 5 carry 400 log entries, which NanoJev's
2,048-token limit makes it trim. #49's worker runs a second time on the stdio worker's GPU, to show how far #49
differs from itself there. On one GPU every field of every response must be equal, except each process's own
counter of inference calls (the concurrent run advances it in arrival order) and the checkpoint directory.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_requests import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PY = os.environ["NANOJEV_PYTHON"]
PR49 = Path(os.environ["PR49_DIR"])
SOURCE = Path(os.environ["NANOJEV_SOURCE_DIR"])
out, gpu49, gpu_stdio, gpu_http = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
pins = json.loads((ROOT / "pins.json").read_text())
pin = pins["nanojevModels"]["nanojev"]
env = {**os.environ, "HF_HUB_OFFLINE": "1"}
snapshot = subprocess.check_output(
    [
        PY,
        "-c",
        "import sys; from huggingface_hub import snapshot_download; "
        "from honing.backends.jev.nanojev import DOWNLOAD_PATTERNS; "
        "print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=DOWNLOAD_PATTERNS))",
        pin["repo"],
        pin["revision"],
    ],
    text=True,
    env=env,
).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


files = ["best.safetensors", "config.json", "backbone_config/config.json", "tokenizer/tokenizer.json"]
files += ["tokenizer/tokenizer_config.json"]
# #49's worker resolves symbolic links and refuses files outside the model directory, so it reads a plain copy of
# the files (the Hugging Face cache links them to shared blobs); Honing reads the cache.
plain = out / "nanojev-files"
for name in files:
    (plain / name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(snapshot) / name, plain / name)
sources = json.loads((PR49 / "league/model-sources.json").read_text())["nanojev"]
prepared = {
    "kind": "prepared-upstream-model",
    "source": sources["source"],
    "sourceRevision": sources["sourceRevision"],
    "model": sources["model"],
    "modelRevision": sources["modelRevision"],
    "path": str(plain.resolve()),
    "filesSha256": {name: digest(plain / name) for name in files},
}
(out / "prepared-nanojev.json").write_text(json.dumps(prepared, indent=2))
reqs = requests()
(out / "requests.json").write_text(json.dumps(reqs))


def stdio(command, cwd, extra_env):
    with (out / f"{Path(command[-1]).name}-{extra_env.get('LAYA_GPU', 'x')}.err").open("a") as errors:
        return run_stdio(command, cwd, extra_env, errors)


def run_stdio(command, cwd, extra_env, errors):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={**env, **extra_env},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=errors,
        text=True,
    )
    ready = json.loads(process.stdout.readline())
    if not ready.get("ready"):
        raise RuntimeError(f"worker failed: {ready}")
    responses = []
    for request in reqs:
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        responses.append(json.loads(process.stdout.readline()))
    process.stdin.close()
    process.wait(timeout=120)
    return ready["metadata"], responses


reference_meta, reference = stdio(
    [
        PY,
        "-m",
        "integration.models.worker",
        "--family",
        "nanojev",
        "--prepared-manifest",
        str(out / "prepared-nanojev.json"),
        "--source-root",
        str(SOURCE),
        "--gpu",
        gpu49,
        "--max-length",
        "2048",
    ],
    PR49,
    {},
)
# A second #49 run on the GPU the stdio worker uses: how far #49 differs from itself there.
_, reference_b = stdio(
    [
        PY,
        "-m",
        "integration.models.worker",
        "--family",
        "nanojev",
        "--prepared-manifest",
        str(out / "prepared-nanojev.json"),
        "--source-root",
        str(SOURCE),
        "--gpu",
        gpu_stdio,
        "--max-length",
        "2048",
    ],
    PR49,
    {},
)
ours_meta, ours = stdio(
    [PY, str(ROOT / "integration/laya/worker.py")],
    ROOT,
    {"DECISION_FAMILY": "nanojev", "LAYA_GPU": gpu_stdio, "NANOJEV_SOURCE_DIR": str(SOURCE)},
)

server_errors = (out / "server.err").open("w")
server = subprocess.Popen(
    [PY, "-m", "honing.backends.jev.server", "--family", "nanojev", "--pins", str(ROOT / "pins.json"),
     "--nanojev-source", str(SOURCE), "--gpu", gpu_http, "--port", "0"],
    cwd=REPO,
    env=env,
    stdout=subprocess.PIPE,
    stderr=server_errors,
    text=True,
)  # fmt: skip
url = json.loads(server.stdout.readline())["url"]


def post(request):
    body = json.dumps(request).encode()
    call = urllib.request.Request(url + "/v1/decide", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(call, timeout=600))


http = [post(r) for r in reqs]
started = time.time()
with ThreadPoolExecutor(4) as pool:
    concurrent = list(pool.map(post, reqs))
concurrent_seconds = time.time() - started
server.terminate()
server.wait(timeout=60)
server_errors.close()

COMPARE = ("answers", "stateText", "tokenAudit", "trimmed", "modelResult")


def comparable(response):
    value = {key: response[key] for key in COMPARE}
    value["modelResult"] = json.loads(json.dumps(value["modelResult"]))
    value["modelResult"]["execution"].pop("inference_call_index", None)
    value["modelResult"]["checkpoint"].pop("directory", None)
    return value


def compare(label, theirs):
    mismatched = [
        (r["id"], key)
        for r, a, b in zip(reqs, reference, theirs, strict=True)
        for key in COMPARE
        if comparable(a)[key] != comparable(b)[key]
    ]
    verdict = "identical on " + ", ".join(COMPARE) if not mismatched else f"differs: {mismatched[:4]}"
    print(f"#49 worker vs {label}: {len(theirs)} requests, {verdict}; largest |dp| {largest_difference(theirs):.2e}")


print(
    "trimming requests:",
    sum(bool(r["trimmed"]) for r in reference),
    "entries trimmed:",
    sum(len(r["trimmed"]) for r in reference),
)
print(
    "same model files:",
    reference_meta["weightSetSha256"] is not None
    and ours_meta["weightsSha256"] == prepared["filesSha256"]["best.safetensors"],
)
for name, value in (
    ("reference", reference),
    ("reference_b", reference_b),
    ("stdio", ours),
    ("http", http),
    ("concurrent", concurrent),
):
    (out / f"{name}.json").write_text(json.dumps(value))


def largest_difference(theirs):
    return max(
        abs(a["answers"][q]["probabilities"][k] - b["answers"][q]["probabilities"][k])
        for a, b in zip(reference, theirs, strict=True)
        for q in a["answers"]
        for k in a["answers"][q]["probabilities"]
    )


compare("#49 worker on the stdio worker's GPU", reference_b)
compare("our stdio worker", ours)
compare("our HTTP server (sequential)", http)
compare("our HTTP server (4 concurrent clients)", concurrent)
print(f"4 concurrent clients: {len(concurrent)} requests in {concurrent_seconds:.1f}s")
