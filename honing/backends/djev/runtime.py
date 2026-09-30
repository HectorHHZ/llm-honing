"""Start and stop a pinned djev deployment: DiffusionGemma on a patched vLLM, behind the djev API.

djev (``Davipar/djev-dev``) answers typed questions with DiffusionGemma in one
denoising read. A deployment is two processes in djev's own environment: vLLM,
patched by djev's installer, holding the model on one GPU; and the djev API,
which compiles questions, checks the context, and turns label probabilities
into answers. Honing starts both, waits until the API is ready, and stops them.
It never imports vLLM, torch, or djev: it talks to the environment through a
subprocess and to the deployment over HTTP, so Honing's own environment is
untouched.

Before starting anything, :func:`verify_environment` checks what djev's
installer recorded against djev's pins:

- the djev checkout is the commit in ``pins.json`` (``djevSource``), unmodified;
- the model djev serves (``djev.config``) is the one in ``pins.json`` (``djevModels``);
- torch, vLLM, transformers, and Pillow have exactly the versions djev's
  ``runtime/sources.json`` names;
- vLLM carries djev's installed-source record (``djev_sources.json``) for the
  same base and structured-read commits.

vLLM starts with the arguments djev's ``runtime/serve.py`` builds, changing only
the port (so several deployments can share a node, one GPU each) and running
the environment's own ``vllm`` rather than the one on ``PATH``.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.laya import gpu_lock

# Run inside djev's environment: what its installer recorded, as JSON on stdout.
_INSPECT = """
import importlib.metadata, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, ".")
from djev import config
pins = json.loads(Path("runtime/sources.json").read_text())
vllm = Path(importlib.util.find_spec("vllm").origin).parent
record = vllm / "djev_sources.json"
print(json.dumps({
    "model": config.MODEL,
    "modelRevision": config.MODEL_REVISION,
    "pinnedPackages": pins["packages"],
    "installedPackages": {name: importlib.metadata.version(name) for name in pins["packages"]},
    "baseCommit": pins["base_commit"],
    "structuredReadCommit": pins["structured_read_commit"],
    "installed": json.loads(record.read_text()) if record.exists() else None,
}))
"""

# Run inside djev's environment: djev's own vLLM command, on another port.
_LAUNCH = """
import os, sys
sys.path.insert(0, ".")
from runtime import serve
command = serve.command()
command[command.index("--port") + 1] = sys.argv[1]
# djev's image has vllm on PATH; here it is the environment's own entry point.
command[0] = os.path.join(os.path.dirname(sys.executable), command[0])
env = {**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_BATCH_INVARIANT": "1",
       "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
os.execvpe(command[0], command, env)
"""


@dataclass(frozen=True)
class DjevRuntimeSettings:
    """Where djev's environment and checkout are, which GPU and ports to use, and the context length.

    ``python`` is the djev environment's interpreter (``scripts/setup-djev.sh``
    builds it). ``max_model_len`` is passed to both processes, as djev requires.
    ``cache_dir`` holds the vLLM, TorchInductor, Triton, and CUDA compilation
    caches, as djev's image keeps them under ``/cache``; it defaults to
    ``.honing-cache`` in the djev checkout and should be on local disk.
    """

    pins_path: Path
    source_dir: Path
    python: Path
    gpu: str = "0"
    api_port: int = 8000
    model_port: int = 8001
    max_model_len: int = 32768
    lock_dir: Path | None = None
    startup_seconds: float = 1800.0
    log_dir: Path | None = None
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 1024 <= self.max_model_len <= 32768:
            raise ValueError("djev accepts a max_model_len from 1,024 to 32,768")
        if self.api_port == self.model_port:
            raise ValueError("the djev API and vLLM need different ports")


def verify_environment(settings: DjevRuntimeSettings, pins: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse a djev checkout, model, or environment that differs from the pins; return what was checked."""
    source = settings.source_dir
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != pins["djevSource"]["revision"] or dirty:
        raise ValueError("djev requires the pinned, unmodified checkout in pins.json")
    found = json.loads(
        subprocess.check_output([str(settings.python), "-c", _INSPECT], cwd=source, text=True).strip().splitlines()[-1]
    )
    models = pins["djevModels"]
    if not any((found["model"], found["modelRevision"]) == (m["repo"], m["revision"]) for m in models.values()):
        raise ValueError(f"djev serves {found['model']}@{found['modelRevision']}, which pins.json does not name")
    if found["installedPackages"] != found["pinnedPackages"]:
        raise ValueError(f"djev's environment differs from its pins: {found['installedPackages']}")
    installed = found["installed"]
    if (
        installed is None
        or installed.get("base_commit") != found["baseCommit"]
        or installed.get("structured_read_commit") != found["structuredReadCommit"]
        or installed.get("packages") != found["pinnedPackages"]
    ):
        raise ValueError("vLLM lacks djev's installed sources for the pinned commits; run djev's installer")
    return {"sourceCommit": revision, **{k: v for k, v in found.items() if k != "installed"}}


def _stop_with_parent() -> None:
    """In the child, before exec: receive SIGTERM when the process that started it dies, however it dies.

    A deployment holds a GPU; if Honing is killed outright, vLLM and the API must not outlive it.
    """
    pr_set_pdeathsig = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(pr_set_pdeathsig, signal.SIGTERM)


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Wait for a terminated process; kill it if it does not exit in two minutes."""
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()


def _get(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def _wait(url: str, processes: list[subprocess.Popen[bytes]], deadline: float) -> Any:
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"a djev process exited with {process.returncode} before {url} was ready")
        status, body = _get(url)
        if status == 200:
            return body
        time.sleep(5)
    raise TimeoutError(f"{url} was not ready in time")


@contextlib.contextmanager
def djev_runtime(settings: DjevRuntimeSettings) -> Iterator[dict[str, Any]]:
    """Verify the environment, start vLLM and the djev API on one locked GPU, and yield where the API is.

    Yields ``{"url": ..., "config": <djev /config>, "environment": <what was verified>}``.
    Both processes are stopped on exit, and receive SIGTERM if this process dies first.
    """
    pins = json.loads(settings.pins_path.read_text())
    environment = verify_environment(settings, pins)
    logs = settings.log_dir or settings.source_dir / ".honing-logs"
    logs.mkdir(parents=True, exist_ok=True)
    cache = settings.cache_dir or settings.source_dir / ".honing-cache"
    shared = {
        **os.environ,
        # The cache layout of djev's image (runtime/Dockerfile), on a local directory.
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "DJEV_MAX_MODEL_LEN": str(settings.max_model_len),
        "TOKENIZERS_PARALLELISM": "false",
        # Everything is pinned and cached by setup; a running deployment fetches nothing.
        "HF_HUB_OFFLINE": "1",
    }
    processes: list[subprocess.Popen[bytes]] = []
    with gpu_lock(settings.gpu, settings.lock_dir) as uuid, contextlib.ExitStack() as stack:
        model_log = stack.enter_context((logs / "vllm.log").open("ab"))
        api_log = stack.enter_context((logs / "api.log").open("ab"))
        try:
            processes.append(
                subprocess.Popen(
                    [str(settings.python), "-c", _LAUNCH, str(settings.model_port)],
                    cwd=settings.source_dir,
                    env={**shared, "CUDA_VISIBLE_DEVICES": uuid},
                    stdout=model_log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_stop_with_parent,
                )
            )
            deadline = time.monotonic() + settings.startup_seconds
            _wait(f"http://127.0.0.1:{settings.model_port}/health", processes, deadline)
            processes.append(
                subprocess.Popen(
                    [str(settings.python), "-m", "djev", "--host", "127.0.0.1", "--port", str(settings.api_port)],
                    cwd=settings.source_dir,
                    env={**shared, "DJEV_UPSTREAM": f"http://127.0.0.1:{settings.model_port}", "DJEV_OFFLINE": "1"},
                    stdout=api_log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=_stop_with_parent,
                )
            )
            url = f"http://127.0.0.1:{settings.api_port}"
            _wait(url + "/ready", processes, deadline)
            _, config = _get(url + "/config")
            yield {"url": url, "config": config, "environment": environment, "gpuUuid": uuid, "logs": str(logs)}
        finally:
            for process in reversed(processes):
                process.terminate()
            for process in reversed(processes):
                _reap(process)
