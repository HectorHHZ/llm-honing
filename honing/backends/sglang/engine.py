"""One SGLang server process, started from its own environment and driven over HTTP.

SGLang pins its own torch, CUDA build and kernels, which differ from every
family's environment, so Honing never imports it: the server runs from the
Python of a separate environment (``SGLangSettings.python``) as its own
process, on one GPU, and this module talks to it with the standard library
only. A crash or a CUDA fault in the engine therefore stays in that process.

The one call a decision needs is :meth:`SGLangEngine.next_token_logprobs`: for
each prompt, the log-probabilities of chosen token ids at the position after
the prompt's last token. It asks for one new token and reads the
log-probabilities SGLang reports for that token's distribution, so the
prompt is prefilled and nothing else is generated. Log-probabilities differ
from the logits by one constant per prompt, so a softmax over a subset of
them equals the softmax over the same logits.

Two settings make these the log-probabilities of the checkpoint's own logits.
The request is greedy: SGLang divides the logits by the sampling temperature
before its log-softmax unless every request in the batch is greedy (at 0.21,
``logprob_processor.py``), so a temperature other than 0 or 1 would change
them. And the output head runs in fp32 (``--enable-fp32-lm-head``): in bf16
the logits are rounded to steps of 1/16 to 1/8 at the magnitudes the decision
letters reach, while the decision families read their letters in fp32.

The engine keeps its radix cache unless ``prefix_cache`` is off: prompts that
begin with the same tokens, the questions of one request or consecutive
decisions of one game, reuse the computed beginning.

With ``chat`` the server also loads the checkpoint's tokenizer and answers
SGLang's OpenAI-compatible routes under :attr:`SGLangEngine.openai_url`
(``/v1/chat/completions``, ``/v1/completions``) as ``served_model``: the
endpoint a harness runner or a :class:`~honing.core.ChatModel` calls. Token-id
requests such as :meth:`~SGLangEngine.next_token_logprobs` work either way.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SGLangSettings:
    """How to start one SGLang server for one checkpoint on one GPU.

    ``python`` is the interpreter of the environment SGLang is installed in;
    ``gpu`` is a CUDA device, by index or UUID, made the only visible one.
    ``mem_fraction`` is SGLang's ``--mem-fraction-static``. ``chat`` loads
    the tokenizer and serves the OpenAI-compatible routes as ``served_model``.
    ``extra_args`` are passed to ``sglang.launch_server`` unchanged.
    """

    python: Path
    gpu: str = "0"
    port: int = 0
    prefix_cache: bool = True
    fp32_lm_head: bool = True
    chat: bool = False
    served_model: str | None = None
    mem_fraction: float = 0.8
    context_length: int | None = None
    extra_args: tuple[str, ...] = ()
    start_timeout: float = 900.0
    log_path: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


def die_with_parent() -> None:
    """In the child before exec: on Linux, receive SIGTERM when the process that started it dies."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _stop(process: subprocess.Popen[bytes]) -> None:
    """Stop the server and every process it started (scheduler, detokenizer): they share its process group."""
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


class SGLangError(RuntimeError):
    """The engine refused a request, answered in an unexpected shape, or exited."""


class SGLangEngine:
    """A running SGLang server. Create it with :meth:`launch`, which stops the process on exit."""

    def __init__(self, url: str, process: subprocess.Popen[bytes] | None, versions: dict[str, Any]) -> None:
        self.url = url
        self._process = process
        self.versions = versions

    @property
    def openai_url(self) -> str:
        """The base URL of the OpenAI-compatible routes (with ``chat``)."""
        return f"{self.url}/v1"

    @classmethod
    @contextlib.contextmanager
    def launch(cls, model_path: Path, settings: SGLangSettings) -> Iterator[SGLangEngine]:
        port = settings.port or _free_port()
        command = [
            str(settings.python), "-m", "sglang.launch_server",
            "--model-path", str(model_path), "--host", "127.0.0.1", "--port", str(port),
            *(() if settings.chat else ("--skip-tokenizer-init",)),
            *(("--served-model-name", settings.served_model) if settings.served_model else ()),
            "--mem-fraction-static", str(settings.mem_fraction),
            *(("--context-length", str(settings.context_length)) if settings.context_length else ()),
            *(() if settings.prefix_cache else ("--disable-radix-cache",)),
            *(("--enable-fp32-lm-head",) if settings.fp32_lm_head else ()),
            *settings.extra_args,
        ]  # fmt: skip
        env = {**os.environ, **settings.env, "CUDA_VISIBLE_DEVICES": settings.gpu}
        # The engine compiles kernels at start-up with tools its environment installs (ninja), so that
        # environment's bin directory goes first, as if it were activated.
        env["PATH"] = os.pathsep.join([str(settings.python.parent), env.get("PATH", "")])
        log = settings.log_path.open("ab") if settings.log_path else subprocess.DEVNULL
        # Its own process group, so stopping it stops SGLang's workers too; and it dies with this process, so a
        # server killed without running its cleanup does not leave an engine holding the GPU.
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=die_with_parent
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + settings.start_timeout
            while True:
                if process.poll() is not None:
                    raise SGLangError(f"the SGLang server exited with {process.returncode}; see {settings.log_path}")
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise SGLangError(f"the SGLang server did not start within {settings.start_timeout:.0f} s")
                time.sleep(2)
            versions = cls._versions(settings.python, url, env)
            yield cls(url, process, versions)
        finally:
            _stop(process)
            if settings.log_path:
                log.close()  # type: ignore[union-attr]

    @staticmethod
    def _versions(python: Path, url: str, env: dict[str, str]) -> dict[str, Any]:
        """The SGLang, torch, CUDA and transformers versions of the engine's environment, and its server settings."""
        probe = (
            "import json, sglang, torch, transformers; print(json.dumps({'sglang': sglang.__version__, "
            "'torch': torch.__version__, 'cuda': torch.version.cuda, 'transformers': transformers.__version__}))"
        )
        versions: dict[str, Any] = json.loads(subprocess.check_output([str(python), "-c", probe], text=True, env=env))
        with urllib.request.urlopen(f"{url}/get_server_info", timeout=30) as response:
            info = json.load(response)
        keep = (
            "version", "disable_radix_cache", "enable_fp32_lm_head", "attention_backend", "dtype",
            "mem_fraction_static", "context_length", "served_model_name", "skip_tokenizer_init",
        )  # fmt: skip
        versions["server"] = {key: info.get(key) for key in keep if key in info}
        return versions

    def _post(self, path: str, payload: dict[str, Any], timeout: float = 600.0) -> Any:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(f"{self.url}{path}", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise SGLangError(f"SGLang {path} answered {error.code}: {error.read()[:500]!r}") from None

    def flush_cache(self) -> None:
        """Empty the radix cache, so the next prompts are computed from their first token."""
        request = urllib.request.Request(f"{self.url}/flush_cache", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=60):
            pass

    def next_token_logprobs(self, prompts: Sequence[Sequence[int]], token_ids: Sequence[int]) -> list[list[float]]:
        """For each prompt, the log-probability of each of ``token_ids`` as the next token, in that order.

        All prompts go to the engine in one request, so they run in one
        scheduled batch and share whatever prefix the cache holds.
        """
        if not prompts:
            return []
        wanted = [int(t) for t in token_ids]
        payload = {
            "input_ids": [list(map(int, p)) for p in prompts],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "logprob_start_len": -1,
            "token_ids_logprob": wanted,
        }
        answers = self._post("/generate", payload)
        if not isinstance(answers, list) or len(answers) != len(prompts):
            raise SGLangError("SGLang answered a batch with another number of results")
        out = []
        for answer in answers:
            entries = answer["meta_info"]["output_token_ids_logprobs"][0]  # the distribution after the prompt
            values = {int(token): float(value) for value, token, *_ in entries}
            if set(values) != set(wanted):
                raise SGLangError("SGLang returned log-probabilities for other tokens than requested")
            out.append([values[t] for t in wanted])
        return out


if __name__ == "__main__":  # a manual check: python -m honing.backends.sglang.engine <sglang-python> <model> <gpu>
    with SGLangEngine.launch(Path(sys.argv[2]), SGLangSettings(python=Path(sys.argv[1]), gpu=sys.argv[3])) as engine:
        print(json.dumps(engine.versions))
        print(engine.next_token_logprobs([[1, 2, 3, 4]], [5, 6, 7]))
