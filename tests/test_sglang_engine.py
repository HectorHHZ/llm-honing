"""The SGLang engine process (RFC #103 prototype): the next-token request and its answer, the server's lifecycle and
recorded versions, and concurrent serving. SGLang needs a GPU and its own environment, so a stand-in server answers
with SGLang's request and response shapes; the GPU check is integrations/ra2/parity/startlux_sglang.py.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev.server import DecisionServer
from honing.backends.jev.server import main as server_main
from honing.backends.sglang import SGLangEngine, SGLangError, SGLangSettings


def logprobs_for(prompt: list[int], wanted: list[int]) -> list[list[Any]]:
    """The stand-in's answer: token t after a prompt has log-probability -(t + len(prompt)) / 10."""
    return [[-(t + len(prompt)) / 10, t, None] for t in wanted]


# SGLang's /generate answer for a batch of input_ids: one result per prompt, the requested token ids'
# log-probabilities under meta_info.output_token_ids_logprobs, one list per generated token.
STAND_IN = """
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def logprobs_for(prompt, wanted):
    return [[-(t + len(prompt)) / 10, t, None] for t in wanted]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({})
        elif self.path == "/get_server_info":
            self._send({"version": "0.0.test", "disable_radix_cache": "--disable-radix-cache" in sys.argv,
                        "context_length": 8200, "argv": sys.argv})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/flush_cache":
            self._send({})
            return
        wanted = payload["token_ids_logprob"]
        self._send([{"meta_info": {"output_token_ids_logprobs": [logprobs_for(p, wanted)]}}
                    for p in payload["input_ids"]])

port = int(sys.argv[sys.argv.index("--port") + 1])
ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def stand_in_python(tmp_path: Path) -> SGLangSettings:
    """Settings whose interpreter finds a stand-in ``sglang`` package (and torch and transformers for the version
    probe) ahead of anything installed."""
    site = tmp_path / "site"
    (site / "sglang").mkdir(parents=True)
    (site / "sglang" / "__init__.py").write_text("__version__ = '0.0.test'\n")
    (site / "sglang" / "launch_server.py").write_text(textwrap.dedent(STAND_IN))
    (site / "torch").mkdir()
    (site / "torch" / "__init__.py").write_text(
        "import types\n__version__ = '9.9.test'\nversion = types.SimpleNamespace(cuda='99.9')\n"
    )
    (site / "transformers").mkdir()
    (site / "transformers" / "__init__.py").write_text("__version__ = '8.8.test'\n")
    return SGLangSettings(
        python=Path(sys.executable), gpu="GPU-test", env={"PYTHONPATH": str(site)}, start_timeout=60,
        log_path=tmp_path / "engine.log",
    )  # fmt: skip


def test_the_engine_starts_records_its_versions_and_answers_logprobs(stand_in_python: SGLangSettings) -> None:
    with SGLangEngine.launch(Path("/models/startlux"), stand_in_python) as engine:
        assert engine.versions["sglang"] == "0.0.test"
        assert engine.versions["torch"] == "9.9.test" and engine.versions["cuda"] == "99.9"
        assert engine.versions["server"]["disable_radix_cache"] is False
        prompts = [[1, 2, 3], [4, 5]]
        got = engine.next_token_logprobs(prompts, [30, 10, 20])
        # One answer per prompt, in the requested token order.
        assert got == [[v for v, _, _ in logprobs_for(p, [30, 10, 20])] for p in prompts]
        assert engine.next_token_logprobs([], [1]) == []
        engine.flush_cache()
        process = engine._process
    assert process is not None and process.poll() is not None  # stopped on exit


def test_turning_the_prefix_cache_off_reaches_the_server(stand_in_python: SGLangSettings) -> None:
    settings = SGLangSettings(**{**stand_in_python.__dict__, "prefix_cache": False})
    with SGLangEngine.launch(Path("/models/startlux"), settings) as engine:
        assert engine.versions["server"]["disable_radix_cache"] is True


def test_a_server_that_exits_is_reported_with_its_log(tmp_path: Path) -> None:
    settings = SGLangSettings(python=Path(sys.executable), env={"PYTHONPATH": str(tmp_path)}, log_path=tmp_path / "x")
    with pytest.raises(SGLangError, match="exited"), SGLangEngine.launch(Path("/models/startlux"), settings):
        pass


class _Generate(BaseHTTPRequestHandler):
    """A /generate that answers with whatever the test put in ``server.answer``, and keeps the request."""

    server: Any

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        self.server.payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(self.server.answer).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def generate() -> Iterator[Any]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Generate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def test_the_request_prefills_and_asks_for_one_token_s_distribution(generate: Any) -> None:
    generate.answer = [{"meta_info": {"output_token_ids_logprobs": [[[-0.5, 7, None], [-1.5, 8, None]]]}}]
    engine = SGLangEngine(f"http://127.0.0.1:{generate.server_address[1]}", None, {})
    assert engine.next_token_logprobs([[1, 2]], [8, 7]) == [[-1.5, -0.5]]
    assert generate.payload == {
        "input_ids": [[1, 2]],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": [8, 7],
    }


@pytest.mark.parametrize(
    "answer",
    [
        [],  # no result for the prompt
        [{"meta_info": {"output_token_ids_logprobs": [[[-0.5, 7, None]]]}}],  # a requested token is missing
        [{"meta_info": {"output_token_ids_logprobs": [[[-0.5, 7, None], [-0.1, 9, None]]]}}],  # another token
    ],
)
def test_an_answer_of_another_shape_is_refused(generate: Any, answer: Any) -> None:
    generate.answer = answer
    engine = SGLangEngine(f"http://127.0.0.1:{generate.server_address[1]}", None, {})
    with pytest.raises(SGLangError):
        engine.next_token_logprobs([[1, 2]], [7, 8])


class _SlowModel:
    """A model whose predict takes a while and records how many calls overlap."""

    def __init__(self, concurrent: bool) -> None:
        self.metadata = {"family": "test"}
        self.concurrent = concurrent
        self.active = self.peak = 0
        self.lock = threading.Lock()

    def predict(self, parts: Any, questions: Any) -> dict[str, object]:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.3)
        with self.lock:
            self.active -= 1
        return {"answers": {}}


@pytest.mark.parametrize(("concurrent", "peak"), [(False, 1), (True, 2)])
def test_a_model_that_batches_itself_is_called_concurrently(concurrent: bool, peak: int) -> None:
    model = _SlowModel(concurrent)
    server = DecisionServer(("127.0.0.1", 0), model)  # type: ignore[arg-type]
    serving = threading.Thread(target=server.serve, daemon=True)
    serving.start()
    try:
        calls = [threading.Thread(target=server.decide, args=({}, {"q": {}})) for _ in range(2)]
        for call in calls:
            call.start()
        for call in calls:
            call.join()
    finally:
        server.stop()
        serving.join()
        server.server_close()
    assert model.peak == peak


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--family", "kev", "--pins", "p.json", "--engine", "sglang", "--sglang-python", "py"],
            "startlux and lettered only",
        ),
        (["--family", "startlux", "--pins", "p.json", "--engine", "sglang"], "--sglang-python"),
    ],
)
def test_the_server_names_what_the_sglang_engine_needs(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        server_main(argv)
    assert message in capsys.readouterr().err
