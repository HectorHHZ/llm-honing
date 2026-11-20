"""Stands in for an SGLang server's native routes, the ones slime's ``AnthropicAdapter`` calls.

``POST /generate`` takes ``input_ids`` and answers with scripted text, in the shape the
adapter reads (``meta_info.output_token_logprobs`` and ``meta_info.finish_reason``,
``slime/agent/adapters/common.py:496-501`` at the pin). ``GET /get_model_info`` answers
the worker's preflight. Token ids are bytes: the prompt is decoded with ``bytes(ids)``,
as ``tests/fakes/byte_tokenizer`` encodes it, so no tokenizer library is needed here.

``reply`` maps the decoded prompt to the model's text. Every request is recorded with
its decoded prompt and sampling parameters. ``delay_seconds`` holds each reply back,
and ``status`` makes ``/generate`` fail with that HTTP status.

``context_len`` gives the model a context length, checked as SGLang v0.5.15.post1's
tokenizer manager checks it (``tokenizer_manager.py:948-997``): a prompt of that many
tokens or more, or a prompt plus ``max_new_tokens`` beyond it, gets SGLang's 400 and
its message. ``GET /server_info`` then answers with ``max_req_input_len``, one less,
as SGLang's scheduler reports it (``tp_worker.py:342-345``); ``server_info=False``
makes it a 404, as for a server that does not report it.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

TOOLS_LINE = "<|tools|>"
TOOL_RESULT = "<|tool|>"


def write_call(path: str, content: str) -> str:
    """A ``Write`` tool call in the XML form slime parses without SGLang (``parsing.py:93-110``)."""
    return (
        "<tool_call>\n<function=Write>\n"
        f"<parameter=file_path>{path}</parameter>\n<parameter=content>{content}</parameter>\n"
        "</function>\n</tool_call>"
    )


def agent_script(call: str, final: str) -> Callable[[str], str]:
    """Call one tool, then answer ``final`` once the tool's result is in the prompt; ``ok`` without tools."""

    def reply(prompt: str) -> str:
        if TOOLS_LINE not in prompt:
            return "ok"
        return final if TOOL_RESULT in prompt else call

    return reply


@dataclass
class Request:
    prompt: str
    sampling_params: dict[str, Any]
    headers: dict[str, str]


@dataclass
class FakeSGLang:
    reply: Callable[[str], str]
    delay_seconds: float = 0.0
    status: int = 200
    context_len: int | None = None
    server_info: bool = True
    requests: list[Request] = field(default_factory=list)

    def overflow(self, input_len: int, max_new_tokens: int | None) -> str | None:
        """SGLang's message when the request does not fit ``context_len``, or ``None``."""
        limit = self.context_len
        if limit is None:
            return None
        if input_len >= limit:
            return f"The input ({input_len} tokens) is longer than the model's context length ({limit} tokens)."
        if max_new_tokens is not None and input_len + max_new_tokens > limit:
            return (
                f"Requested token count exceeds the model's maximum context length of {limit} tokens. You requested"
                f" a total of {input_len + max_new_tokens} tokens: {input_len} tokens from the input messages and"
                f" {max_new_tokens} tokens for the completion. Please reduce the number of tokens in the input"
                " messages or the completion to fit within the limit."
            )
        return None

    def __post_init__(self) -> None:
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, status: int, body: object) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path in ("/get_model_info", "/health"):
                    self._send(200, {"model_path": "fake-byte-model", "tokenizer_path": "fake-byte-model"})
                elif self.path == "/server_info" and fake.context_len is not None and fake.server_info:
                    self._send(200, {"context_length": None, "max_req_input_len": fake.context_len - 1})
                else:
                    self._send(404, {"error": self.path})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/generate":
                    self._send(404, {"error": self.path})
                    return
                prompt = bytes(body["input_ids"]).decode("utf-8", errors="replace")
                fake.requests.append(Request(prompt, body.get("sampling_params", {}), dict(self.headers)))
                time.sleep(fake.delay_seconds)
                if fake.status != 200:
                    self._send(fake.status, {"error": "fake upstream failure"})
                    return
                message = fake.overflow(len(body["input_ids"]), body.get("sampling_params", {}).get("max_new_tokens"))
                if message is not None:
                    self._send(400, {"error": {"message": message}})
                    return
                text = fake.reply(prompt)
                logprobs = [[-0.1, byte, None] for byte in text.encode("utf-8")]
                self._send(
                    200,
                    {
                        "text": text,
                        "meta_info": {"output_token_logprobs": logprobs, "finish_reason": {"type": "stop"}},
                    },
                )

            def log_message(self, *args: object) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> FakeSGLang:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
