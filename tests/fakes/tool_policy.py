"""A stand-in OpenAI-compatible policy that makes the agent run one shell command, then answers.

The first reply to a conversation calls the ``bash`` tool with ``command``; once the conversation holds a tool result,
the reply is ``answer``. It streams when asked, as pi always does, and keeps every request's headers and body so a
test can check what reached the model. With it, a test drives a real agent through a tool call without a model.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.fakes.sse import send_stream


class _Handler(BaseHTTPRequestHandler):
    server: ToolCallingPolicy

    def log_message(self, *_: object) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with self.server.lock:
            self.server.requests.append({"headers": dict(self.headers), "body": body})
        if any(message.get("role") == "tool" for message in body.get("messages", [])):
            message: dict[str, Any] = {"role": "assistant", "content": self.server.answer}
            finish = "stop"
        else:
            arguments = json.dumps({"command": self.server.command})
            call = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": arguments}}
            message, finish = {"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls"
        reply = {
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if body.get("stream"):
            send_stream(self, reply)
            return
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ToolCallingPolicy(ThreadingHTTPServer):
    """Serves the stand-in on a free loopback port; use it as a context manager."""

    daemon_threads = True

    def __init__(self, command: str, answer: str = "done") -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.command = command
        self.answer = answer
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self) -> ToolCallingPolicy:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"
