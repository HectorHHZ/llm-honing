"""A loopback server that answers OpenAI chat completions only, as SGLang does, and records each request.

It replies ``pong from <model>`` naming the model it was asked for, as a JSON body or, when the request
streams, as server-sent events. Any other route is a 404, so a client that needs another route fails.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class ChatUpstream:
    """Runs the server on a free port for the duration of a ``with`` block."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                upstream.requests.append(
                    {"path": self.path, "authorization": self.headers.get("Authorization"), **body}
                )
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                text = f"pong from {body.get('model')}"
                if body.get("stream"):
                    self._stream(body.get("model"), text)
                else:
                    self._json(body.get("model"), text)

            def _json(self, model: str, text: str) -> None:
                payload = json.dumps(
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _stream(self, model: str, text: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                # The role, the text, then an empty delta that carries the finish reason.
                deltas = [{"role": "assistant", "content": ""}, {"content": text}, {}]
                for index, delta in enumerate(deltas):
                    chunk = {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if index == 2 else None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    def __enter__(self) -> ChatUpstream:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"
