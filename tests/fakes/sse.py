"""Chat completions as server-sent events, for the stand-in policy servers when a client asks to stream.

pi always sends ``"stream": true``; the stand-ins build a whole reply and send it here as the chunks an
OpenAI-compatible server streams: the message as one delta (with its tool calls, if any), then the finish reason,
then usage, then ``[DONE]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler
from typing import Any


def stream_chunks(reply: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``chat.completion.chunk`` objects that stream ``reply``, a ``chat.completion``."""
    base = {key: reply[key] for key in ("id", "created", "model") if key in reply}
    base["object"] = "chat.completion.chunk"
    chunks = []
    for choice in reply["choices"]:
        delta = {"role": "assistant", "content": choice["message"].get("content") or ""}
        calls = choice["message"].get("tool_calls")
        if calls:
            delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(calls)]
        chunks.append({**base, "choices": [{"index": choice["index"], "delta": delta, "finish_reason": None}]})
        finish = {"index": choice["index"], "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}
        chunks.append({**base, "choices": [finish]})
    if "usage" in reply:
        chunks.append({**base, "choices": [], "usage": reply["usage"]})
    return chunks


def send_stream(handler: BaseHTTPRequestHandler, reply: Mapping[str, Any]) -> None:
    """Answers ``handler``'s request with ``reply`` as server-sent events."""
    data = b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in stream_chunks(reply)) + b"data: [DONE]\n\n"
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
