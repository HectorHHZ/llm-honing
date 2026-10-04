"""A loopback OpenAI-compatible server that stands in for both models of the live GEPA run (#100).

The URL's first path segment picks a behaviour, so one config entry switches a
model between them: ``/ok/v1`` answers, ``/429/v1`` and ``/401/v1`` always fail
with that status (and echo the key and the request in the error body, as some
providers do), ``/slow/v1`` answers after ``SLOW_SECONDS``, and
``/window-A-B/v1`` fails policy calls ``A`` to ``B - 1`` (counted from 0) with
429, so an outage can start and end inside a run.

Both models can share one name, as the live config's single model pin makes
them, so a request is told apart by its shape: the minimal agent always sends
a system message, and GEPA's reflection sends one user message. The policy
answers as the minimal agent expects. It knows each task's reference fix and returns it when
the system prompt names every hidden convention the task's cases rely on, and
the task's file unchanged otherwise; a task with no convention is always fixed.
So the harness decides the score, as it would with a real model. One
convention matters only to a training task, so a child that learns it can
beat its parent's minibatch and still tie at the gate, which rejects it.

The reflection model appends the next line of ``REFLECTION_LINES`` to the
instruction it is shown, in a fenced block, as a real one replies.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SUITE = Path(__file__).resolve().parents[2] / "examples/tasks/python-bugfix"
SLOW_SECONDS = 5.0
# What a system prompt must say for each convention, lower-cased.
CONVENTIONS = {"none": "return none", "half-open": "half-open", "casefold": "casefold", "first": "first occurrence"}
TASK_CONVENTIONS = {
    "word_count": (),
    # Only in training: a line that fixes it improves a minibatch but not the gate, so the gate rejects it.
    "max_index": ("first",),
    "in_range": ("half-open",),
    "average": ("none",),
    "find_user": ("casefold", "none"),
    "count_between": ("half-open", "none"),
    "last_word": (),
    "clamp_index": ("half-open",),
    "median": ("none",),
    "count_tag": ("casefold", "none"),
    "overlap": ("half-open", "none"),
}
REFLECTION_LINES = (
    "On ties, return the first occurrence.",
    "Use half-open ranges: include the low end, exclude the high end.",
    "Compare strings after strip() and casefold().",
    "On empty or invalid input, return None instead of raising.",
    "Keep the function's name and signature.",
)
FILE = re.compile(r"<<<FILE (?P<name>[^>\n]+)>>>\n(?P<body>.*?)<<<END>>>", re.DOTALL)


class _Handler(BaseHTTPRequestHandler):
    server: OpenAIStub

    def log_message(self, *_: object) -> None:
        return

    def do_POST(self) -> None:
        mode, _, route = self.path.strip("/").partition("/")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        self.server.calls.append((mode, str(body.get("model"))))
        if route != "v1/chat/completions":
            self._send(404, {"error": {"message": "no such route"}})
            return
        messages = body.get("messages", [])
        reflection = not any(message.get("role") == "system" for message in messages)
        if mode.startswith("window-") and not reflection:
            low, high = (int(bound) for bound in mode.removeprefix("window-").split("-"))
            with self.server._lock:
                index = self.server.window_calls
                self.server.window_calls += 1
            if low <= index < high:
                mode = "429"
        if mode in ("429", "401"):
            echoed = f"{json.dumps(body)} {self.headers.get('Authorization')}"
            self._send(int(mode), {"error": {"message": echoed, "type": "error", "code": mode}})
            return
        if mode == "slow":
            time.sleep(SLOW_SECONDS)
        text = self.server.reflect(str(messages[-1]["content"])) if reflection else _policy_reply(messages)
        self._send(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1,
                "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _policy_reply(messages: list[dict[str, Any]]) -> str:
    system = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "system").lower()
    user = str(messages[-1].get("content", ""))
    blocks = []
    for match in FILE.finditer(user):
        module = match["name"].strip().removesuffix(".py")
        if module not in TASK_CONVENTIONS:  # the format's own example, ``<<<FILE name.py>>>``
            continue
        needed = TASK_CONVENTIONS[module]
        if all(CONVENTIONS[name] in system for name in needed):
            code = (SUITE / module / "reference" / f"{module}.py").read_text()
        else:
            code = match["body"]
        blocks.append(f"<<<FILE {match['name']}>>>\n{code.rstrip()}\n<<<END>>>")
    return "\n".join([*blocks, "SUMMARY: fixed the bug I found."])


class OpenAIStub(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.calls: list[tuple[str, str]] = []
        self.reflections = 0
        self.window_calls = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self) -> OpenAIStub:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()

    def url(self, mode: str = "ok") -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/{mode}/v1"

    def reflect(self, prompt: str) -> str:
        current = prompt.split("```\n", 1)[1].split("\n```", 1)[0]
        with self._lock:
            line = REFLECTION_LINES[self.reflections % len(REFLECTION_LINES)]
            self.reflections += 1
        lines = [text for text in current.split("\n") if text.strip()]
        if line not in lines:
            lines.append(line)
        return "Here is the new instruction.\n```\n" + "\n".join(lines) + "\n```"
