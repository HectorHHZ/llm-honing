"""Deterministic random A/B decision server for a same-harness model ablation."""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        if self.path != "/v1/decide":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        choices = request["questions"]["control"]["criteria"]
        if set(choices) != {"A", "B"}:
            self.send_error(400, "expected an A/B question")
            return
        bit = hashlib.sha256(request["id"].encode()).digest()[0] & 1
        choice = "AB"[bit]
        body = json.dumps({"answers": {"control": {"choice": choice}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18842), Handler).serve_forever()
