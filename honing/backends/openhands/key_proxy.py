"""Forward a containerized agent's model calls to the policy endpoint, adding the API key on the host.

A worker in a container gets this proxy's address instead of the key, so nothing the agent's terminal can
read there, such as ``env`` or ``/proc/1/environ``, holds the key. The agent can still call the model
through the proxy, as it already can through OpenHands; it cannot take the key elsewhere.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers that describe one connection, not the request or response, so they are not forwarded.
_HOP_BY_HOP = frozenset(
    {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding"}
)
# Request headers the proxy sets itself: the key, and what urllib derives from the URL and body.
_REPLACED = frozenset({"authorization", "host", "content-length", "upgrade"})
# Response headers BaseHTTPRequestHandler.send_response writes itself.
_OWN = frozenset({"server", "date"})


class KeyProxy(ThreadingHTTPServer):
    """Serves ``upstream`` on the host's loopback and adds ``Authorization: Bearer <key>`` to each request.

    A request for ``/chat/completions`` goes to ``<upstream>/chat/completions``. Responses, error statuses
    and streamed ones included, are passed back as the upstream sent them. Use it as a context manager,
    which serves from a background thread.

    Args:
        upstream: The endpoint's base URL, such as ``https://openrouter.ai/api/v1``.
        key: The API key.
        timeout_seconds: The limit on one upstream call.
    """

    daemon_threads = True

    def __init__(self, upstream: str, key: str, timeout_seconds: float) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.upstream = upstream.rstrip("/")
        self.key = key
        self.timeout_seconds = timeout_seconds
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        """The loopback port it listens on."""
        return int(self.server_address[1])

    def __enter__(self) -> KeyProxy:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()


class _Handler(BaseHTTPRequestHandler):
    server: KeyProxy

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_BY_HOP and name.lower() not in _REPLACED
        }
        headers["Authorization"] = f"Bearer {self.server.key}"
        request = urllib.request.Request(
            self.server.upstream + self.path, data=body, headers=headers, method=self.command
        )
        try:
            response: HTTPResponse = urllib.request.urlopen(request, timeout=self.server.timeout_seconds)
        except urllib.error.HTTPError as error:  # an error status is the upstream's answer, passed back as is
            response = error  # type: ignore[assignment]
        except OSError as error:
            self.send_error(502, f"the policy endpoint could not be reached: {type(error).__name__}")
            return
        with response:
            # HTTP/1.0: the connection closes after the response, so a body of unknown length needs no chunking.
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in _HOP_BY_HOP and name.lower() not in _OWN:
                    self.send_header(name, value)
            self.end_headers()
            # read1 returns what has arrived, so a streamed response reaches the agent as it is sent.
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
