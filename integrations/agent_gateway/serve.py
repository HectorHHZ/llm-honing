"""Serve LiteLLM's proxy with only the routes coding agents use; ``honing.backends.agent_gateway`` runs this.

LiteLLM mounts several hundred routes: provider pass-throughs (``/openrouter/...``, ``/anthropic/...``,
``/vllm/...``) that forward any request with whatever provider key the proxy can see, and admin routes. The key
agents present is the proxy's master key, which opens all of them, and LiteLLM's ``allowed_routes`` does not
apply to it. So every request is checked here, before LiteLLM sees it: a path outside :data:`ROUTES` is a 404, and
a websocket is closed. The path is compared after percent-decoding, exactly, so ``/vllm/..%2F..`` is refused too.

The proxy reads its config from ``CONFIG_FILE_PATH``.
"""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import uvicorn
from litellm.proxy.proxy_server import app

ROUTES = frozenset(
    {
        "/v1/messages",  # Claude Code
        "/v1/messages/count_tokens",
        "/v1/responses",  # Codex
        "/v1/chat/completions",  # other OpenAI clients
        "/v1/models",
        "/health/liveliness",
    }
)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]


async def gateway(
    scope: Scope, receive: Callable[[], Awaitable[Message]], send: Callable[[Message], Awaitable[None]]
) -> None:
    """The ASGI application: LiteLLM's, behind the route allowlist."""
    if scope["type"] == "lifespan" or (scope["type"] == "http" and scope["path"] in ROUTES):
        await app(scope, receive, send)
    elif scope["type"] == "http":
        await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"error": "not found"}'})
    else:
        await send({"type": "websocket.close", "code": 1008})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(gateway, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
