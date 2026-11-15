"""Serve one model to coding agents such as Claude Code and Codex (#188).

The gateway is a LiteLLM proxy in its own environment (``integrations/agent_gateway``), started with a config
Honing writes. It answers Anthropic's ``/v1/messages``, OpenAI's ``/v1/responses`` and ``/v1/chat/completions``,
and sends every request, whatever model it names, to one OpenAI-compatible chat API: an SGLang server holding a
released checkpoint, or any provider. An engineer points an agent at the gateway and keeps the agent, its
sessions and its history; only the model behind it changes.

Keys stay in the environment. The config names the variables that hold the gateway's key and the upstream's key,
and LiteLLM reads them itself, so neither value is written to disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from honing.backends.sglang.engine import die_with_parent
from honing.core import ContractError, PolicyEndpoint, require

PUBLIC_MODEL = "honing"
"""The model name the gateway lists; it also answers any other name an agent asks for."""

KEY_ENV = "HONING_GATEWAY_KEY"
"""The variable that holds the key agents must present, by default."""


@dataclass(frozen=True)
class Upstream:
    """The OpenAI-compatible chat API every request is sent to.

    Args:
        base_url: The API's base URL, ending in ``/v1`` for most servers and providers.
        model: The model name the API answers to.
        api_key_env: The variable holding the API's key, or None for a server that takes none.
    """

    base_url: str
    model: str
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        require(
            self.base_url.startswith(("http://", "https://")), f"base_url must be an HTTP URL, got {self.base_url}"
        )
        require(bool(self.model), "model must be set")

    @classmethod
    def from_endpoint(cls, endpoint: PolicyEndpoint, api_key_env: str | None = None) -> Upstream:
        """Returns the upstream for a policy endpoint, as a campaign's runners reach it.

        Args:
            endpoint: An ``sglang`` or ``openai_chat`` endpoint. An SGLang server's root gains ``/v1``.
            api_key_env: The variable holding the endpoint's key, if it needs one.

        Raises:
            ContractError: The endpoint does not speak OpenAI-compatible chat.
        """
        if endpoint.runtime not in ("sglang", "openai_chat"):
            raise ContractError(f"the gateway needs an OpenAI-compatible endpoint, not a {endpoint.runtime} one")
        base_url = endpoint.base_url.rstrip("/")
        if endpoint.runtime == "sglang" and not base_url.endswith("/v1"):
            base_url += "/v1"
        return cls(base_url=base_url, model=endpoint.served_model, api_key_env=api_key_env)


def proxy_config(upstream: Upstream, key_env: str = KEY_ENV) -> dict[str, object]:
    """Returns the LiteLLM proxy config that sends every request to ``upstream``.

    Args:
        upstream: Where requests go.
        key_env: The variable holding the key agents must present.

    Returns:
        The config, which LiteLLM reads as YAML; JSON is valid YAML.
    """
    params = {
        # As a self-hosted OpenAI-compatible server, so Anthropic's /v1/messages is sent on as chat completions;
        # as plain "openai/" it would go to the upstream's /v1/responses, which SGLang and many providers lack.
        "model": f"hosted_vllm/{upstream.model}",
        "api_base": upstream.base_url,
        # LiteLLM's OpenAI client needs a key even for a server that checks none.
        "api_key": f"os.environ/{upstream.api_key_env}" if upstream.api_key_env else "unused",
        # The same for OpenAI's /v1/responses, which Codex speaks.
        "use_chat_completions_api": True,
    }
    return {
        # The wildcard catches the names agents ask for (claude-sonnet-..., gpt-5-codex) so none needs configuring.
        "model_list": [{"model_name": name, "litellm_params": params} for name in (PUBLIC_MODEL, "*")],
        # Agents send provider-specific parameters (thinking budgets, reasoning effort) an upstream may not accept.
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": f"os.environ/{key_env}"},
    }


class AgentGateway:
    """A running gateway. Start it with :meth:`launch`, which stops it on exit.

    Args:
        url: The gateway's root URL.
        process: The proxy process.
    """

    def __init__(self, url: str, process: subprocess.Popen[bytes]) -> None:
        self.url = url
        self._process = process

    def exited(self) -> bool:
        """Whether the proxy process has exited."""
        return self._process.poll() is not None

    @classmethod
    @contextlib.contextmanager
    def launch(
        cls,
        upstream: Upstream,
        *,
        python: Path,
        work_dir: Path,
        host: str = "127.0.0.1",
        port: int = 4000,
        key_env: str = KEY_ENV,
        start_timeout: float = 120.0,
    ) -> Iterator[AgentGateway]:
        """Starts the proxy and waits until it answers.

        Args:
            upstream: Where requests go.
            python: The interpreter of the gateway's environment (:func:`default_python`).
            work_dir: Where the config and the proxy's log are written.
            host: The address to listen on; ``0.0.0.0`` serves other machines.
            port: The port to listen on.
            key_env: The variable holding the key agents must present.
            start_timeout: How long to wait for the proxy to answer.

        Yields:
            The running gateway.

        Raises:
            ContractError: The environment is not installed, a key variable is not set, the port is taken, or the
                proxy exited or did not answer in time.
        """
        if not python.exists():
            raise ContractError(f"no gateway environment at {python}; see integrations/agent_gateway/README.md")
        for name in (key_env, upstream.api_key_env):
            if name and not os.environ.get(name):
                raise ContractError(f"the gateway needs the environment variable {name}")
        # A server already on the port would answer the health check in the proxy's place.
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1" if host == "0.0.0.0" else host, port)) == 0:
                raise ContractError(f"port {port} is already in use")
        work_dir.mkdir(parents=True, exist_ok=True)
        config = work_dir / "litellm.yaml"
        config.write_text(json.dumps(proxy_config(upstream, key_env), indent=2) + "\n")
        log_path = work_dir / "gateway.log"
        command = [
            str(python), "-m", "litellm.proxy.proxy_cli",
            "--config", str(config), "--host", host, "--port", str(port),
        ]  # fmt: skip
        with log_path.open("ab") as log:
            # Its own process group, so stopping it stops the proxy's workers, and it dies with this process.
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=die_with_parent
            )
            url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
            try:
                deadline = time.monotonic() + start_timeout
                while not _alive(url):
                    if process.poll() is not None:
                        raise ContractError(f"the gateway exited with {process.returncode}; see {log_path}")
                    if time.monotonic() > deadline:
                        raise ContractError(f"the gateway did not answer within {start_timeout:.0f} s; see {log_path}")
                    time.sleep(1)
                yield cls(url, process)
            finally:
                _stop(process)


def default_python(root: Path) -> Path:
    """Returns the gateway environment's interpreter, as ``integrations/agent_gateway/README.md`` creates it.

    Args:
        root: The repository root.
    """
    return (
        root
        / "integrations"
        / "agent_gateway"
        / ".venv"
        / ("Scripts" if sys.platform == "win32" else "bin")
        / "python"
    )


def _stop(process: subprocess.Popen[bytes]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            continue


def _alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health/liveliness", timeout=5) as response:
            return bool(response.status == 200)
    except OSError:
        return False
