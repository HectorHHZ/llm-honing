"""Serve one model to coding agents such as Claude Code and Codex; see ``integrations/agent_gateway/README.md``.

The gateway is a LiteLLM proxy in its own environment, started with a config Honing writes. It answers
Anthropic's ``/v1/messages`` and OpenAI's ``/v1/responses`` and ``/v1/chat/completions``, and sends every request,
whatever model it names, to one OpenAI-compatible chat API as chat completions.

Run ``python -m honing.backends.agent_gateway --base-url http://gpu-node:30000/v1 --model honed``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from honing.backends.sglang.engine import die_with_parent
from honing.core import ContractError, PolicyEndpoint, require

PUBLIC_MODEL = "honing"
KEY_ENV = "HONING_GATEWAY_KEY"


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


def proxy_config(upstream: Upstream) -> dict[str, object]:
    """Returns the LiteLLM proxy config that sends every request to ``upstream``.

    Args:
        upstream: Where requests go.

    Returns:
        The config. It names the variables the proxy reads the keys from, not their values.
    """
    params = {
        # hosted_vllm and use_chat_completions_api make LiteLLM send /v1/messages and /v1/responses on as chat
        # completions; as "openai/" it would call the upstream's /v1/responses, which SGLang lacks.
        "model": f"hosted_vllm/{upstream.model}",
        "api_base": upstream.base_url,
        "api_key": "os.environ/HONING_UPSTREAM_KEY" if upstream.api_key_env else "unused",
        "use_chat_completions_api": True,
    }
    return {
        # The wildcard answers the model names agents ask for, such as claude-sonnet-4-5 or gpt-5-codex.
        "model_list": [{"model_name": name, "litellm_params": params} for name in (PUBLIC_MODEL, "*")],
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": "os.environ/HONING_GATEWAY_KEY"},
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
        root: Path,
        work_dir: Path,
        host: str = "127.0.0.1",
        port: int = 4000,
        key_env: str = KEY_ENV,
        start_timeout: float = 120.0,
    ) -> Iterator[AgentGateway]:
        """Starts the proxy and waits until it answers.

        The proxy runs ``integrations/agent_gateway/serve.py``, which answers only the agents' routes. It sees
        ``PATH``, ``HOME``, and the two keys, under the names the config reads, and none of the operator's other
        variables.

        Args:
            upstream: Where requests go.
            root: The repository root.
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
        python = default_python(root)
        if not python.exists():
            raise ContractError(f"no gateway environment at {python}; see integrations/agent_gateway/README.md")
        environment = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
        for inside, outside in (("HONING_GATEWAY_KEY", key_env), ("HONING_UPSTREAM_KEY", upstream.api_key_env)):
            if outside is None:
                continue
            if not os.environ.get(outside):
                raise ContractError(f"the gateway needs the environment variable {outside}")
            environment[inside] = os.environ[outside]
        local = "127.0.0.1" if host == "0.0.0.0" else host
        with socket.socket() as probe:
            if probe.connect_ex((local, port)) == 0:
                raise ContractError(f"port {port} is already in use")
        work_dir.mkdir(parents=True, exist_ok=True)
        config = work_dir / "litellm.yaml"
        config.write_text(json.dumps(proxy_config(upstream), indent=2) + "\n")
        environment["CONFIG_FILE_PATH"] = str(config)
        log_path = work_dir / "gateway.log"
        server = root / "integrations" / "agent_gateway" / "serve.py"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [str(python), str(server), "--host", host, "--port", str(port)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=die_with_parent,
            )
            url = f"http://{local}:{port}"
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
    venv = root / "integrations" / "agent_gateway" / ".venv"
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one model to Claude Code, Codex, and other coding agents.")
    parser.add_argument("--base-url", required=True, help="the upstream's OpenAI-compatible base URL")
    parser.add_argument("--model", required=True, help="the model name the upstream answers to")
    parser.add_argument("--api-key-env", help="the variable holding the upstream's key, if it needs one")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves other machines")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--key-env", default=KEY_ENV, help="the variable holding the key agents must present")
    parser.add_argument("--work-dir", type=Path, default=Path("runs/agent-gateway"))
    args = parser.parse_args(argv)

    # A SIGTERM stops the proxy too, as Ctrl-C does; outside Linux nothing else would.
    signal.signal(signal.SIGTERM, _interrupt)
    upstream = Upstream(base_url=args.base_url, model=args.model, api_key_env=args.api_key_env)
    root = Path(__file__).resolve().parents[2]
    with AgentGateway.launch(
        upstream, root=root, work_dir=args.work_dir, host=args.host, port=args.port, key_env=args.key_env
    ) as gateway:
        print(f"Serving {upstream.model} from {upstream.base_url} at {gateway.url}; agents present ${args.key_env}.")
        print(f"Claude Code: ANTHROPIC_BASE_URL={gateway.url} ANTHROPIC_AUTH_TOKEN=${args.key_env} claude")
        print(
            f'Codex: codex -c model_provider=honing -m {PUBLIC_MODEL} -c \'model_providers.honing={{name="Honing", '
            f'base_url="{gateway.url}/v1", env_key="{args.key_env}", wire_api="responses"}}\''
        )
        print(f"Log: {args.work_dir / 'gateway.log'}. Stop with Ctrl-C.", flush=True)
        try:
            while not gateway.exited():
                time.sleep(1)
        except KeyboardInterrupt:
            return
        raise SystemExit(f"the gateway exited; see {args.work_dir / 'gateway.log'}")


def _interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def _stop(process: subprocess.Popen[bytes]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # macOS answers EPERM, not ESRCH, once the whole group has exited.
        with contextlib.suppress(ProcessLookupError, PermissionError):
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


if __name__ == "__main__":
    main()
