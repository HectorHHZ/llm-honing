"""Run the agent gateway until interrupted; see ``integrations/agent_gateway/README.md``.

``python -m honing.backends.agent_gateway --base-url http://gpu-node:30000/v1 --model honed``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from honing.backends.agent_gateway.gateway import KEY_ENV, PUBLIC_MODEL, AgentGateway, Upstream, default_python

_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve one model to Claude Code, Codex, and other coding agents.")
    parser.add_argument(
        "--base-url", required=True, help="the upstream's OpenAI-compatible base URL, usually ending in /v1"
    )
    parser.add_argument("--model", required=True, help="the model name the upstream answers to")
    parser.add_argument("--api-key-env", help="the variable holding the upstream's key, if it needs one")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves other machines")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument("--key-env", default=KEY_ENV, help="the variable holding the key agents must present")
    parser.add_argument(
        "--python", type=Path, default=default_python(_ROOT), help="the gateway environment's interpreter"
    )
    parser.add_argument("--work-dir", type=Path, default=_ROOT / "runs" / "agent-gateway")
    args = parser.parse_args(argv)

    upstream = Upstream(base_url=args.base_url, model=args.model, api_key_env=args.api_key_env)
    with AgentGateway.launch(
        upstream, python=args.python, work_dir=args.work_dir, host=args.host, port=args.port, key_env=args.key_env
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


if __name__ == "__main__":
    main()
