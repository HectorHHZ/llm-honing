"""A gateway that serves one model to coding agents; see ``integrations/agent_gateway/README.md``.

Nothing here imports LiteLLM's proxy, which runs in its own environment.
"""

from honing.backends.agent_gateway.gateway import (
    KEY_ENV,
    PUBLIC_MODEL,
    AgentGateway,
    Upstream,
    default_python,
    proxy_config,
)

__all__ = ["KEY_ENV", "PUBLIC_MODEL", "AgentGateway", "Upstream", "default_python", "proxy_config"]
