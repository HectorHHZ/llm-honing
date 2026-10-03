"""OpenHands as a Honing harness agent.

:class:`OpenHandsEpisodeRunner` runs one OpenHands conversation per episode in
OpenHands' own environment (``integrations/openhands``), with the harness's
instructions, skills, tools, and MCP servers delivered to the agent. Nothing
here imports OpenHands.
"""

from honing.backends.openhands.runner import (
    AGENT,
    AGENT_SETTINGS,
    DEFAULT_TOOLS,
    MCP_SETTINGS,
    OpenHandsDeliveryProbe,
    OpenHandsEpisodeRunner,
    ToolPolicy,
    default_python,
)

__all__ = [
    "AGENT",
    "AGENT_SETTINGS",
    "DEFAULT_TOOLS",
    "MCP_SETTINGS",
    "OpenHandsDeliveryProbe",
    "OpenHandsEpisodeRunner",
    "ToolPolicy",
    "default_python",
]
