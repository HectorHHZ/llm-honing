"""Coding agents that harness evolution methods run as proposers, each a ``honing.core.ProposerAgent``."""

from honing.harness.proposers.claude_code import DEFAULT_TOOLS, TIMEOUT_EXIT_CODE, ClaudeCodeProposer

__all__ = ["DEFAULT_TOOLS", "TIMEOUT_EXIT_CODE", "ClaudeCodeProposer"]
