"""Harness evolution: the methods that rewrite an agent's harness between releases.

Each method implements :class:`honing.core.EvolveMethod`. ``tasks`` reads a task
suite manifest, and ``graders`` grades an episode with the task's own command.
Runners arrive here as their own issues land (RFC #15).
"""
