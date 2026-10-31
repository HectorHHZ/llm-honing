"""Harness evolution: the methods that rewrite an agent's harness between releases.

Each method implements :class:`honing.core.EvolveMethod`. ``tasks`` reads a task
suite manifest, ``graders`` grades an episode with the task's own command, and
``endpoints`` maps a checkpoint to an engine Honing did not start.
Runners arrive here as their own issues land (RFC #15).
"""
