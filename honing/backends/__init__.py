"""Adapters to engines, trainers, and model runtimes outside Honing.

This is the only package allowed to import a training framework or a model
runtime (``tests/contracts/test_boundaries.py``). Each module imports its
runtime lazily, so the base install and the CPU test suite need none of them.
"""
