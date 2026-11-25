"""Honing's recipe for Reef (#137): its stages and a training backend that spools job data for Honing.

Reef loads these by import path in its own Python 3.12 environment. This
package imports Reef and never Honing. The record format it reads is defined in
``honing/backends/reef/records.py``.
"""
