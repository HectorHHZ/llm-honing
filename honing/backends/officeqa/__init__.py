"""OfficeQA (databricks/officeqa) as a Honing task suite: questions over U.S. Treasury Bulletins.

:func:`load_suite` reads the questions a run uses, :class:`OfficeQAGrader`
scores answers with OfficeQA's own scorer, and :mod:`honing.backends.officeqa.fetch`
downloads a few questions and only the bulletins they need.
"""

from honing.backends.officeqa.grader import OfficeQAGrader
from honing.backends.officeqa.suite import CORPUS_DIR, OfficeQAQuestion, OfficeQASuite, load_suite, read_questions

__all__ = ["CORPUS_DIR", "OfficeQAGrader", "OfficeQAQuestion", "OfficeQASuite", "load_suite", "read_questions"]
