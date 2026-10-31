"""Check that the passages an answer quotes actually exist in your sources."""

from .corpus import Corpus
from .gate import AnswerReport, Policy, QuoteResult, Status, check_answer

__version__ = "0.1.0"
__all__ = ["AnswerReport", "Corpus", "Policy", "QuoteResult", "Status", "check_answer"]
