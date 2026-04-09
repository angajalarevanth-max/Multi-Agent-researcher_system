"""Service layer: LLM factory, research pipeline, and JSON artifact persistence."""

from .llm import get_chat_llm
from .pipeline import run_research
from .result_store import save_result

__all__ = ["get_chat_llm", "run_research", "save_result"]
