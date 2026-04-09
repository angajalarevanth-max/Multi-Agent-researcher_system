"""Thin facade over :func:`config.get_llm` so agents import from ``services.llm`` only."""

from typing import Optional

from langchain_openai import AzureChatOpenAI

from config import get_llm


def get_chat_llm(temperature: Optional[float] = None) -> AzureChatOpenAI:
    """Azure-hosted chat model; delegates to ``config.get_llm`` for a single source of truth."""
    return get_llm(temperature=temperature)
