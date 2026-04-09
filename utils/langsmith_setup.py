"""
Enable LangSmith tracing for LangGraph / LangChain when configured.

LangChain reads ``LANGCHAIN_TRACING_V2``, ``LANGSMITH_API_KEY`` (or ``LANGCHAIN_API_KEY``),
and ``LANGCHAIN_PROJECT`` from ``os.environ``. We sync ``Settings`` into the environment,
and load ``.env`` once so legacy ``LANGCHAIN_*`` keys in the file are visible to the tracer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from utils.logging import get_logger

if TYPE_CHECKING:
    from config import Settings

logger = get_logger(__name__)

_dotenv_loaded = False
_logged_enabled = False


def _load_dotenv_once() -> None:
    """Put ``.env`` into ``os.environ`` (``override=False`` keeps existing exports)."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env", override=False)


def apply_langsmith_env(settings: "Settings") -> bool:
    """
    Ensure LangSmith env vars are set, then return whether tracing should run.

    Priority: ``LANGSMITH_TRACING`` + ``LANGSMITH_API_KEY`` from settings, then existing
    ``LANGCHAIN_TRACING_V2`` + API key already in the environment (e.g. from ``.env``).
    """
    global _logged_enabled

    _load_dotenv_once()

    if getattr(settings, "LANGSMITH_TRACING", False):
        key = (getattr(settings, "LANGSMITH_API_KEY", None) or "").strip()
        if not key:
            logger.warning(
                "LANGSMITH_TRACING is true but LANGSMITH_API_KEY is empty; LangSmith disabled"
            )
        else:
            project = (
                getattr(settings, "LANGSMITH_PROJECT", None) or "multi-agent-researcher"
            ).strip()
            endpoint = (getattr(settings, "LANGSMITH_ENDPOINT", None) or "").strip()
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = key
            os.environ["LANGCHAIN_API_KEY"] = key
            os.environ["LANGCHAIN_PROJECT"] = project
            if endpoint:
                os.environ["LANGSMITH_ENDPOINT"] = endpoint.rstrip("/")

    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
    key_eff = (
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
    ).strip()

    if tracing_on and key_eff:
        if not _logged_enabled:
            logger.info(
                "LangSmith tracing active (project=%s)",
                os.getenv("LANGCHAIN_PROJECT", "multi-agent-researcher"),
            )
            _logged_enabled = True
        return True

    return False
