"""
Shared helpers for LangGraph agents: state coercion and strict JSON from the chat model.

``invoke_structured_or_json`` prefers ``llm.with_structured_output(Model)`` when the provider supports
it; otherwise it parses a single JSON object from ``llm.invoke`` text (stripping optional fences).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Type, TypeVar, Union

from pydantic import BaseModel
from state import AgentState

T = TypeVar("T", bound=BaseModel)


def coerce_state(state: Union[AgentState, Dict[str, Any]]) -> AgentState:
    """Normalize dict or model to :class:`state.AgentState`."""
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


def try_effective_llm(s: AgentState) -> Any:
    """
    Return a chat model with ``invoke`` / ``with_structured_output``.

    LangGraph merges can replace ``state.llm`` with a non-invokable value; fall back to
    ``get_chat_llm()`` when needed. Returns ``None`` if no client is available (e.g. missing config).
    """
    llm = s.llm
    if llm is not None and callable(getattr(llm, "invoke", None)):
        return llm
    try:
        from services.llm import get_chat_llm

        return get_chat_llm()
    except Exception:
        return None


def strip_code_fence(raw: str) -> str:
    """Remove leading/trailing markdown code fences (e.g. json) if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def parse_json_model(text: str, model: Type[T]) -> T:
    """Parse JSON object from ``text`` and validate with ``model.model_validate``."""
    cleaned = strip_code_fence(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return model.model_validate(data)


def invoke_structured_or_json(llm: Any, prompt: str, model: Type[T]) -> T:
    """
    Invoke the chat model and validate the result as ``model`` (subclass of ``BaseModel``).

    Tries structured output binding first; on failure falls back to raw text JSON parsing via
    :func:`parse_json_model`.
    """
    structured = getattr(llm, "with_structured_output", None)
    if callable(structured):
        try:
            out = structured(model).invoke(prompt)
            if isinstance(out, model):
                return out
            return model.model_validate(out)
        except Exception:
            pass
    response = llm.invoke(prompt)
    text = getattr(response, "content", None) or str(response)
    return parse_json_model(text, model)


def sources_blob(sources: list, max_per_source: int = 6000) -> str:
    """Concatenate source bodies into a single prompt string (filename headers + truncated content)."""
    parts: list[str] = []
    for i, src in enumerate(sources or []):
        if isinstance(src, dict):
            meta = src.get("metadata") or {}
            fn = meta.get("filename", f"source_{i}")
            content = str(src.get("content", ""))[:max_per_source]
        else:
            fn = f"source_{i}"
            content = str(src)[:max_per_source]
        parts.append(f"### {fn}\n{content}")
    return "\n\n".join(parts) if parts else "(no sources)"
