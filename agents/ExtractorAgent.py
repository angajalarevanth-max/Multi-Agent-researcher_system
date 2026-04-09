"""
Extractor agent: derive key insights, facts, and trends from ``state.sources``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field
from state import AgentState

from .json_llm import coerce_state, invoke_structured_or_json, sources_blob, try_effective_llm


class ExtractionJSON(BaseModel):
    """LLM output schema for structured highlights from ``state.sources``."""

    key_insights: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    trends: List[str] = Field(default_factory=list)


def _fallback_extraction(sources: list) -> Dict[str, Any]:
    lines: List[str] = []
    for src in sources or []:
        if isinstance(src, dict):
            lines.append(str(src.get("content", ""))[:400])
        else:
            lines.append(str(src)[:400])
    blob = " ".join(lines)[:2000]
    sentences = [x.strip() for x in blob.replace("\n", " ").split(".") if len(x.strip()) > 20][:5]
    return {
        "key_insights": sentences[:2] if sentences else ["Insufficient source text."],
        "facts": sentences[2:4] if len(sentences) > 2 else sentences[:1],
        "trends": sentences[4:5] if len(sentences) > 4 else [],
    }


def extract_data(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Return extraction payload: ``key_insights``, ``facts``, ``trends`` (lists of strings).
    """
    s = coerce_state(state)
    sources = s.sources or []
    if not sources:
        return {"key_insights": [], "facts": [], "trends": []}

    llm = try_effective_llm(s)
    if llm is None:
        return _fallback_extraction(sources)

    blob = sources_blob(sources)
    salient = (s.plan or {}).get("query_salient_terms") or []
    salient_line = (
        f"Prioritize material tied to these query topics when present in the text: "
        f"{json.dumps(salient, ensure_ascii=False)}. "
        if salient
        else ""
    )
    system = (
        "You extract structured highlights from the sources **for the user's question**. Reply with ONE JSON object only. "
        'Keys: "key_insights" (array of strings), "facts" (array of strings), '
        '"trends" (array of strings). No markdown or code fences. '
        "If the documents center on different product or company names than the query topics, state that in "
        "`key_insights` and extract only what is relevant to the query topics where they appear; do not invent "
        "connections between unrelated names."
    )
    user = (
        f"User query: {json.dumps(s.query[:500], ensure_ascii=False)}\n"
        f"{salient_line}\n"
        f"Sources:\n{blob[:24000]}"
    )
    prompt = f"{system}\n\n{user}"
    try:
        validated = invoke_structured_or_json(llm, prompt, ExtractionJSON)
        return {
            "key_insights": list(validated.key_insights or []),
            "facts": list(validated.facts or []),
            "trends": list(validated.trends or []),
        }
    except Exception:
        return _fallback_extraction(sources)


def extractor_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: run :func:`extract_data` and store a single dict in ``extracted_data``."""
    s = coerce_state(state)
    if not (s.sources or []):
        msg = "No sources to extract."
        err = f"{s.error}; {msg}" if (s.error or "").strip() else msg
        return {
            "extracted_data": [],
            "status": "extractor_failed",
            "error": err,
        }

    payload = extract_data(s)
    return {
        "extracted_data": [payload],
        "status": "extracted",
        "error": s.error or "",
    }
