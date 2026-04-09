"""
Comparator agent: compare information across ``state.sources``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from pydantic import BaseModel, Field
from state import AgentState

from .json_llm import coerce_state, invoke_structured_or_json, sources_blob, try_effective_llm


class ComparisonJSON(BaseModel):
    """LLM output schema for cross-source comparison (agreements, conflicts, differences)."""

    agreements: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    summary: str = ""


def _fallback_comparison(sources: list) -> Dict[str, Any]:
    n = len(sources or [])
    return {
        "agreements": [f"{n} sources loaded for review."] if n else [],
        "conflicts": [],
        "differences": [f"Source count: {n}."] if n else [],
        "summary": "Heuristic comparison only (no LLM).",
    }


def compare_data(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Return comparison payload: ``agreements``, ``conflicts``, ``differences``, ``summary``.
    """
    s = coerce_state(state)
    sources = s.sources or []
    if not sources:
        return {
            "agreements": [],
            "conflicts": [],
            "differences": [],
            "summary": "No sources to compare.",
        }

    llm = try_effective_llm(s)
    if llm is None:
        return _fallback_comparison(sources)

    blob = sources_blob(sources)
    salient = (s.plan or {}).get("query_salient_terms") or []
    salient_line = (
        f"Query topics to address (from the question): {json.dumps(salient, ensure_ascii=False)}. "
        if salient
        else ""
    )
    system = (
        "You compare multiple sources **in light of the user's question**. Reply with ONE JSON object only. Keys: "
        '"agreements" (array of strings), "conflicts" (array of strings), '
        '"differences" (array of strings), "summary" (string). '
        "No markdown or code fences. "
        "If the sources mainly describe vendors or products **not** named in the query, say that plainly in "
        "`summary` (e.g. corpus compares AcmeDoc vs PaperMind while the user asked about other names). "
        "Do **not** rename or equate query subjects with unrelated vendor names from the documents."
    )
    user = (
        f"User query: {json.dumps(s.query[:500], ensure_ascii=False)}\n"
        f"{salient_line}\n"
        f"Sources:\n{blob[:24000]}"
    )
    prompt = f"{system}\n\n{user}"
    try:
        validated = invoke_structured_or_json(llm, prompt, ComparisonJSON)
        return {
            "agreements": list(validated.agreements or []),
            "conflicts": list(validated.conflicts or []),
            "differences": list(validated.differences or []),
            "summary": validated.summary or "",
        }
    except Exception:
        return _fallback_comparison(sources)


def comparator_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: run :func:`compare_data` into ``comparison``."""
    s = coerce_state(state)
    if not (s.sources or []):
        msg = "No sources to compare."
        err = f"{s.error}; {msg}" if (s.error or "").strip() else msg
        return {
            "comparison": {
                "agreements": [],
                "conflicts": [],
                "differences": [],
                "summary": "No sources to compare.",
            },
            "status": "comparator_failed",
            "error": err,
        }

    payload = compare_data(s)
    return {
        "comparison": payload,
        "status": "compared",
        "error": s.error or "",
    }
