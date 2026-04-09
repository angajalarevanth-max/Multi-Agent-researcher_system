"""
Planner agent: use the LLM to decide workflow steps and store a strict JSON plan in ``state.plan``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, field_validator
from state import AgentState

from .json_llm import try_effective_llm


class PlannerJSON(BaseModel):
    """Schema for strict JSON returned by the planner LLM (also used with structured output)."""

    tasks: List[str] = Field(
        ...,
        min_length=1,
        description="Ordered workflow steps to execute (short identifiers or labels).",
    )
    strategy: str = Field(
        default="",
        description="Brief rationale for the chosen steps.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Planner confidence in the proposed workflow.",
    )

    @field_validator("tasks", mode="before")
    @classmethod
    def _coerce_tasks(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []


def _coerce_state(state: Union[AgentState, Dict[str, Any]]) -> AgentState:
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


def _normalize_query(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def _parse_planner_json(text: str) -> PlannerJSON:
    cleaned = _strip_code_fence(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Planner output must be a JSON object")
    return PlannerJSON.model_validate(data)


def _default_plan(query: str) -> Dict[str, Any]:
    return {
        "tasks": ["retrieve_sources", "analyze", "synthesize_report"],
        "strategy": "Default workflow: gather sources, analyze, then produce a report.",
        "confidence": 0.35,
    }


def plan_task(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce planner fields for ``state.plan``: ``tasks``, ``strategy``, ``confidence``.

    Uses the chat model with structured output when available; otherwise parses strict JSON
    from the model text. Returns a plain dict suitable for merging into ``AgentState.plan``.
    """
    s = _coerce_state(state)
    query = _normalize_query(s.query)

    if not query:
        return {"tasks": [], "strategy": "", "confidence": 0.0}

    llm = try_effective_llm(s)
    if llm is None:
        p = _default_plan(query)
        p["strategy"] = "(no LLM) " + p["strategy"]
        return p

    system = (
        "You are a planning module for a multi-agent research workflow. "
        "Respond with a single JSON object only. Keys: "
        '"tasks" (JSON array of non-empty strings, at least one step), '
        '"strategy" (string), '
        '"confidence" (number from 0 to 1). '
        "No markdown, no code fences, no extra keys."
    )
    user = f"User query:\n{json.dumps(query, ensure_ascii=False)}\n\nPropose concrete workflow steps."
    prompt = f"{system}\n\n{user}"

    structured = getattr(llm, "with_structured_output", None)
    if callable(structured):
        try:
            bound = structured(PlannerJSON)
            out = bound.invoke(prompt)
            if isinstance(out, PlannerJSON):
                validated = out
            else:
                validated = PlannerJSON.model_validate(out)
            return {
                "tasks": validated.tasks,
                "strategy": validated.strategy,
                "confidence": validated.confidence,
            }
        except Exception:
            pass

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        validated = _parse_planner_json(text)
        if not validated.tasks:
            raise ValueError("empty tasks")
        return {
            "tasks": validated.tasks,
            "strategy": validated.strategy,
            "confidence": validated.confidence,
        }
    except Exception:
        return _default_plan(query)


def planner_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: merge :func:`plan_task` output (tasks, strategy, confidence) into ``state.plan``."""
    s = _coerce_state(state)
    query = _normalize_query(s.query)

    if not query:
        return {
            "plan": {**(s.plan or {}), "tasks": [], "strategy": "", "confidence": 0.0},
            "confidence": 0.0,
            "status": "planner_failed",
            "error": "Empty query.",
        }

    piece = plan_task(s)
    merged = {**(s.plan or {}), **piece}
    return {
        "plan": merged,
        "confidence": float(piece.get("confidence", 0.0)),
        "status": "planned",
        "error": "",
    }
