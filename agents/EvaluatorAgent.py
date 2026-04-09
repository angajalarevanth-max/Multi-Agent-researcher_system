"""
Evaluator agent: set ``confidence`` and ``status`` from source count (and optional comparison context).

Rules:
- fewer than 2 sources -> low confidence
- more than 2 sources -> reduced confidence
- exactly 2 sources -> high confidence
"""

from __future__ import annotations

from typing import Any, Dict, Union

from state import AgentState

from .json_llm import coerce_state

# Numeric targets for the three branches
CONFIDENCE_LOW = 0.35
CONFIDENCE_HIGH = 0.9
CONFIDENCE_REDUCED = 0.55

STATUS_LOW = "low_confidence"
STATUS_HIGH = "high_confidence"
STATUS_REDUCED = "reduced_confidence"

# Conditional routing targets for ``graph.add_conditional_edges`` (evaluator -> next node)
WORKFLOW_HIGH = "high_confidence"
WORKFLOW_REDUCED = "reduced_confidence"
WORKFLOW_LOW_RETRY = "low_confidence_retry"
WORKFLOW_LOW_FINAL = "low_confidence_final"


def evaluate_confidence(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    Read ``state.sources`` (and ``state.comparison`` for future use only).

    Returns ``confidence`` (float), ``status`` (string), and ``route`` (mirrors status for ``plan``).
    """
    s = coerce_state(state)
    n = len(s.sources or [])

    if n < 2:
        confidence = CONFIDENCE_LOW
        status = STATUS_LOW
    elif n > 2:
        confidence = CONFIDENCE_REDUCED
        status = STATUS_REDUCED
    else:
        confidence = CONFIDENCE_HIGH
        status = STATUS_HIGH

    return {
        "confidence": confidence,
        "status": status,
        "route": status,
    }


def evaluator_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: set ``confidence``, ``status``, and ``plan.workflow_edge`` from ``len(sources)``.

    Low source count triggers at most one ``low_confidence_retry`` (increment
    ``plan.evaluator_retrieval_retries``); otherwise ``low_confidence_final`` ends the run.
    """
    s = coerce_state(state)
    out = evaluate_confidence(s)
    status = out["status"]
    prev = s.plan or {}
    retries = int(prev.get("evaluator_retrieval_retries", 0) or 0)

    if status == STATUS_HIGH:
        workflow_edge = WORKFLOW_HIGH
        new_retries = 0
    elif status == STATUS_REDUCED:
        workflow_edge = WORKFLOW_REDUCED
        new_retries = 0
    else:
        # low_confidence: one extra retrieval pass, then finish
        if retries < 1:
            workflow_edge = WORKFLOW_LOW_RETRY
            new_retries = retries + 1
        else:
            workflow_edge = WORKFLOW_LOW_FINAL
            new_retries = retries

    plan = {
        **prev,
        "route": out["route"],
        "evaluator_status": status,
        "workflow_edge": workflow_edge,
        "evaluator_retrieval_retries": new_retries,
    }
    return {
        "plan": plan,
        "confidence": out["confidence"],
        "status": status,
        "error": s.error or "",
    }


def evaluate_data(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """Alias for :func:`evaluate_confidence` (backward-compatible name)."""
    return evaluate_confidence(state)
