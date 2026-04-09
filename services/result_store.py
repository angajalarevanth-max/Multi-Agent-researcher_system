"""
Persist workflow outcomes to JSON files for traceability and auditing.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from state import AgentState


def _results_dir() -> Path:
    """Directory for saved JSON; ``RESULTS_JSON_DIR`` env (absolute or relative to package root)."""
    pkg_root = Path(__file__).resolve().parent.parent
    raw = (os.getenv("RESULTS_JSON_DIR") or "results").strip()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (pkg_root / p).resolve()
    return p


def _build_metadata(
    plan: Dict[str, Any],
    sources_count: int,
    confidence: float,
    status: str,
) -> Dict[str, Any]:
    """Subset of ``plan`` + run stats aligned with ``run_research`` ``metadata`` for saved JSON."""
    return {
        "route": plan.get("route"),
        "evaluator_status": plan.get("evaluator_status"),
        "status": status,
        "confidence": confidence,
        "sources_count": sources_count,
        "query_coverage_ok": plan.get("query_coverage_ok"),
        "query_coverage_ratio": plan.get("query_coverage_ratio"),
        "query_dataset_salient_ratio": plan.get("query_dataset_salient_ratio"),
        "web_source_count": plan.get("web_source_count"),
        "query_salient_terms": plan.get("query_salient_terms"),
    }


def save_result(
    state: Union[AgentState, Dict[str, Any]],
    run_id: Optional[str] = None,
) -> Path:
    """
    Write a full run artifact: query, report, confidence, status, error, citations,
    open_questions, comparison, and metadata (aligned with API responses).

    Filename: ``result_<run_id>.json`` (or new UUID if ``run_id`` omitted).
    """
    if isinstance(state, AgentState):
        query = state.query or ""
        report = state.final_report or ""
        confidence = float(state.confidence)
        status = state.status or ""
        error = state.error or ""
        plan = state.plan or {}
        citations = list(state.citations or [])
        open_questions = list(state.open_questions or [])
        comparison = state.comparison or {}
        sources_count = len(state.sources or [])
    else:
        query = str(state.get("query") or "")
        report = str(state.get("final_report") or "")
        confidence = float(state.get("confidence") or 0.0)
        status = str(state.get("status") or "")
        error = str(state.get("error") or "")
        plan = state.get("plan") or {}
        if not isinstance(plan, dict):
            plan = {}
        citations = list(state.get("citations") or [])
        open_questions = list(state.get("open_questions") or [])
        comp = state.get("comparison") or {}
        comparison = comp if isinstance(comp, dict) else {}
        src = state.get("sources") or []
        sources_count = len(src) if isinstance(src, list) else 0

    rid = (run_id or plan.get("run_id") or str(uuid.uuid4())).strip()
    out_dir = _results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"result_{rid}.json"

    # Omit internal retry counter from persisted plan (noise for human readers).
    plan_public: Dict[str, Any] = {
        k: v
        for k, v in plan.items()
        if k
        not in (
            "evaluator_retrieval_retries",
        )
    }

    payload = {
        "run_id": rid,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "report": report,
        "confidence": confidence,
        "status": status,
        "error": error,
        "citations": citations,
        "open_questions": open_questions,
        "comparison": comparison,
        "metadata": _build_metadata(plan, sources_count, confidence, status),
        "plan": plan_public,
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
