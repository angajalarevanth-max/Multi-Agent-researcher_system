"""
Orchestration entry points for a single research run.

``invoke_research_graph`` builds :class:`state.AgentState`, compiles the LangGraph app from
:mod:`graph`, invokes with LangSmith-friendly config, persists via :func:`services.result_store.save_result`,
and returns the final state plus ``run_id``.

``run_research`` wraps that for HTTP: maps to the dict shape expected by ``POST /v1/research``.
"""

import json
import uuid
from typing import Any, Dict

from config import get_settings
from graph import build_graph, initial_state
from services.llm import get_chat_llm
from services.result_store import save_result
from state import AgentState
from utils.langsmith_setup import apply_langsmith_env
from utils.logging import get_logger

logger = get_logger(__name__)


class EmptyQueryError(ValueError):
    """Raised only when the user query is missing or whitespace; do not conflate with other ValueError."""


def invoke_research_graph(query: str) -> tuple[AgentState, str]:
    """
    Run the compiled LangGraph workflow and return the final ``AgentState`` plus ``run_id``.

    Injects ``run_id`` into ``state.plan`` for tracing. On graph exceptions, saves a failed-state
    artifact when possible, then re-raises.

    Raises:
        EmptyQueryError: If ``query`` is empty after strip (distinct from other ``ValueError``).
    """
    q = (query or "").strip()
    if not q:
        raise EmptyQueryError("query must be a non-empty string")

    apply_langsmith_env(get_settings())

    run_id = str(uuid.uuid4())
    llm = get_chat_llm()
    app = build_graph()
    st0 = initial_state(q, llm)
    st = st0.model_copy(
        update={"plan": {**(st0.plan or {}), "run_id": run_id}},
    )

    invoke_config = {
        "run_name": f"research_{run_id[:8]}",
        "tags": ["multi_agent_researcher", "langgraph"],
        "metadata": {
            "run_id": run_id,
            "query_preview": q[:500],
            "ls_run_name": f"research_{run_id[:8]}",
        },
    }

    logger.info("[%s] Graph invoke start", run_id)
    try:
        raw = app.invoke(st.to_graph_dict(), config=invoke_config)
    except Exception:
        logger.exception("[%s] Graph invoke raised", run_id)
        failed = st.model_copy(
            update={
                "status": "failed",
                "error": "Graph execution error (see server logs).",
                "plan": {**(st.plan or {}), "run_id": run_id},
            }
        )
        try:
            path = save_result(failed, run_id=run_id)
            logger.info("[%s] Saved failed run to %s", run_id, path)
        except Exception:
            logger.exception("[%s] save_result failed after graph error", run_id)
        raise

    logger.info("[%s] Graph invoke end", run_id)

    if isinstance(raw, AgentState):
        final = raw
    elif isinstance(raw, dict):
        final = AgentState.from_graph_result(raw)
    else:
        final = AgentState.from_graph_result(raw.model_dump(mode="python"))

    try:
        out_path = save_result(final, run_id=run_id)
        logger.info("[%s] Saved result to %s", run_id, out_path)
    except Exception:
        logger.exception("[%s] save_result failed", run_id)

    return final, run_id


def run_research(query: str) -> Dict[str, Any]:
    """
    Execute the LangGraph pipeline for a single user query.

    Returns:
        JSON-serializable dict with ``answer``, ``errors``, ``citations``, ``open_questions``,
        and ``metadata`` (``run_id``, ``route``, ``status``, ``confidence``, ``sources_count``,
        coverage fields). Empty query yields a soft response with errors and no ``run_id``.
    """
    try:
        final, run_id = invoke_research_graph(query)
    except EmptyQueryError:
        return {
            "answer": "",
            "errors": ["query must be a non-empty string"],
            "metadata": {"run_id": None},
        }

    err_list = [final.error] if (final.error or "").strip() else []
    answer = (final.final_report or "").strip()
    if not answer and final.extracted_data:
        parts: list[str] = []
        for x in final.extracted_data:
            if isinstance(x, dict):
                parts.append(json.dumps(x, ensure_ascii=False, indent=2))
            elif str(x).strip():
                parts.append(str(x))
        answer = "\n\n".join(parts)
    plan = final.plan or {}
    return {
        "answer": answer,
        "errors": err_list,
        "citations": list(final.citations or []),
        "open_questions": list(final.open_questions or []),
        "metadata": {
            "run_id": run_id,
            "route": plan.get("route"),
            "status": final.status,
            "confidence": final.confidence,
            "sources_count": len(final.sources or []),
            "query_coverage_ok": plan.get("query_coverage_ok"),
            "query_coverage_ratio": plan.get("query_coverage_ratio"),
            "query_dataset_salient_ratio": plan.get("query_dataset_salient_ratio"),
            "web_source_count": plan.get("web_source_count"),
        },
    }
