"""
Wrap LangGraph agent nodes with logging, one retry on exception, and unified failure state.

On repeated failure: ``status="failed"``, ``error=<message>``. Upstream ``failed`` skips later nodes.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Union

from state import AgentState

from utils.logging import get_logger

# LangGraph node: accepts state dict or AgentState, returns partial state updates.
NodeFn = Callable[[Union[AgentState, Dict[str, Any]]], Dict[str, Any]]

_LOG_VALUE_MAX = 600


def _state_snapshot(state: Union[AgentState, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(state, AgentState):
        return state.model_dump(mode="python")
    return dict(state)


def _is_failed_status(status: str) -> bool:
    return (status or "").strip().lower() == "failed"


def _upstream_failed(state: Union[AgentState, Dict[str, Any]]) -> bool:
    snap = _state_snapshot(state)
    return _is_failed_status(str(snap.get("status") or ""))


def _summarize_output(node_name: str, out: Dict[str, Any]) -> str:
    """Compact, log-safe summary of a node return dict (traceability)."""
    parts: list[str] = []
    for key in sorted(out.keys()):
        if key == "llm":
            continue
        val = out[key]
        if isinstance(val, str):
            clip = val[:_LOG_VALUE_MAX] + ("..." if len(val) > _LOG_VALUE_MAX else "")
            parts.append(f"{key}={clip!r}")
        elif isinstance(val, list):
            parts.append(f"{key}=<list len={len(val)}>")
        elif isinstance(val, dict):
            subk = list(val.keys())[:12]
            parts.append(f"{key}=<dict keys={subk}>")
        else:
            parts.append(f"{key}={val!r}")
    return f"{node_name} output: " + ("; ".join(parts) if parts else "(empty)")


def wrap_agent_node(node_name: str, node_fn: NodeFn) -> NodeFn:
    """
    Log start/end, log structured extraction (return dict summary), retry once on exception.

    After two failures: ``{"status": "failed", "error": ...}``. If a prior node already set
    ``failed``, this node is skipped (no-op merge).
    """
    log = get_logger(f"agent.{node_name}")

    def _run(state: Union[AgentState, Dict[str, Any]]) -> Dict[str, Any]:
        if _upstream_failed(state):
            log.info("node=%s event=skip reason=upstream_failed", node_name)
            return {}

        snap = _state_snapshot(state)
        run_id = (snap.get("plan") or {}).get("run_id") or "-"
        qprev = str(snap.get("query") or "")[:160]
        log.info(
            "node=%s event=start run_id=%s query_preview=%r",
            node_name,
            run_id,
            qprev,
        )
        t0 = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                out = node_fn(state)
                if not isinstance(out, dict):
                    out = {}
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                log.info(
                    "node=%s event=ok run_id=%s elapsed_ms=%s %s",
                    node_name,
                    run_id,
                    elapsed_ms,
                    _summarize_output(node_name, out),
                )
                return out
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "node=%s event=attempt_failed run_id=%s attempt=%s error=%s",
                    node_name,
                    run_id,
                    attempt + 1,
                    exc,
                    exc_info=True,
                )
        err_msg = str(last_exc) if last_exc else "unknown error"
        log.error(
            "node=%s event=failed run_id=%s error=%s",
            node_name,
            run_id,
            err_msg,
        )
        return {"status": "failed", "error": err_msg}

    return _run
