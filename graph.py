"""
LangGraph workflow: orchestrates planner -> retriever -> extractor -> comparator -> writer -> evaluator.

Conditional routing after ``evaluator`` reads ``state.plan["workflow_edge"]`` (set in
``agents.EvaluatorAgent.evaluator_node``): high/reduced confidence end the run; low confidence may
loop once back to ``retriever`` before ``low_confidence_final`` ends the run.

All nodes are wrapped with :func:`utils.agent_tracing.wrap_agent_node` for logging and one retry.
"""

from typing import Any, Union

from langgraph.graph import END, StateGraph

from agents.ComparatorAgent import comparator_node
from agents.EvaluatorAgent import (
    WORKFLOW_HIGH,
    WORKFLOW_LOW_FINAL,
    WORKFLOW_LOW_RETRY,
    WORKFLOW_REDUCED,
    evaluator_node,
)
from agents.ExtractorAgent import extractor_node
from agents.PlannerAgent import planner_node
from agents.WriterAgent import writer_node
from agents.retriever import retriever_node
from state import AgentState
from utils.agent_tracing import wrap_agent_node


def route_after_evaluator(state: Union[AgentState, dict]) -> str:
    """
    Return the next LangGraph edge name from ``state.plan["workflow_edge"]``.

    Unknown values default to ``WORKFLOW_HIGH`` to avoid dangling routes.
    """
    s = state if isinstance(state, AgentState) else AgentState.model_validate(state)
    edge = (s.plan or {}).get("workflow_edge") or WORKFLOW_HIGH
    if edge not in (
        WORKFLOW_HIGH,
        WORKFLOW_REDUCED,
        WORKFLOW_LOW_RETRY,
        WORKFLOW_LOW_FINAL,
    ):
        return WORKFLOW_HIGH
    return edge


def build_graph():
    """
    Build and compile the ``StateGraph(AgentState)`` with linear edges and evaluator conditional edges.

    Returns:
        Compiled graph ready for ``invoke`` with a dict from ``AgentState.to_graph_dict()``.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", wrap_agent_node("planner", planner_node))
    graph.add_node("retriever", wrap_agent_node("retriever", retriever_node))
    graph.add_node("extractor", wrap_agent_node("extractor", extractor_node))
    graph.add_node("comparator", wrap_agent_node("comparator", comparator_node))
    graph.add_node("writer", wrap_agent_node("writer", writer_node))
    graph.add_node("evaluator", wrap_agent_node("evaluator", evaluator_node))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "extractor")
    graph.add_edge("extractor", "comparator")
    graph.add_edge("comparator", "writer")
    graph.add_edge("writer", "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            WORKFLOW_HIGH: END,
            WORKFLOW_REDUCED: END,
            WORKFLOW_LOW_RETRY: "retriever",
            WORKFLOW_LOW_FINAL: END,
        },
    )

    return graph.compile()


def initial_state(query: str, llm: Any) -> AgentState:
    """Seed state for ``app.invoke`` before ``planner`` runs."""
    return AgentState(query=query, llm=llm, status="initialized")
