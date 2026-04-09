"""
Shared agent state (Pydantic) for LangGraph workflows.

LangGraph merges node return values into this schema. Use ``model_dump(mode="python")``
when passing into ``invoke`` if you need raw dicts with non-JSON values (e.g. ``llm``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """
    Workflow state passed between LangGraph nodes; partial dict returns from each node are merged.

    **Fields**

    - ``query`` - user question for this HTTP request (single-turn).
    - ``plan`` - merged planner output, ``run_id``, coverage metrics, evaluator ``workflow_edge``, etc.
    - ``sources`` - retriever rows: ``id``, ``content``, ``metadata`` (dataset_file or web).
    - ``extracted_data`` - extractor payload (typically one dict with insights/facts/trends).
    - ``comparison`` - comparator agreements/conflicts/differences/summary.
    - ``citations`` - writer-built list for the API (label, reference, source_type).
    - ``open_questions`` - writer-suggested follow-ups.
    - ``confidence`` / ``status`` - evaluator output (source-count heuristic).
    - ``final_report`` - Markdown report body.
    - ``error`` - non-fatal node messages; graph may still complete.
    - ``llm`` - runtime chat model reference (not serialized to clients).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = ""
    plan: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Any] = Field(default_factory=list)
    extracted_data: List[Any] = Field(default_factory=list)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    final_report: str = ""
    error: str = ""
    status: str = "initialized"

    # Runtime dependency (not returned to API clients)
    llm: Optional[Any] = Field(default=None)

    def to_graph_dict(self) -> Dict[str, Any]:
        """Serialize for LangGraph ``invoke`` (keeps ``llm`` reference)."""
        return self.model_dump(mode="python")

    @classmethod
    def from_graph_result(cls, data: Dict[str, Any]) -> "AgentState":
        """Normalize graph output back into a model."""
        return cls.model_validate(data)
