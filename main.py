"""
HTTP API for the multi-agent research LangGraph workflow.

**Endpoints**

- ``GET /health`` - liveness check.
- ``POST /research`` - runs the graph via :func:`services.pipeline.invoke_research_graph` and returns
  report-centric fields (Markdown report, confidence, status, citations, coverage metrics).
- ``POST /v1/research`` - same pipeline via :func:`services.pipeline.run_research`; response bundles
  ``answer``, ``errors``, ``metadata``, ``citations``, and ``open_questions`` for assessment-style clients.

Startup (lifespan) configures logging and optional LangSmith environment variables from settings.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import get_settings
from services.pipeline import invoke_research_graph, run_research
from utils.langsmith_setup import apply_langsmith_env
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and LangSmith env once when the ASGI app starts."""
    settings = get_settings()
    setup_logging(level=settings.LOG_LEVEL)
    apply_langsmith_env(settings)
    logger.info("Starting %s", settings.APP_NAME)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Multi-agent researcher",
    description="LangGraph + Azure OpenAI research API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return 500 JSON for uncaught exceptions (logged with stack trace)."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


class ResearchRequest(BaseModel):
    """Body for research POST endpoints: natural-language question (1-16k chars after strip)."""

    query: str = Field(..., min_length=1, max_length=16_000)

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v:
            raise ValueError("query must be a non-empty string")
        return v


class ResearchResponse(BaseModel):
    """Shape returned by ``POST /v1/research`` (answer text plus citations and run metadata)."""

    answer: str
    errors: list
    metadata: dict
    citations: list = Field(default_factory=list)
    open_questions: list = Field(default_factory=list)


class ResearchWorkflowResponse(BaseModel):
    """Response for ``POST /research`` (final graph state fields)."""

    report: str
    confidence: float
    status: str
    citations: list = Field(default_factory=list)
    open_questions: list = Field(default_factory=list)
    query_coverage_ok: bool = True
    query_coverage_ratio: Optional[float] = None
    query_dataset_salient_ratio: Optional[float] = None
    web_source_count: Optional[int] = None


@app.get("/health")
def health():
    """Return ``{"status": "ok"}`` for load balancers and probes."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchWorkflowResponse)
def research_workflow(req: ResearchRequest):
    """
    Run the full LangGraph workflow: initialize ``AgentState``, invoke graph, return report and metrics.
    """
    try:
        final, _run_id = invoke_research_graph(req.query)
        plan = final.plan or {}
        return ResearchWorkflowResponse(
            report=final.final_report or "",
            confidence=float(final.confidence),
            status=final.status or "",
            citations=list(final.citations or []),
            open_questions=list(final.open_questions or []),
            query_coverage_ok=bool(plan.get("query_coverage_ok", True)),
            query_coverage_ratio=plan.get("query_coverage_ratio"),
            query_dataset_salient_ratio=plan.get("query_dataset_salient_ratio"),
            web_source_count=plan.get("web_source_count"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Research workflow failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Workflow execution failed",
        ) from e


@app.post("/v1/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the research pipeline and map the final graph state into the v1 API envelope."""
    try:
        result = run_research(req.query)
        return ResearchResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
