# Multi-agent researcher API

FastAPI service with **LangGraph** orchestration and **Azure OpenAI** for chat.

## Source code (key entry points)

| File | Role |
|------|------|
| [`main.py`](main.py) | HTTP API: `GET /health`, `POST /v1/research` (answer + metadata envelope), `POST /research` (report-centric response). |
| [`services/pipeline.py`](services/pipeline.py) | `invoke_research_graph` (build state, run graph, `run_id`), `run_research` (dict for `/v1/research`). |
| [`graph.py`](graph.py) | LangGraph: `planner` -> `retriever` -> `extractor` -> `comparator` -> `writer` -> `evaluator` and conditional edges to `END` / retry. |
| [`state.py`](state.py) | Pydantic `AgentState` merged across nodes. |
| [`config.py`](config.py) | `pydantic-settings` from `.env` (Azure OpenAI, optional LangSmith). |
| [`services/result_store.py`](services/result_store.py) | `save_result` writes each run to **`results/result_<run_id>.json`** (override directory with env `RESULTS_JSON_DIR`). |

Environment template: [`.env.example`](.env.example). **Keep** your local **`.env`** (copy from the example and edit); it is required to run with Azure OpenAI. **Do not commit** `.env` to git (secrets).

## Layout

- `main.py` - HTTP API (`/health`, `POST /v1/research`, `POST /research`)
- `graph.py` - LangGraph definition and routing after `evaluator`
- `state.py` - Pydantic `AgentState` for all agents
- `config.py` - `pydantic-settings` from `.env`
- `agents/` - graph nodes: `planner`, `retriever`, `extractor`, `comparator`, `writer`, `evaluator` (see `graph.py`)
- `services/` - LLM factory (`llm.py`), `pipeline.py`, `result_store.py`
- `utils/` - logging, LangSmith env, agent node wrappers (`agent_tracing.py`), corpus coverage / citation gating (`coverage.py`), optional DuckDuckGo (`web_search.py`)

## Scenario 3 (assessment alignment)

This service implements a **multi-agent research and reporting** flow aligned with a typical "Scenario 3" spec:

| Expectation | Implementation |
|-------------|------------------|
| **>=2 sources** | Local corpus (multiple files or split chunks) plus optional **web** (DuckDuckGo) when `WEB_SEARCH_ENABLED=true`. |
| **Agreements vs conflicts** | `comparator` outputs `agreements` / `conflicts` / `differences`; the writer surfaces **Agreements & tensions** in the Markdown report. |
| **Structured output** | API returns **report** (or `answer`), **citations**, **open questions**, and coverage-related fields. |
| **Pipeline** | `planner` -> `retriever` -> `extractor` -> `comparator` -> `writer` -> `evaluator` (with optional one-time retrieval retry). |
| **Confidence** | `evaluator` sets a numeric score from **source count** (heuristic, not semantic judging). |
| **Persist output** | Each run writes **`results/result_<run_id>.json`** (see `save_result` in `services/result_store.py`) including report, citations, open questions, comparison, and metadata. |

### What "persistent context" means here

- **Included:** **Per-run** persistence of the final artifact on disk (and optional **LangSmith** traces). Each HTTP request builds fresh `AgentState`; nothing is implied about remembering prior chats unless you add storage.
- **Not included:** **Multi-turn session memory** across requests, or a **LangGraph checkpointer** that resumes an old thread from halfway through the graph.

## Local run

**Keep these for a working setup:** the **`.env`** file, your **`.venv/`** (or other virtualenv) after `pip install`, all **`.py`** sources and **`requirements.txt`**, optional **`run.sh`**, and the **`results/`** folder (created automatically; holds `result_<run_id>.json` from each run). Only **bytecode caches** (`__pycache__/`, `*.pyc`) are safe to delete; Python recreates them.

From this folder (`multi_agent_researcher/`):

```bash
cd multi_agent_researcher
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure OpenAI values

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Example request (`/v1/research`):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?"}'
```

Alternative endpoint (`/research` returns report, confidence, status, and coverage fields directly):

```bash
curl -s -X POST http://127.0.0.1:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query":"What is RAG?"}'
```

## Azure App Service

1. Create a **Linux** App Service, Python 3.10+.
2. Deploy this folder (ZIP, GitHub Actions, etc.).
3. In **Configuration** > **Application settings**, add the same variables as `.env.example` (no quotes needed in portal).
4. **Startup command** (set `PORT` from Azure):

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT:-8000} --timeout 120
```

Or:

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Increase `--timeout` / worker timeout if graph runs are slow.

## Environment variables

See `.env.example`. Never commit `.env`.

Optional file logs: `LOG_TO_FILE=true`.

Optional result directory: `RESULTS_JSON_DIR` (defaults to `results/` under the package root; relative paths are resolved from the package root).

## LangSmith (tracing & monitoring)

1. Create a project and API key at [smith.langchain.com](https://smith.langchain.com).
2. In `.env` set:

   - `LANGSMITH_TRACING=true`
   - `LANGSMITH_API_KEY=<your key>`
   - `LANGSMITH_PROJECT=multi-agent-researcher` (or any project name)

3. Restart the API. On startup you should see a log line: `LangSmith tracing enabled`.

The app sets `LANGCHAIN_TRACING_V2` and related env vars before each graph run so **LangGraph nodes** and **Azure OpenAI** calls are traced. Run metadata includes `run_id` and a query preview. Optional: `LANGSMITH_ENDPOINT` for self-hosted LangSmith.

You can still set `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` manually in the environment instead of using the `LANGSMITH_*` settings above.
