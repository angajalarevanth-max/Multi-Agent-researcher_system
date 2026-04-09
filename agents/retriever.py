"""
Retriever agent: load local corpus files, score by query relevance, populate ``state.sources``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from PyPDF2 import PdfReader

from config import get_research_sources_dir, get_retriever_paths
from utils.web_search import fetch_web_sources
from state import AgentState
from utils.coverage import combined_coverage_assessment

MAX_CHARS_PER_SOURCE = 16_000
MIN_SOURCES = 2
MAX_SOURCES = 8
# Extensions handled as specified: .txt raw, .pdf PyPDF2, .html text, .csv pandas
_READABLE_EXTENSIONS = {".txt", ".pdf", ".html", ".htm", ".csv"}


def _coerce(state: Union[AgentState, Dict[str, Any]]) -> AgentState:
    if isinstance(state, AgentState):
        return state
    return AgentState.model_validate(state)


def _normalize_query(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _query_terms(query: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", query.lower()) if t]


def _load_file_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".html", ".htm"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        reader = PdfReader(str(path))
        parts: List[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    if ext == ".csv":
        df = pd.read_csv(path)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()
    return ""


def _relevance_score(query: str, filename: str, body: str) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    fn = filename.lower()
    blob = body.lower()
    score = 0.0
    for t in terms:
        c = blob.count(t)
        if c:
            score += min(c, 50) + 0.5 * len(t)
        if t in fn:
            score += 3.0
    return score


def _split_single_document(filename: str, ext: str, content: str, base_score: float) -> List[Dict[str, Any]]:
    """When only one file exists, produce two sources so downstream always has >= 2."""
    content = (content or "").strip()
    if len(content) < 80:
        dup = content or "(empty)"
        return [
            _source_dict(0, dup, filename, ext, base_score, chunk=1),
            _source_dict(1, dup, filename, ext, base_score, chunk=2),
        ]
    mid = len(content) // 2
    br = content.rfind("\n\n", 0, min(mid + 800, len(content)))
    if br < 40:
        br = mid
    part_a, part_b = content[:br].strip(), content[br:].strip()
    if not part_b:
        part_b = part_a[-(len(part_a) // 2) :] or part_a
        part_a = part_a[: len(part_a) - len(part_b)].strip() or part_a
    return [
        _source_dict(0, part_a[:MAX_CHARS_PER_SOURCE], filename, ext, base_score, chunk=1),
        _source_dict(1, part_b[:MAX_CHARS_PER_SOURCE], filename, ext, base_score, chunk=2),
    ]


def _source_dict(
    idx: int,
    content: str,
    filename: str,
    ext: str,
    score: float,
    *,
    chunk: int | None = None,
    path: Path | None = None,
) -> Dict[str, Any]:
    citation = str(path.resolve()) if path is not None else filename
    meta: Dict[str, Any] = {
        "filename": filename,
        "extension": ext,
        "relevance_score": round(score, 4),
        "kind": "file_source",
        "source_type": "dataset_file",
        "citation": citation,
        "title": filename,
    }
    if chunk is not None:
        meta["chunk"] = chunk
    return {"id": idx, "content": content, "metadata": meta}


def _select_query_aware(
    query: str, loaded: List[Tuple[Path, str, str, str]]
) -> List[Dict[str, Any]]:
    """
    ``loaded`` entries: (path, filename, extension, content).
    Prefer higher relevance; always return at least ``MIN_SOURCES`` when enough files exist.
    """
    scored: List[Tuple[float, Path, str, str, str]] = []
    for path, filename, ext, text in loaded:
        sc = _relevance_score(query, filename, text)
        scored.append((sc, path, filename, ext, text))

    scored.sort(key=lambda x: (-x[0], x[2].lower()))

    if len(scored) == 1:
        _, _path, filename, ext, text = scored[0]
        parts = _split_single_document(filename, ext, text, scored[0][0])
        return parts

    positive = [(s, p, f, e, t) for s, p, f, e, t in scored if s > 0]
    chosen: List[Tuple[float, Path, str, str, str]] = []
    if len(positive) >= MIN_SOURCES:
        chosen = positive[:MAX_SOURCES]
    elif positive:
        chosen = list(positive)
        for row in scored:
            if row in chosen:
                continue
            chosen.append(row)
            if len(chosen) >= MIN_SOURCES:
                break
    else:
        chosen = scored[: max(MIN_SOURCES, min(len(scored), MAX_SOURCES))]

    chosen = chosen[:MAX_SOURCES]
    out: List[Dict[str, Any]] = []
    for i, (s, _path, filename, ext, text) in enumerate(chosen):
        body = (text or "").strip()[:MAX_CHARS_PER_SOURCE]
        out.append(_source_dict(i, body, filename, ext, s, path=_path))
    return out


def retrieve_sources(state: AgentState | Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load corpus from the configured research sources directory, rank by query relevance,
    and return at least two sources when multiple files exist (or two chunks from one file).

    Writes shape matches ``AgentState.sources``: list of dicts with ``id``, ``content``, ``metadata``.
    """
    s = _coerce(state)
    query = _normalize_query(s.query)
    if not query:
        return []

    root = get_research_sources_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"Research sources directory not found: {root}")

    loaded: List[Tuple[Path, str, str, str]] = []
    for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in _READABLE_EXTENSIONS:
            continue
        try:
            text = _load_file_text(path)
        except Exception:
            continue
        loaded.append((path, path.name, ext, text))

    if not loaded:
        raise FileNotFoundError(f"No readable sources (.txt/.pdf/.html/.csv) in {root}")

    return _select_query_aware(query, loaded)


def retriever_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: load corpus + optional web hits, compute coverage metrics into ``plan``,
    and set ``sources`` / ``extracted_data`` / ``status``.
    """
    s = _coerce(state)
    query = _normalize_query(s.query)
    plan: Dict[str, Any] = {**(s.plan or {}), "query_normalized": query}

    if not query:
        return {
            "plan": plan,
            "sources": [],
            "extracted_data": [],
            "status": "retriever_failed",
            "error": "Empty query.",
        }

    try:
        sources = retrieve_sources(s)
    except FileNotFoundError as e:
        return {
            "plan": plan,
            "sources": [],
            "extracted_data": [],
            "status": "retriever_failed",
            "error": str(e),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "plan": plan,
            "sources": [],
            "extracted_data": [],
            "status": "retriever_failed",
            "error": f"Retriever error: {e}",
        }

    paths_cfg = get_retriever_paths()
    if paths_cfg.WEB_SEARCH_ENABLED:
        web_items = fetch_web_sources(query, max_results=paths_cfg.WEB_SEARCH_MAX_RESULTS)
        base_id = len(sources)
        for j, item in enumerate(web_items):
            row = dict(item)
            row["id"] = base_id + j
            sources.append(row)

    assess = combined_coverage_assessment(query, sources)
    plan = {
        **plan,
        "query_coverage_ratio": assess["ratio"],
        "query_coverage_ok": assess["ok"],
        "query_dataset_salient_ratio": assess["dataset_salient_ratio"],
        "query_salient_terms": assess["salient_terms"],
        "web_source_count": assess["web_count"],
    }

    extracted = [str(item.get("content", "")).strip() for item in sources if str(item.get("content", "")).strip()]
    return {
        "plan": plan,
        "sources": sources,
        "extracted_data": extracted,
        "status": "retrieved",
        "error": "",
    }
