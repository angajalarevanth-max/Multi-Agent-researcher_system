"""
Writer agent: produce ``state.final_report``, ``state.open_questions``, and ``state.citations``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field
from state import AgentState

from utils.coverage import filter_sources_for_citations, salient_query_terms

from .json_llm import coerce_state, invoke_structured_or_json, try_effective_llm


class WriterStructured(BaseModel):
    """LLM output schema: Markdown ``report`` plus string ``open_questions``."""

    report: str = ""
    open_questions: List[str] = Field(default_factory=list)


def citations_from_sources(query: str, sources: List[Any]) -> List[Dict[str, Any]]:
    """Build API-ready citations; drop dataset files that do not match query salient terms."""
    filtered = filter_sources_for_citations(query, sources)
    out: List[Dict[str, Any]] = []
    for i, src in enumerate(filtered):
        meta = src.get("metadata") or {}
        label = meta.get("title") or meta.get("filename") or f"source_{i}"
        ref = meta.get("citation") or meta.get("filename") or ""
        stype = meta.get("source_type") or "unknown"
        out.append({"label": str(label), "reference": str(ref), "source_type": str(stype)})
    return out


def _format_extraction(extracted_data: list) -> str:
    if not extracted_data:
        return "(none)"
    first = extracted_data[0]
    if isinstance(first, dict):
        return json.dumps(first, ensure_ascii=False, indent=2)[:12000]
    return "\n".join(str(x) for x in extracted_data)[:12000]


def _format_comparison(comp: Dict[str, Any]) -> str:
    if not comp:
        return "(none)"
    return json.dumps(comp, ensure_ascii=False, indent=2)[:12000]


def _format_web_sources(sources: List[Any], max_per_source: int = 1400) -> str:
    """Compact web hits for the writer prompt (URLs + snippets the model does not see elsewhere)."""
    chunks: List[str] = []
    for i, src in enumerate(sources or []):
        if not isinstance(src, dict):
            continue
        meta = src.get("metadata") or {}
        if meta.get("source_type") != "web":
            continue
        title = str(meta.get("title") or meta.get("citation") or f"web_{i}").strip()
        url = str(meta.get("citation") or "").strip()
        body = str(src.get("content") or "").strip().replace("\n", " ")[:max_per_source]
        chunks.append(f"- **{title}**\n  - URL: {url}\n  - Snippet: {body}")
    if not chunks:
        return "(none - no web results in retrieved sources)"
    return "\n".join(chunks)[:18000]


def _web_sources_text_blob(sources: List[Any]) -> str:
    """Lowercase-safe concatenation of web titles, URLs, and bodies for substring checks."""
    parts: List[str] = []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        meta = src.get("metadata") or {}
        if meta.get("source_type") != "web":
            continue
        parts.append(str(meta.get("title", "")))
        parts.append(str(meta.get("citation", "")))
        parts.append(str(src.get("content", "")))
    return " ".join(parts)


def _salient_terms_missing_from_evidence(
    salient: List[str],
    extraction_block: str,
    comparison_block: str,
    sources: List[Any],
) -> List[str]:
    """
    Salient tokens that do not appear as substrings in corpus (extraction/comparison) or web text.
    Used to block hallucinated vendor/industry claims for names with no supporting hits.
    """
    if not salient:
        return []
    blob = (
        f"{extraction_block}\n{comparison_block}\n{_web_sources_text_blob(sources)}"
    ).lower()
    return [t for t in salient if t not in blob]


def write_structured_report(state: AgentState | Dict[str, Any]) -> WriterStructured:
    """LLM: Markdown report plus explicit open questions (strict JSON / structured output)."""
    s = coerce_state(state)
    extraction_block = _format_extraction(s.extracted_data or [])
    comparison_block = _format_comparison(s.comparison or {})
    cov_ok = (s.plan or {}).get("query_coverage_ok", True)
    cov_ratio = (s.plan or {}).get("query_coverage_ratio")

    llm = try_effective_llm(s)
    if llm is None:
        web_fallback = _format_web_sources(s.sources or [])
        return WriterStructured(
            report=(
                f"# Research summary\n\n## Query\n{s.query}\n\n"
                f"## Extracted highlights\n{extraction_block}\n\n"
                f"## Cross-source comparison\n{comparison_block}\n\n"
                f"## Web results\n{web_fallback}\n"
            ),
            open_questions=[
                "Enable Azure OpenAI in .env for a generated narrative and richer open questions.",
            ],
        )

    cov_note = ""
    if not cov_ok:
        cov_note = (
            f"\nCoverage signal: query term overlap with retrieved text is low (ratio ~{cov_ratio}). "
            "State clearly when evidence does not address the user question; suggest what to add.\n"
        )

    plan = s.plan or {}
    salient = plan.get("query_salient_terms") or salient_query_terms(s.query)
    ds_ratio = plan.get("query_dataset_salient_ratio")
    n_web = plan.get("web_source_count", 0)
    entity_note = (
        "\nENTITY ALIGNMENT (critical):\n"
        f"- Salient topics from the user query: {salient if salient else '(derive from query text)'}.\n"
        "- The extraction/comparison may discuss companies or products that differ from those topics "
        "(e.g. internal docs about AcmeDoc vs PaperMind when the user asked about Google vs Azure).\n"
        "- Do NOT substitute, rename, or conflate vendors. Never treat AcmeDoc as Google or Azure.\n"
        "- If the structured extraction is mostly about entities not named in the query, say that in the Overview, "
        "clearly separate **Corpus / internal documents** vs **Web results**, and use web evidence for query subjects.\n"
        "- Only state pricing or facts for a named vendor when that vendor appears in the evidence block for that claim.\n"
        "- If the **Extraction** or **Comparison** text clearly describes a query-named vendor (e.g. PaperMind AI) "
        "and that vendor also appears in your **Sources** as an internal ``dataset_file`` path, prefer those corpus "
        "facts for that vendor over marketing-style web snippets; use web only to fill gaps or for subjects with "
        "no internal document.\n"
    )
    if isinstance(ds_ratio, (int, float)) and ds_ratio < 1.0 and n_web:
        entity_note += (
            f"- Dataset salient match is incomplete (~{ds_ratio}); prefer **Web results** below for the user's "
            "topics and do not over-claim from internal files that do not substantiate those subjects.\n"
        )

    web_block = _format_web_sources(s.sources or [])
    salient_for_gap = list(salient) if isinstance(salient, list) else salient_query_terms(s.query)
    missing_salient = _salient_terms_missing_from_evidence(
        salient_for_gap,
        extraction_block,
        comparison_block,
        s.sources or [],
    )
    if missing_salient:
        entity_note += (
            "\nEVIDENCE GAPS (mandatory; prevents invented comparisons):\n"
            f"- These query topics do **not** appear anywhere in the Extraction, Comparison, or Web results text "
            f"(substring match, case-insensitive): {missing_salient}.\n"
            "- Do **not** describe them as cloud providers, SaaS vendors, competitors, products, or platforms, "
            "and do **not** assign an industry or service category.\n"
            "- Do **not** state agreements, tensions, or parallels between a well-evidenced subject and these "
            "topics (e.g. do not say both are enterprise cloud providers).\n"
            "- Say clearly they are **not evidenced** in retrieved material; they may be a person's name, typo, "
            "or a non-public or unknown entity.\n"
            "- Add at least one **open_question** asking the user to clarify (e.g. company vs person, intended "
            "product name, or a link).\n"
        )
    if missing_salient and not n_web:
        entity_note += (
            "\nNO WEB CHANNEL (mandatory):\n"
            "- Web results for this run are **empty** (search off or no hits). Do **not** write that web "
            "evidence was used for Azure or other public cloud topics.\n"
            "- The internal corpus does not answer the full query (see evidence gaps). Do **not** fill "
            "**Key points** and **Agreements & tensions** with a long comparative write-up of unrelated "
            "vendors from the extraction (e.g. AcmeDoc vs PaperMind) as if that were the user's question.\n"
            "- At most **one** short bullet may note that those names appear only in unrelated internal documents; "
            "then focus recommendations on clarifying the query or turning web search on.\n"
        )

    prompt = (
        "You are a senior analyst. Produce a structured result for the user query.\n"
        "- report: a clear Markdown document with sections such as Overview, Key points, "
        "Agreements & tensions, Recommendations, and Sources (brief).\n"
        "- open_questions: array of strings listing unresolved gaps, missing data, or follow-ups "
        "(include gaps for any query subject missing from evidence).\n"
        f"{cov_note}{entity_note}"
        "Ground claims in the Extraction and Comparison blocks **and** in the **Web results** block when present. "
        "If Web results exist, the Sources section must acknowledge them (titles/URLs); do not claim there is "
        "no web evidence. If web and dataset disagree, say so.\n"
        "Respond as ONE JSON object only with keys \"report\" (string) and \"open_questions\" (array of strings). "
        "No markdown fences.\n\n"
        f"User query:\n{json.dumps(s.query, ensure_ascii=False)}\n\n"
        f"Extraction (JSON or text):\n{extraction_block}\n\n"
        f"Comparison (JSON):\n{comparison_block}\n\n"
        f"Web results (retrieved for this query; use for facts about web-only subjects):\n{web_block}\n"
    )

    try:
        return invoke_structured_or_json(llm, prompt, WriterStructured)
    except Exception:
        raw = llm.invoke(prompt)
        text = getattr(raw, "content", None) or str(raw)
        return WriterStructured(report=text.strip(), open_questions=[])


def write_report(state: AgentState | Dict[str, Any]) -> str:
    """Backward-compatible: report body only."""
    return write_structured_report(state).report


def writer_node(state: AgentState | Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: build ``final_report``, ``citations`` (query-filtered), and ``open_questions``.

    Prepends a coverage warning to the report when ``plan.query_coverage_ok`` is false.
    """
    s = coerce_state(state)
    if not (s.extracted_data or []) and not (s.comparison or {}):
        msg = "Nothing to write."
        err = f"{s.error}; {msg}" if (s.error or "").strip() else msg
        return {
            "final_report": "",
            "citations": citations_from_sources(s.query, s.sources or []),
            "open_questions": ["No extraction or comparison available to summarize."],
            "status": "writer_failed",
            "error": err,
        }

    structured = write_structured_report(s)
    report = structured.report.strip()
    if not (s.plan or {}).get("query_coverage_ok", True):
        report = (
            "## Coverage note\n"
            "Retrieved evidence may not fully match your query (low lexical overlap with source text). "
            "Validate conclusions against the citations below or refine the question.\n\n"
        ) + report

    cites = citations_from_sources(s.query, s.sources or [])
    oq = [str(x).strip() for x in (structured.open_questions or []) if str(x).strip()]
    if not (s.plan or {}).get("query_coverage_ok", True):
        oq.insert(
            0,
            "Consider adding sources that explicitly mention your query topics, or narrow the question to match the loaded corpus.",
        )

    return {
        "final_report": report,
        "citations": cites,
        "open_questions": oq,
        "status": "written",
        "error": s.error or "",
    }
