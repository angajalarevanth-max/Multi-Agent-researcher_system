"""
Measure how well retrieved sources cover the user query and gate dataset citations.

Salient terms (excluding generic research words) drive :func:`source_supports_query` and
dataset-only coverage. Web rows are always eligible for citations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Terms that appear in almost any vendor doc; do not use alone to claim relevance.
_GENERIC_TERMS = frozenset(
    {
        "pricing",
        "price",
        "prices",
        "compare",
        "comparison",
        "between",
        "versus",
        "report",
        "analysis",
        "cost",
        "costs",
        "features",
        "feature",
        "vendor",
        "vendors",
        "cloud",
        "service",
        "services",
        "platform",
        "platforms",
        "document",
        "documents",
        "month",
        "annual",
        "year",
        "plan",
        "plans",
    }
)


def salient_query_terms(query: str) -> List[str]:
    """Tokens that indicate what the user cares about (excluding generic research words)."""
    q = (query or "").strip().lower()
    found: List[str] = []
    for t in re.findall(r"[a-z0-9]{4,}", q):
        if t not in _GENERIC_TERMS:
            found.append(t)
    for short in ("gcp", "aws", "sap", "ibm"):
        if short in q and short not in found:
            found.append(short)
    # de-dupe preserve order
    seen: set[str] = set()
    out: List[str] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# Hyperscaler / mega-vendor tokens that appear in many unrelated docs; do not let them
# alone justify a dataset citation in multi-entity queries (see ``source_supports_query``).
_OVERLOADED_SALIENCE = frozenset(
    {
        "azure",
        "google",
        "aws",
        "gcp",
        "amazon",
        "oracle",
        "ibm",
        "microsoft",
        "salesforce",
        "digitalocean",
        "meta",
        "facebook",
    }
)


def _source_header_blob(src: dict) -> str:
    """Filename, title, and citation path only (no body) for strict citation anchoring."""
    meta = src.get("metadata") or {}
    fn = str(meta.get("filename", "")).lower()
    title = str(meta.get("title", "")).lower()
    cit = str(meta.get("citation", "")).lower()
    return f"{fn} {title} {cit}"


def _source_blob_sample(src: dict, max_chars: int = 5000) -> str:
    """Lowercased filename, title, citation path, and leading body (for token-in-text checks)."""
    meta = src.get("metadata") or {}
    fn = str(meta.get("filename", "")).lower()
    body = str(src.get("content", ""))[:max_chars].lower()
    title = str(meta.get("title", "")).lower()
    cit = str(meta.get("citation", "")).lower()
    return f"{fn} {title} {cit} {body}"


def source_supports_query(query: str, source: dict) -> bool:
    """
    Whether a source is plausibly relevant for citations.

    Web rows are always kept. Dataset files:
    - One salient token: that token must appear in the file blob.
    - Several tokens: require **all** in blob **unless** at least one token is
      \"distinctive\" (not in ``_OVERLOADED_SALIENCE``). Then a file may cite if
      it matches **any** distinctive token (e.g. PaperMind corpus doc for
      \"revanth vs PaperMind\" even when \"revanth\" is absent), while still
      rejecting files that only match overloaded terms like \"azure\" without a
      distinctive query subject. For **partial** multi-term matches (not every
      salient term in the same file), a distinctive token must appear in the
      **filename, title, or citation path** so tangential body mentions (e.g.
      CSV/PDF cells) do not qualify.
    - If every salient token is overloaded (e.g. Azure vs Google), keep strict
      **all**-must-match for dataset files.
    """
    if not isinstance(source, dict):
        return False
    meta = source.get("metadata") or {}
    stype = meta.get("source_type") or ""
    if stype == "web":
        return True
    salient = salient_query_terms(query)
    blob = _source_blob_sample(source)
    if salient:
        hits = sum(1 for t in salient if t in blob)
        if len(salient) == 1:
            return hits == 1
        distinctive = [t for t in salient if t not in _OVERLOADED_SALIENCE]
        if not distinctive:
            return hits == len(salient)
        if not any(t in blob for t in distinctive):
            return False
        if hits == len(salient):
            return True
        header = _source_header_blob(source)
        return any(t in header for t in distinctive)
    return float(meta.get("relevance_score") or 0) > 0


def filter_sources_for_citations(query: str, sources: List[Any]) -> List[dict]:
    """Drop dataset files that do not match the query; keep all web hits."""
    out: List[dict] = []
    for s in sources or []:
        if isinstance(s, dict) and source_supports_query(query, s):
            out.append(s)
    return out


def dataset_salient_support_ratio(query: str, sources: List[Any]) -> float:
    """Return fraction of salient query terms present in dataset source text."""
    salient = salient_query_terms(query)
    if not salient:
        return 1.0
    ds = [
        s
        for s in (sources or [])
        if isinstance(s, dict) and (s.get("metadata") or {}).get("source_type") == "dataset_file"
    ]
    if not ds:
        return 0.0
    blob = ""
    for s in ds:
        blob += _source_blob_sample(s, max_chars=4000)
    hits = sum(1 for t in salient if t in blob)
    return hits / len(salient)


def web_source_count(sources: List[Any]) -> int:
    """Count sources whose ``metadata.source_type`` is ``web``."""
    n = 0
    for s in sources or []:
        if isinstance(s, dict) and (s.get("metadata") or {}).get("source_type") == "web":
            n += 1
    return n


def combined_coverage_assessment(query: str, sources: List[Any]) -> Dict[str, Any]:
    """
    Lexical ratio over all sources (legacy) plus dataset salient support.
    ``ok`` is False when dataset ignores salient topics and there is no web fallback.
    """
    legacy = query_coverage_ratio(query, sources)
    ds_ratio = dataset_salient_support_ratio(query, sources)
    n_web = web_source_count(sources)
    salient = salient_query_terms(query)

    if not salient:
        ok = True
    elif n_web >= 1:
        # Web was fetched for this query; treat coverage as acceptable for broad questions.
        ok = True
    else:
        ok = ds_ratio >= 0.5

    return {
        "ratio": round(legacy, 4),
        "dataset_salient_ratio": round(ds_ratio, 4),
        "ok": ok,
        "salient_terms": salient,
        "web_count": n_web,
    }


def query_coverage_ratio(query: str, sources: List[Any]) -> float:
    """
    Fraction of query terms (length >= 3) that appear in source text or titles.
    Returns 1.0 if there are no extractable terms.
    """
    q = (query or "").strip().lower()
    terms = [t for t in re.findall(r"[a-z0-9]{3,}", q)]
    if not terms:
        return 1.0

    parts: list[str] = []
    for src in sources or []:
        if isinstance(src, dict):
            parts.append(str(src.get("content", "")))
            meta = src.get("metadata") or {}
            parts.append(str(meta.get("title", "")))
            parts.append(str(meta.get("citation", "")))
        else:
            parts.append(str(src))
    blob = " ".join(parts).lower()

    hits = sum(1 for t in terms if t in blob)
    return hits / len(terms)


def coverage_ok(ratio: float, min_ratio: float = 0.12) -> bool:
    """True if legacy lexical ``query_coverage_ratio`` meets the minimum threshold."""
    return ratio >= min_ratio
