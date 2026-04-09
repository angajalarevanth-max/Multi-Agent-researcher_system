"""Optional web retrieval via DuckDuckGo (no API key) for a second source channel."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.logging import get_logger

logger = get_logger(__name__)


def fetch_web_sources(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Return source dicts aligned with ``state.sources`` shape: ``id``, ``content``, ``metadata``.

    ``metadata`` includes ``source_type=web``, ``citation`` (URL), ``title``.
    """
    q = (query or "").strip()
    if not q:
        return []

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search not installed; skipping web retrieval")
        return []

    out: List[Dict[str, Any]] = []
    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(q, max_results=max_results)):
                href = (r.get("href") or r.get("url") or "").strip()
                body = (r.get("body") or "").strip()
                title = (r.get("title") or href or f"result_{i}").strip()
                text = f"{title}\n\n{body}".strip()
                if not text:
                    continue
                out.append(
                    {
                        "id": f"web_{i}",
                        "content": text[:12000],
                        "metadata": {
                            "source_type": "web",
                            "citation": href or title,
                            "title": title,
                            "kind": "web_result",
                        },
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Web search failed: %s", exc)
        return []

    return out
