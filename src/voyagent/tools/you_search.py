"""Live web search via the You.com Search API — used specifically as a recency cross-check on the
Eligibility Agent's corpus-grounded visa answer, not as a general-purpose search tool. The visa
corpus has no automated freshness mechanism (a known, documented limitation); this closes part of
that gap by checking whether anything recent appears to contradict a corpus-grounded answer.

Verified live before building against it: POST https://ydc-index.io/v1/search, header
X-API-Key, JSON body {"query": ...}, response shape {"results": {"web": [{"url", "title",
"description", "snippets": [...]}, ...]}}.
"""

import requests

from voyagent.config import settings

SEARCH_URL = "https://ydc-index.io/v1/search"


class YouSearchNotConfigured(RuntimeError):
    """Raised when no You.com API key is set — distinct from a real API failure."""


def search(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of {url, title, snippets} — raises on any failure (missing key, network
    error, bad response). The caller decides what to do about it, same as every other tool."""
    if not settings.you_api_key:
        raise YouSearchNotConfigured("YOU_API_KEY not set in .env")

    response = requests.post(
        SEARCH_URL,
        headers={"X-API-Key": settings.you_api_key, "Content-Type": "application/json"},
        json={"query": query},
        timeout=20,
    )
    response.raise_for_status()
    hits = response.json().get("results", {}).get("web", [])[:max_results]

    return [
        {"url": h.get("url", ""), "title": h.get("title", ""), "snippets": h.get("snippets", [])}
        for h in hits
    ]
