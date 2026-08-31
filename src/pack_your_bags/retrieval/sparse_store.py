"""Sparse (BM25) retrieval to complement Pinecone's dense search — the sparse half of hybrid
retrieval. Built directly from the same markdown source files and chunking functions ingest.py
uses, so the sparse corpus is always exactly consistent with what's in Pinecone: no separate
artifact to keep in sync, no extra API calls, purely local computation. Catches exact-term matches
(visa category codes, country names, dollar amounts) that a pure embedding similarity search can
miss."""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=None)
def _get_index(namespace: str) -> tuple[BM25Okapi, tuple[dict, ...]]:
    # Imported lazily: ingest.py imports vector_store.py, which imports this module at load
    # time, so a module-level import here would be circular.
    from pack_your_bags.retrieval import ingest

    docs = ingest.load_documents()
    chunks = tuple(chunk for doc in docs for chunk in ingest.chunk_document(doc))
    bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
    return bm25, chunks


def _matches_filter(metadata: dict, filter_: dict | None) -> bool:
    if not filter_:
        return True
    for key, condition in filter_.items():
        value = metadata.get(key)
        if isinstance(condition, dict) and "$in" in condition:
            allowed = set(condition["$in"])
            if isinstance(value, list):
                if not (set(value) & allowed):
                    return False
            elif value not in allowed:
                return False
        elif value != condition:
            return False
    return True


def sparse_search(query_text: str, namespace: str, filter: dict | None = None, top_k: int = 10) -> list[dict]:
    bm25, chunks = _get_index(namespace)
    scores = bm25.get_scores(_tokenize(query_text))

    candidates = [
        (score, chunk) for score, chunk in zip(scores, chunks)
        if score > 0 and _matches_filter(chunk["metadata"], filter)
    ]
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {"id": chunk["id"], "metadata": chunk["metadata"], "bm25_score": score}
        for score, chunk in candidates[:top_k]
    ]
