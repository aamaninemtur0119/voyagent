from pinecone import Pinecone

from pack_your_bags.config import settings
from pack_your_bags.retrieval.sparse_store import sparse_search

NAMESPACE = "entry_requirements"  # default namespace, kept for backward compatibility
RERANK_MODEL = "bge-reranker-v2-m3"
RRF_K = 60  # standard Reciprocal Rank Fusion constant

_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = _pc.Index(settings.pinecone_index_name)


def upsert_chunks(vectors: list[dict], namespace: str = NAMESPACE) -> None:
    _index.upsert(vectors=vectors, namespace=namespace)


def clear_namespace(namespace: str = NAMESPACE) -> None:
    stats = _index.describe_index_stats()
    if namespace in stats.get("namespaces", {}):
        _index.delete(delete_all=True, namespace=namespace)


def query(embedding: list[float], top_k: int = 5, filter: dict | None = None, namespace: str = NAMESPACE) -> list[dict]:
    result = _index.query(
        vector=embedding,
        top_k=top_k,
        filter=filter,
        namespace=namespace,
        include_metadata=True,
    )
    return result["matches"]


def rerank(query_text: str, matches: list[dict], top_n: int) -> list[dict]:
    if not matches:
        return []
    documents = [{"id": m["id"], "text": m["metadata"].get("text", "")} for m in matches]
    result = _pc.inference.rerank(
        model=RERANK_MODEL,
        query=query_text,
        documents=documents,
        rank_fields=["text"],
        top_n=top_n,
        return_documents=False,
    )
    reranked = []
    for item in result.data:
        m = matches[item.index]
        reranked.append(
            {"id": m["id"], "score": m["score"], "metadata": m["metadata"], "rerank_score": item.score}
        )
    return reranked


def _fuse(dense_matches: list[dict], sparse_matches: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: combine two ranked lists by rank position, not raw score, so
    cosine-similarity scores and BM25 scores (different scales entirely) never need to be
    compared directly."""
    fused: dict[str, dict] = {}
    for rank, m in enumerate(dense_matches):
        entry = fused.setdefault(m["id"], {"id": m["id"], "metadata": m["metadata"], "score": 0.0})
        entry["score"] += 1.0 / (RRF_K + rank + 1)
    for rank, m in enumerate(sparse_matches):
        entry = fused.setdefault(m["id"], {"id": m["id"], "metadata": m["metadata"], "score": 0.0})
        entry["score"] += 1.0 / (RRF_K + rank + 1)
    return sorted(fused.values(), key=lambda m: m["score"], reverse=True)


def retrieve(
    query_text: str,
    embedding: list[float],
    top_k: int = 10,
    top_n: int = 4,
    filter: dict | None = None,
    namespace: str = NAMESPACE,
) -> list[dict]:
    """Hybrid retrieval: dense (Pinecone embedding search) + sparse (BM25 over the same corpus),
    fused by Reciprocal Rank Fusion, then reranked. Dense catches semantic/paraphrased matches;
    sparse catches exact terms (visa category codes, country names, dollar figures) that
    embedding similarity alone can miss."""
    dense_matches = query(embedding, top_k=top_k, filter=filter, namespace=namespace)
    sparse_matches = sparse_search(query_text, namespace=namespace, filter=filter, top_k=top_k)
    fused_matches = _fuse(dense_matches, sparse_matches)[:top_k]
    return rerank(query_text, fused_matches, top_n=top_n)
