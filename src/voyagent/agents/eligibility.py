"""Eligibility Agent — visa/entry-requirements research. Retrieve -> verified-evidence gate ->
generate, carried over from the Week 2 project (crosscheck) including its hard-won fixes: nationality
normalization (corpus is tagged by country noun, not demonym), a deterministic confidence gate on
the top rerank score (skips generation entirely on genuinely weak retrieval, rather than hoping the
LLM notices), and an explicit topic-vs-purpose mismatch check (catches a document that shares
surface keywords with an off-topic question, which a score threshold alone can miss).

This module is deliberately callable standalone (no LangGraph import here) so it can be unit-tested
in isolation from the graph that wraps it.
"""

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401 - re-exported for parity/tests
from pydantic import BaseModel, Field

from voyagent.config import settings
from voyagent.retrieval.embeddings import embed_texts
from voyagent.retrieval.vector_store import retrieve as vector_retrieve
from voyagent.tools.you_search import search as you_search

_llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)

ANSWER_TYPES = Literal[
    "visa_required",
    "visa_free_waiver",
    "visa_free_waiver_with_ETA",
    "visa_free_waiver_with_ETIAS",
    "different_visa_category_required",
    "refuse_and_verify",
]

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35


class NormalizedNationality(BaseModel):
    nationality: str = Field(
        description=(
            "The country NAME this nationality/citizenship refers to (e.g. 'China', 'India', 'Canada'), "
            "never the demonym/adjective form (not 'Chinese', 'Indian', 'Canadian'). The corpus is tagged "
            "by country name. If already a country name, return it unchanged."
        ),
    )


class Answer(BaseModel):
    answer_type: ANSWER_TYPES
    answer_text: str = Field(
        default="",
        description=(
            "The full answer, in plain language. REQUIRED even when answer_type is "
            "'refuse_and_verify' — explain what's uncertain and why. Never leave this empty."
        ),
    )
    requirements: str = Field(default="", description="Just the visa/entry requirement itself.")
    processing_timeline: str = Field(default="", description="Processing time, if stated in the sources.")
    estimated_cost: str = Field(default="", description="Fee amount with currency, if stated in the sources.")
    citation: str = Field(default="", description="Which retrieved source(s) this relies on, by id/source_name.")


def normalize_nationality(raw: str) -> str:
    parsed = _llm.with_structured_output(NormalizedNationality).invoke(f"Normalize this to a country name: {raw}")
    return parsed.nationality


def retrieve(situation: dict, user_question: str) -> list[dict]:
    query_text = (
        f"{situation.get('nationality', '')} citizen traveling to {situation.get('destination', '')} "
        f"for {situation.get('purpose', '')}, staying {situation.get('duration', '')}. "
        f"Specific question: {user_question}"
    )
    embedding = embed_texts([query_text])[0]

    filter_: dict = {"destination_country": situation["destination"]}
    nationality = situation.get("nationality", "")
    if nationality:
        filter_["applicable_nationalities"] = {"$in": [nationality, "all"]}

    matches = vector_retrieve(query_text, embedding, top_k=10, top_n=4, filter=filter_)
    return [
        {
            "id": m["id"],
            "text": m["metadata"].get("text", ""),
            "source_url": m["metadata"].get("source_url", ""),
            "source_name": m["metadata"].get("source_name", ""),
            "confidence": m["metadata"].get("confidence", "uncertain"),
            "topic": m["metadata"].get("topic", ""),
            "rerank_score": m["rerank_score"],
        }
        for m in matches
    ]


def has_sufficient_evidence(retrieved_chunks: list[dict]) -> bool:
    return bool(retrieved_chunks) and retrieved_chunks[0]["rerank_score"] >= RETRIEVAL_CONFIDENCE_THRESHOLD


def _no_evidence_answer(retrieved_chunks: list[dict]) -> Answer:
    top_score = retrieved_chunks[0]["rerank_score"] if retrieved_chunks else 0.0
    return Answer(
        answer_type="refuse_and_verify",
        answer_text=(
            "No source in the corpus is a strong enough match for this specific situation to answer "
            "confidently — this is a deterministic retrieval-confidence check, not a guess. Please "
            "verify directly with the relevant government source."
        ),
        citation=f"(no source cleared the retrieval-confidence threshold — top match scored {top_score:.2f})",
    )


def generate_answer(situation: dict, user_question: str, retrieved_chunks: list[dict]) -> Answer:
    context = "\n\n---\n\n".join(
        f"[{c['id']}] (topic: {c['topic'] or 'unspecified'}, confidence: {c['confidence']}, source: {c['source_name']})\n{c['text']}"
        for c in retrieved_chunks
    ) or "No relevant sources were retrieved."

    prompt = (
        "You are a travel entry-requirements assistant. Answer the traveler's actual question using "
        "ONLY the retrieved sources below. Do not use outside knowledge. If the specific question "
        "isn't addressed, use answer_type 'refuse_and_verify' rather than answering a different, "
        "easier question. Check each source's 'topic' field against the traveler's actual purpose: a "
        "source sharing the same country/nationality but a different topic is NOT relevant evidence, "
        "even if it's the most similar-looking source retrieved.\n\n"
        f"Traveler's actual question: {user_question}\n\nTraveler situation: {situation}\n\n"
        f"Retrieved sources:\n{context}"
    )
    answer = _llm.with_structured_output(Answer).invoke(prompt)
    if not answer.answer_text.strip():
        answer.answer_text = answer.requirements or f"({answer.answer_type}, no explanation provided.)"
    return answer


class RecencyCheck(BaseModel):
    contradiction_found: bool = Field(
        description=(
            "True ONLY if a search result from a clearly official source (a government domain, "
            "an embassy/consulate site, an official immigration authority — NOT a travel blog, "
            "aggregator, or unofficial 'the real answer' style site) explicitly and unambiguously "
            "states something that contradicts the corpus-grounded answer. A vague signal, an "
            "unofficial source disagreeing, or an ambiguous/older-looking result is NOT a "
            "contradiction — default to False when in doubt. The corpus was itself sourced from "
            "official government pages, so it should only be overridden by evidence at least as "
            "credible."
        )
    )
    contradicting_url: str = Field(default="", description="The specific URL that contradicts, if any.")
    note: str = Field(default="", description="One sentence explaining the finding either way.")


def _recency_check(answer: Answer, situation: dict) -> RecencyCheck:
    """Best-effort cross-check against live search — closes part of the corpus's known freshness
    gap (no automated re-fetch mechanism). Deliberately conservative: the corpus was built from
    official sources, so only an equally official-looking live source can override it. Raises on
    a genuine infrastructure failure (network, missing key) — the caller decides what to do."""
    query = f"{situation['destination']} visa requirements {situation['nationality']} citizens recent changes"
    results = you_search(query, max_results=5)
    if not results:
        return RecencyCheck(contradiction_found=False, note="No search results returned.")

    listing = "\n\n".join(
        f"[{r['url']}] {r['title']}\n" + " / ".join(r["snippets"][:3]) for r in results
    )
    prompt = (
        "A visa-requirements answer was generated from a corpus of official government sources. "
        "Check whether any of these LIVE web search results, from a source you'd judge at least "
        "as official/credible as a government immigration page, clearly and explicitly "
        "contradicts it. Be conservative — an unofficial or vague source does not count.\n\n"
        f"Corpus-grounded answer: {answer.answer_text}\n\nRequirements stated: {answer.requirements}\n\n"
        f"Live search results:\n{listing}"
    )
    return _llm.with_structured_output(RecencyCheck).invoke(prompt)


def run(nationality: str, destination: str, purpose: str, duration: str) -> dict:
    """The Eligibility Agent's entry point — retrieve, gate, generate. Raises on a genuine
    infrastructure failure (embedding/Pinecone/LLM call error) rather than swallowing it, so the
    orchestrator graph's retry policy can see and handle the failure explicitly."""
    nationality = normalize_nationality(nationality)
    situation = {"nationality": nationality, "destination": destination, "purpose": purpose, "duration": duration}
    question = (
        f"What are the entry/visa requirements for a {nationality} citizen traveling to "
        f"{destination} for {purpose}, staying {duration}?"
    )

    retrieved_chunks = retrieve(situation, question)
    answer = (
        _no_evidence_answer(retrieved_chunks)
        if not has_sufficient_evidence(retrieved_chunks)
        else generate_answer(situation, question, retrieved_chunks)
    )

    # Recency check: best-effort, never blocks the core answer. Only worth running on an answer
    # confident enough that a contradiction would actually matter — a refusal is already
    # hedged. Retry once, then skip silently on failure (missing key, network error) rather than
    # treating a supplementary safety net as a hard dependency.
    recency_status = "skipped"
    if answer.answer_type != "refuse_and_verify":
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                check = _recency_check(answer, situation)
                recency_status = "contradiction_found" if check.contradiction_found else "ok"
                if check.contradiction_found:
                    answer.answer_type = "refuse_and_verify"
                    answer.answer_text = (
                        f"{answer.answer_text}\n\n⚠️ A live source ({check.contradicting_url}) appears "
                        f"to contradict this corpus-grounded answer: {check.note} Treating this as "
                        "unconfirmed rather than asserting either version with confidence."
                    )
                    answer.citation = f"{answer.citation}; contradicted by {check.contradicting_url}"
                last_exc = None
                break
            except Exception as e:  # noqa: BLE001 - best-effort safety net, never a hard dependency
                last_exc = e
        if last_exc is not None:
            recency_status = "not_configured" if "YOU_API_KEY" in str(last_exc) else f"failed: {last_exc}"

    return {
        "situation": situation,
        "answer_type": answer.answer_type,
        "answer_text": answer.answer_text,
        "requirements": answer.requirements,
        "processing_timeline": answer.processing_timeline,
        "estimated_cost": answer.estimated_cost,
        "citation": answer.citation,
        "retrieved_chunk_ids": [c["id"] for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
        "recency_check_status": recency_status,
    }
