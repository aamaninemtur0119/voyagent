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
    }
