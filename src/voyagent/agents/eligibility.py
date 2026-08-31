"""Eligibility Agent — visa/entry-requirements research. Retrieve -> verified-evidence gate ->
generate -> reconcile against live search, carried over from the Week 2 project (crosscheck)
including its hard-won fixes: nationality normalization (corpus is tagged by country noun, not
demonym), a deterministic confidence gate on the top rerank score (skips generation entirely on
genuinely weak retrieval, rather than hoping the LLM notices), and an explicit topic-vs-purpose
mismatch check (catches a document that shares surface keywords with an off-topic question, which a
score threshold alone can miss).

The corpus was built from official government pages but has NO automated freshness mechanism, so
after the corpus answer is generated it is reconciled against live You.com results: when a live
result from a clearly official source (a government domain / immigration authority / embassy)
directly addresses this traveler's nationality + destination + purpose, it is treated as
authoritative (it's more current); otherwise the corpus answer stands and live results only fill
gaps. Every run returns a `sources` list tagging each source as `corpus` or `live`, plus which one
drove the final answer.

This module is deliberately callable standalone (no LangGraph import here) so it can be unit-tested
in isolation from the graph that wraps it.
"""

from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401 - re-exported for parity/tests
from pydantic import BaseModel, Field

from voyagent.llm import structured
from voyagent.retrieval.embeddings import embed_texts
from voyagent.retrieval.vector_store import retrieve as vector_retrieve
from voyagent.tools.you_search import search as you_search

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
    # default to the raw input: an unrecognized/misspelled nationality then simply retrieves
    # nothing and falls through to refuse_and_verify, rather than the call failing.
    parsed = structured(
        NormalizedNationality,
        f"Normalize this to a country name: {raw}",
        default=NormalizedNationality(nationality=raw),
    )
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
        "GROUNDING RULES — do NOT state any of the following unless a retrieved source contains that "
        "exact fact: whether a country is or isn't on a visa-waiver / VWP / ETA / eVisitor list; a "
        "specific fee amount or currency; the name or date of a law, regulation, or presidential "
        "proclamation; where to apply (consulate, embassy, portal, 'main destination country'); a "
        "processing time; a passport-validity window; or a maximum stay length. If such a detail "
        "would help but isn't in the sources, write 'not specified in the available sources' — never "
        "supply it from general knowledge. Keep answer_text short.\n\n"
        f"Traveler's actual question: {user_question}\n\nTraveler situation: {situation}\n\n"
        f"Retrieved sources:\n{context}"
    )
    answer = structured(
        Answer,
        prompt,
        default=Answer(
            answer_type="refuse_and_verify",
            answer_text="The corpus answer couldn't be generated this time — please verify directly with the relevant government source.",
            citation="(generation failed)",
        ),
    )
    if not answer.answer_text.strip():
        answer.answer_text = answer.requirements or f"({answer.answer_type}, no explanation provided.)"
    return answer


class GroundedRewrite(BaseModel):
    answer_text: str = Field(
        description="The answer rewritten so every factual claim is directly supported by a source. "
        "Anything not in the sources is removed or replaced with '(not specified in the available "
        "sources)'. Keep the overall conclusion and everything that IS supported. Keep it short."
    )
    requirements: str = Field(default="", description="ONLY what a source directly states about the requirement.")
    processing_timeline: str = Field(default="", description="ONLY if a source states it, else empty.")
    estimated_cost: str = Field(default="", description="ONLY if a source states it, else empty.")
    removed_claims: list[str] = Field(
        default_factory=list, description="The specific claims dropped or softened because no source supports them."
    )


def _enforce_grounding(answer: Answer, retrieved_chunks: list[dict]) -> Answer:
    """In-pipeline faithfulness gate: after generation, strip any claim the retrieved sources don't
    actually support, so the corpus answer can't quietly import real-world knowledge (VWP lists,
    fee amounts, proclamation dates, where-to-apply). A refusal has nothing to ground; skip it."""
    if answer.answer_type == "refuse_and_verify" or not retrieved_chunks:
        return answer
    context = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in retrieved_chunks)
    rewrite = structured(
        GroundedRewrite,
        "Below is a visa/entry-requirements answer and the ONLY sources it may rely on. Rewrite it so "
        "every factual claim is directly supported by a source. Remove or replace with '(not "
        "specified in the available sources)' anything the sources don't state — visa-waiver/VWP/ETA "
        "membership, fee amounts, law or proclamation names and dates, where to apply, processing "
        "times, passport-validity windows, maximum stay lengths. Keep the overall conclusion and "
        "everything that IS supported.\n\n"
        f"Answer text: {answer.answer_text}\nRequirements: {answer.requirements}\n"
        f"Processing timeline: {answer.processing_timeline}\nEstimated cost: {answer.estimated_cost}\n\n"
        f"Sources:\n{context}",
        default=None,
    )
    if rewrite is None:
        return answer
    if rewrite.answer_text.strip():
        answer.answer_text = rewrite.answer_text.strip()
    answer.requirements = rewrite.requirements.strip()
    answer.processing_timeline = rewrite.processing_timeline.strip()
    answer.estimated_cost = rewrite.estimated_cost.strip()
    return answer


class Reconciliation(BaseModel):
    """The best single answer after weighing the corpus answer against live search results."""

    answer_type: ANSWER_TYPES
    answer_text: str = Field(
        default="",
        description=(
            "The best plain-language answer for this traveler. If corpus and live sources "
            "materially disagreed, the text itself should say so briefly and which one it follows."
        ),
    )
    requirements: str = Field(default="", description="Just the visa/entry requirement itself.")
    processing_timeline: str = Field(default="", description="Processing time, if stated in either source.")
    estimated_cost: str = Field(default="", description="Fee amount with currency, if stated in either source.")
    primary_source: Literal["corpus", "live", "both"] = Field(
        default="corpus",
        description=(
            "'live' if an OFFICIAL live result (government domain / immigration authority / embassy — "
            "not a blog, aggregator, or visa-reseller) drove the answer; 'corpus' if the corpus did; "
            "'both' if an official live source and the corpus agreed."
        ),
    )
    divergence_note: str = Field(
        default="",
        description=(
            "Non-empty ONLY when the corpus and an official live source MATERIALLY DISAGREED (a "
            "different visa type, a newly-required authorization, a removed exemption): one or two "
            "sentences naming what each said and which was used and why. Leave EMPTY when they "
            "agree, even if one is more detailed than the other."
        ),
    )
    authoritative_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs of the OFFICIAL live results that actually informed this answer (a government / "
            "immigration-authority / embassy page). Empty if the corpus alone drove it. Never "
            "include blogs, aggregators, or visa-reseller sites here."
        ),
    )


def _reconcile(corpus_answer: Answer, live_results: list[dict], situation: dict) -> Reconciliation:
    """Weigh the corpus answer against live search. Rule: an OFFICIAL live source addressing this
    exact nationality+destination+purpose is authoritative (more current than the corpus); the
    corpus otherwise stands and live only fills gaps. Falls back to the corpus answer unchanged
    if there are no live results or the reconciliation call fails."""
    fallback = Reconciliation(
        answer_type=corpus_answer.answer_type,
        answer_text=corpus_answer.answer_text,
        requirements=corpus_answer.requirements,
        processing_timeline=corpus_answer.processing_timeline,
        estimated_cost=corpus_answer.estimated_cost,
        primary_source="corpus",
    )
    if not live_results:
        return fallback

    listing = "\n\n".join(
        f"[{r['url']}] {r['title']}\n" + " / ".join(r.get("snippets", [])[:3]) for r in live_results
    )
    prompt = (
        "Reconcile two sources of visa/entry-requirements information for ONE traveler and produce "
        "the single best answer.\n\n"
        "CORPUS ANSWER — built from curated official government pages, but with NO automated "
        "freshness mechanism, so it can be out of date:\n"
        f"  answer_type: {corpus_answer.answer_type}\n"
        f"  answer_text: {corpus_answer.answer_text}\n"
        f"  requirements: {corpus_answer.requirements}\n"
        f"  processing_timeline: {corpus_answer.processing_timeline}\n"
        f"  estimated_cost: {corpus_answer.estimated_cost}\n\n"
        f"LIVE WEB SEARCH RESULTS — current, but mixed quality:\n{listing}\n\n"
        "RULE: if a LIVE result from a clearly OFFICIAL source (a government domain, an official "
        "immigration authority, an embassy/consulate — NOT a travel blog, aggregator, visa-service "
        "reseller, or SEO 'the real answer' site) directly addresses THIS traveler's nationality + "
        "destination + purpose, treat it as authoritative and set primary_source='live' (it is more "
        "current than the corpus). Otherwise the corpus answer stands (primary_source='corpus') and "
        "live results only fill gaps the corpus doesn't cover. If corpus and an official live source "
        "agree, set primary_source='both'. If neither the corpus nor any official live source "
        "actually addresses this specific case, use answer_type 'refuse_and_verify'.\n"
        "List in authoritative_urls only the OFFICIAL live URLs you actually relied on. Set "
        "divergence_note ONLY on a material disagreement — leave it empty when they agree.\n"
        "GROUNDING: state only facts that appear in the corpus answer above or in the live snippets "
        "of a URL you list in authoritative_urls. Do NOT add outside knowledge the neither source "
        "contains — fee amounts, law/proclamation names or dates, where to apply, processing times.\n\n"
        f"Traveler situation: {situation}"
    )
    return structured(Reconciliation, prompt, default=fallback)


def _collect_sources(retrieved_chunks: list[dict], live_results: list[dict], authoritative_urls: list[str]) -> list[dict]:
    """Tagged, deduped list of the sources that actually informed the answer: every retrieved
    corpus chunk's source, plus the live results the reconciliation flagged as official/relied-on
    (SEO and aggregator hits that a live search inevitably also returns are left out)."""
    authoritative = {u for u in (authoritative_urls or []) if u}
    seen: set = set()
    sources: list[dict] = []
    for c in retrieved_chunks:
        name = c.get("source_name") or c.get("id") or ""
        url = c.get("source_url", "") or ""
        if name and ("corpus", name, url) not in seen:
            seen.add(("corpus", name, url))
            sources.append({"type": "corpus", "name": name, "url": url})
    for r in live_results:
        url = r.get("url", "") or ""
        if url in authoritative and ("live", url) not in seen:
            seen.add(("live", url))
            sources.append({"type": "live", "name": r.get("title") or url, "url": url})
    return sources


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
    if not has_sufficient_evidence(retrieved_chunks):
        corpus_answer = _no_evidence_answer(retrieved_chunks)
    else:
        # generate, then run the grounding gate so the corpus answer can't import unsourced facts
        corpus_answer = _enforce_grounding(
            generate_answer(situation, question, retrieved_chunks), retrieved_chunks
        )

    # Reconcile against live search — best-effort, never a hard dependency. Runs even when the
    # corpus answer is a refusal (a live official source may still have the answer). Retry once,
    # then fall back to the corpus answer if search is unconfigured / erroring.
    live_results: list[dict] = []
    final = corpus_answer
    primary_source = "corpus"
    divergence_note = ""
    authoritative_urls: list[str] = []
    cross_status = "skipped"
    last_exc: Exception | None = None
    for _ in range(2):
        try:
            live_results = you_search(
                f"{situation['destination']} visa requirements for {situation['nationality']} "
                f"citizens {situation['purpose']} 2026",
                max_results=5,
            )
            rec = _reconcile(corpus_answer, live_results, situation)
            final = rec
            primary_source = rec.primary_source
            authoritative_urls = rec.authoritative_urls
            # A divergence note only counts when the sources actually disagreed — 'both' means they
            # agreed, so drop any commentary the model put there.
            divergence_note = "" if primary_source == "both" else rec.divergence_note.strip()
            cross_status = "diverged" if divergence_note else "reconciled"
            last_exc = None
            break
        except Exception as e:  # noqa: BLE001 - supplementary safety net, never a hard dependency
            last_exc = e
    if last_exc is not None:
        cross_status = "not_configured" if "YOU_API_KEY" in str(last_exc) else f"failed: {last_exc}"

    sources = _collect_sources(retrieved_chunks, live_results, authoritative_urls)

    return {
        "situation": situation,
        "answer_type": final.answer_type,
        "answer_text": final.answer_text or corpus_answer.answer_text,
        "requirements": final.requirements or corpus_answer.requirements,
        "processing_timeline": final.processing_timeline or corpus_answer.processing_timeline,
        "estimated_cost": final.estimated_cost or corpus_answer.estimated_cost,
        "citation": corpus_answer.citation,
        # the corpus-only answer, before live reconciliation — lets the accuracy eval see when
        # (and which way) the live cross-check changed the conclusion, and judge faithfulness
        # against the right text (corpus answer vs. corpus chunks, not the reconciled answer)
        "corpus_answer_type": corpus_answer.answer_type,
        "corpus_answer_text": corpus_answer.answer_text,
        "corpus_requirements": corpus_answer.requirements,
        "primary_source": primary_source,
        "divergence_note": divergence_note,
        "sources": sources,
        "retrieved_chunk_ids": [c["id"] for c in retrieved_chunks],
        "retrieved_chunks": retrieved_chunks,
        "cross_check_status": cross_status,
    }
