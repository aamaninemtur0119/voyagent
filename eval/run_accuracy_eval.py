"""Accuracy / RAG evaluation for the Eligibility Agent, against `eval/golden_set.csv` (25 labeled
visa-requirements questions). This is the Week 2 harness, brought into the Week 3 repo and adapted
to Voyagent's eligibility pipeline — which now also reconciles the corpus answer against live
You.com search, so a fourth axis tracks what that reconciliation did.

Four independent axes, deliberately kept separate rather than collapsed into one score:
  - answer_type accuracy: did the pipeline reach the right conclusion?
  - retrieval hit-rate: did it actually retrieve the document that controls the answer?
  - faithfulness (LLM-judged): is the CORPUS answer (pre-reconciliation) grounded only in what was
    retrieved, with no invented claims? Judged on the corpus answer specifically, because the final
    answer may also draw on live search — grounding that against corpus chunks alone would flag it
    unfaithful by construction. Independent of whether retrieval found the *right* doc.
  - reconciliation effect: the corpus-only answer vs. the final answer — did the live cross-check
    leave it alone, fix a stale corpus answer, or break a correct one?

Run:  uv run python -m eval.run_accuracy_eval
Writes eval/accuracy_results.csv (per-row) and eval/accuracy_report.md (aggregate + failures).
This is separate from eval/run_eval.py, which tests agentic behavior (control flow, HITL, ...).
"""

import csv
from pathlib import Path

from pydantic import BaseModel, Field

from voyagent.agents.eligibility import run as eligibility_run
from voyagent.llm import structured

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.csv"

# Row id -> acceptable source-file stems (without .md). Empty list = no single controlling source
# is expected (ask_back / genuine-refusal rows) — excluded from the retrieval hit-rate.
EXPECTED_SOURCES: dict[str, list[str]] = {
    "1": ["japan-visa-china"],
    "2": ["japan-visa-india"],
    "3": ["japan-visa-exemption-list"],
    "4": ["japan-visa-exemption-list"],
    "5": ["us-b1b2-visa", "us-vwp-full-list"],
    "6": ["us-b1b2-visa", "us-vwp-full-list"],
    "7": ["us-b1b2-visa", "us-vwp-full-list"],
    "8": ["us-entry-canada"],
    "9": ["uk-standard-visitor-visa"],
    "10": ["uk-standard-visitor-visa"],
    "11": ["uk-eta-national-list"],
    "12": ["uk-eta-status"],
    "13": ["schengen-visa-required-full-list", "schengen-visa-requirement-list"],
    "14": ["schengen-visa-required-full-list", "schengen-visa-requirement-list"],
    "15": ["etias-status"],
    "16": ["etias-status"],
    "17": ["australia-visitor-visa-600"],
    "18": ["australia-visitor-visa-600"],
    "19": ["australia-visitor-visa-600"],
    "20": ["australia-eta-601"],
    "21": ["us-b1b2-visa"],
    "22": ["uk-student-visa"],
    "23": [],  # ask_back: no single Schengen-wide source should exist
    "24": ["schengen-90-180-rule"],
    "25": [],  # genuine refusal: not addressed by any specific doc
}

VALID_ANSWER_TYPES = {
    "visa_required", "visa_free_waiver", "visa_free_waiver_with_ETA",
    "visa_free_waiver_with_ETIAS", "different_visa_category_required", "refuse_and_verify",
}


class FaithfulnessJudgment(BaseModel):
    faithful: bool = Field(
        description="True only if every factual claim in the answer is directly supported by the "
        "provided sources, with no invented or assumed details."
    )
    unsupported_claims: str = Field(
        default="", description="If not faithful, which specific claims aren't supported. Empty if faithful."
    )


def judge_faithfulness(answer_text: str, requirements: str, retrieved_chunks: list[dict]):
    """Returns a FaithfulnessJudgment, or None if the judge call itself failed (excluded from the rate)."""
    if not retrieved_chunks:
        return FaithfulnessJudgment(
            faithful=not answer_text.strip(), unsupported_claims="No sources were retrieved at all."
        )
    sources_text = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in retrieved_chunks)
    prompt = (
        "You are auditing a RAG system for hallucination. Below is an answer it gave and the "
        "sources it had available. Judge ONLY whether every factual claim in the answer is actually "
        "supported by the sources — not whether the answer is good or complete, and not whether the "
        "sources are correct or current. If the answer states something the sources don't say (even "
        "if true in the real world), that's unfaithful.\n\n"
        f"Answer text: {answer_text}\n\nRequirements stated: {requirements}\n\n"
        f"Sources available:\n{sources_text}"
    )
    return structured(FaithfulnessJudgment, prompt, default=None)


def _reconciliation_effect(corpus_type: str, final_type: str, expected_type: str) -> str:
    if corpus_type == final_type:
        return "unchanged"
    if not (expected_type in VALID_ANSWER_TYPES):
        return "changed (expected type not representable)"
    corpus_ok, final_ok = corpus_type == expected_type, final_type == expected_type
    if final_ok and not corpus_ok:
        return "FIXED a stale corpus answer"
    if corpus_ok and not final_ok:
        return "BROKE a correct corpus answer"
    return f"changed {corpus_type} -> {final_type} (both wrong)"


def run() -> None:
    with open(GOLDEN_SET_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(rows, 1):
        rid = row["id"]
        print(f"[{i}/{len(rows)}] row {rid}: {row['nationality']} -> {row['destination']} ({row['purpose']})", flush=True)
        result = eligibility_run(row["nationality"], row["destination"], row["purpose"], row["duration"])

        expected_type = row["expected_answer_type"]
        actual_type = result["answer_type"]
        corpus_type = result.get("corpus_answer_type", actual_type)
        schema_supports_expected = expected_type in VALID_ANSWER_TYPES
        type_match = schema_supports_expected and actual_type == expected_type

        expected_sources = EXPECTED_SOURCES.get(rid, [])
        retrieved_ids = [c["id"] for c in result["retrieved_chunks"]]
        retrieval_applicable = bool(expected_sources)
        retrieval_hit = (
            any(cid.startswith(stem) for cid in retrieved_ids for stem in expected_sources)
            if retrieval_applicable else None
        )

        # faithfulness is judged on the CORPUS answer AND its own requirements vs. the retrieved
        # chunks — the reconciled final answer also draws on live search, which isn't in the chunks.
        judgment = judge_faithfulness(
            result.get("corpus_answer_text") or result["answer_text"],
            result.get("corpus_requirements") or result["requirements"],
            result["retrieved_chunks"],
        )

        results.append({
            "id": rid,
            "query": row["query"],
            "category": row["category"],
            "nationality": row["nationality"],
            "destination": row["destination"],
            "purpose": row["purpose"],
            "duration": row["duration"],
            "expected_answer_type": expected_type,
            "actual_answer_type": actual_type,
            "corpus_answer_type": corpus_type,
            "corpus_answer_text": result.get("corpus_answer_text", ""),
            "schema_supports_expected": schema_supports_expected,
            "type_match": type_match,
            "retrieval_applicable": retrieval_applicable,
            "retrieval_hit": retrieval_hit,
            "retrieved_ids": "; ".join(retrieved_ids),
            "faithful": None if judgment is None else judgment.faithful,
            "unsupported_claims": "" if judgment is None else judgment.unsupported_claims,
            "primary_source": result.get("primary_source", ""),
            "cross_check_status": result.get("cross_check_status", ""),
            "divergence_note": result.get("divergence_note", ""),
            "reconciliation_effect": _reconciliation_effect(corpus_type, actual_type, expected_type),
            "answer_text": result["answer_text"],
            "notes": row["notes"],
        })
        r = results[-1]
        print(f"    type {'✓' if r['type_match'] else '✗'} (exp {expected_type} / got {actual_type})"
              f"  retrieval {'✓' if r['retrieval_hit'] else ('–' if r['retrieval_hit'] is None else '✗')}"
              f"  faithful {'✓' if r['faithful'] else ('?' if r['faithful'] is None else '✗')}"
              f"  recon: {r['reconciliation_effect']}", flush=True)
        # persist after every row so a slow / interrupted run still yields data
        _write_csv(results)
        _write_report(results)

    _write_csv(results)
    _write_report(results)
    print(f"\nDone. See {EVAL_DIR / 'accuracy_results.csv'} and {EVAL_DIR / 'accuracy_report.md'}", flush=True)


def _write_csv(results: list[dict]) -> None:
    with open(EVAL_DIR / "accuracy_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def _pct(num: int, den: int) -> str:
    return f"{num / den:.0%}" if den else "n/a"


def _write_report(results: list[dict]) -> None:
    n = len(results)
    schema_rows = [r for r in results if r["schema_supports_expected"]]
    type_hits = sum(r["type_match"] for r in schema_rows)
    retr_rows = [r for r in results if r["retrieval_applicable"]]
    retr_hits = sum(bool(r["retrieval_hit"]) for r in retr_rows)
    faith_rows = [r for r in results if r["faithful"] is not None]
    faith_hits = sum(r["faithful"] for r in faith_rows)
    unsupported_type_rows = [r for r in results if not r["schema_supports_expected"]]
    recon = {}
    for r in results:
        recon[r["reconciliation_effect"]] = recon.get(r["reconciliation_effect"], 0) + 1

    lines = [
        "# Eligibility Agent — Accuracy Eval\n",
        f"Golden set: {n} questions (`eval/golden_set.csv`). Per-row output: `eval/accuracy_results.csv`.\n",
        "## Summary\n",
        f"- **Answer-type accuracy**: {_pct(type_hits, len(schema_rows))} ({type_hits}/{len(schema_rows)} rows the schema can represent)",
        f"- **Retrieval hit-rate**: {_pct(retr_hits, len(retr_rows))} ({retr_hits}/{len(retr_rows)} rows with a known controlling source; {n - len(retr_rows)} excluded)",
        f"- **Faithfulness**: {_pct(faith_hits, len(faith_rows))} ({faith_hits}/{len(faith_rows)} judged; LLM-judged vs. only what was retrieved)",
        "",
        "## Live-reconciliation effect (corpus answer vs. final answer)\n",
    ]
    for effect, count in sorted(recon.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {effect}: {count}")
    lines.append("")

    lines.append("## By category\n")
    lines.append("| Category | n | Type acc | Retrieval | Faithful |")
    lines.append("|---|---|---|---|---|")
    for cat in sorted({r["category"] for r in results}):
        cr = [r for r in results if r["category"] == cat]
        cs = [r for r in cr if r["schema_supports_expected"]]
        crr = [r for r in cr if r["retrieval_applicable"]]
        cf = [r for r in cr if r["faithful"] is not None]
        lines.append(
            f"| {cat} | {len(cr)} | {_pct(sum(r['type_match'] for r in cs), len(cs))} "
            f"| {_pct(sum(bool(r['retrieval_hit']) for r in crr), len(crr))} "
            f"| {_pct(sum(r['faithful'] for r in cf), len(cf))} |"
        )
    lines.append("")

    if unsupported_type_rows:
        lines.append("## Schema gap\n")
        lines.append(
            f"{len(unsupported_type_rows)} golden-set row(s) expect an `answer_type` Voyagent's `ANSWER_TYPES` "
            "has no value for (e.g. `ask_back`). Not a retrieval/generation failure — a deliberate schema "
            "limitation (the pipeline can't ask a clarifying question before committing to an answer_type).\n"
        )
        for r in unsupported_type_rows:
            lines.append(f"- **Row {r['id']}** expects `{r['expected_answer_type']}`, returned `{r['actual_answer_type']}` — {r['notes']}")
        lines.append("")

    failures = [
        r for r in results
        if (r["schema_supports_expected"] and not r["type_match"])
        or r["retrieval_hit"] is False
        or r["faithful"] is False
        or r["reconciliation_effect"].startswith("BROKE")
    ]
    lines.append("## Failure analysis\n")
    if not failures:
        lines.append("No failures beyond any schema gap noted above.\n")
    for r in failures:
        lines.append(f"### Row {r['id']}: {r['nationality']} -> {r['destination']} ({r['purpose']}, {r['duration']})\n")
        lines.append(f"- Query: \"{r['query']}\"")
        lines.append(f"- Expected `{r['expected_answer_type']}` — corpus said `{r['corpus_answer_type']}` — final `{r['actual_answer_type']}`" + (" ✅" if r["type_match"] else " ❌ MISMATCH"))
        lines.append(f"- Reconciliation: {r['reconciliation_effect']}" + (f" (primary_source={r['primary_source']}, {r['cross_check_status']})" if r["primary_source"] else ""))
        if r["divergence_note"]:
            lines.append(f"  - divergence note: {r['divergence_note']}")
        if r["retrieval_applicable"]:
            lines.append(f"- Retrieval: {'✅ hit' if r['retrieval_hit'] else '❌ MISS'} — retrieved {r['retrieved_ids'] or '(nothing)'}")
        if r["faithful"] is False:
            lines.append(f"- Faithfulness: ❌ UNFAITHFUL — {r['unsupported_claims']}")
        lines.append(f"- Golden-set note: {r['notes']}")
        lines.append(f"- Answer: {r['answer_text'][:400]}{'...' if len(r['answer_text']) > 400 else ''}")
        lines.append("")

    (EVAL_DIR / "accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
