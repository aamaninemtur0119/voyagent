# Evaluations

Two suites, measuring different things. Both make real LLM + API calls (no mocked reasoning,
except where a failure is deliberately simulated).

## 1. Agentic behavior — `run_eval.py`

```bash
uv run python -m eval.run_eval
```

Tests the properties specific to being an agentic system, as code:

- **Core** — control flow (conditional routing, parallel fan-out/fan-in, the revise cycle,
  MAX_REPLANS termination), state (delta updates, deadline recompute on a date revise), tool
  failure recovery (an agent fails → graph continues; a malformed structured-output response
  degrades to a default), and the two human-in-the-loop write gates (calendar, email) — each
  blocks, each is independent, Cancel ends the run.
- **Robustness** — breaking-case inputs (typos, unknown nationality, city/country mismatch,
  gibberish) where the bar is *graceful behavior*, not an accurate answer; plus the Eligibility
  Agent's corpus-vs-live reconciliation and source tagging.

Writes `eval_report.md`.

## 2. Eligibility accuracy — `run_accuracy_eval.py`

```bash
uv run python -m eval.run_accuracy_eval
```

The Week 2 RAG harness, brought into this repo and pointed at Voyagent's `eligibility.run`, against
`golden_set.csv` (25 labeled visa-requirements questions). Four axes, kept separate:

- **answer_type accuracy** — did it reach the right conclusion?
- **retrieval hit-rate** — did it retrieve the document that controls the answer?
- **faithfulness** (LLM-judged) — is the answer grounded only in what was retrieved?
- **reconciliation effect** — corpus-only answer vs. final answer: did the live cross-check leave
  it alone, fix a stale corpus answer, or break a correct one?

Writes `accuracy_report.md` + `accuracy_results.csv`.

`golden_set.csv` is shared with the Week 2 project; `EXPECTED_SOURCES` in `run_accuracy_eval.py`
maps each row to its controlling corpus file (the same `data/rules/` corpus, same Pinecone index).
Row 23 expects `ask_back`, which `ANSWER_TYPES` has no value for — reported as a schema gap, not a
failure.
