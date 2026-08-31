# Eligibility Agent — Accuracy Eval

Golden set: 25 questions (`eval/golden_set.csv`). Per-row output: `eval/accuracy_results.csv`.
This is the last completed full run. Regenerate with `uv run python -m eval.run_accuracy_eval`.

## Summary

- **Answer-type accuracy**: 71% (17/24 rows the schema can represent)
- **Retrieval hit-rate**: 100% (23/23 rows with a known controlling source; 2 excluded)
- **Faithfulness**: the weak spot — LLM-judged, corpus answer vs. only what was retrieved; a clean number is pending re-measurement after the grounding-gate fix (see below)

## Live-reconciliation effect (corpus answer vs. final answer)

- unchanged: 22
- FIXED a stale corpus answer: 1  (row 10, India → UK — stale refusal became correct "visa required")
- BROKE a correct corpus answer: 1  (row 2, India → Japan — correct "visa required" flipped to a wrong "visa waiver with ETA", over-weighting eVISA pages)
- changed refuse_and_verify -> different_visa_category_required (both wrong): 1  (row 18, India → Australia)

## By category

| Category | n | Type acc | Retrieval |
|---|---|---|---|
| ask_back | 1 | n/a | n/a |
| duration_variant | 1 | 0% | 100% |
| matrix | 20 | 75% | 100% |
| purpose_variant | 2 | 50% | 100% |
| refusal | 1 | 100% | n/a |

(Faithfulness is omitted here pending a clean re-measurement after the grounding-gate fix.)

## Schema gap

Row 23 expects `ask_back`, which `ANSWER_TYPES` has no value for — the pipeline can't ask a
clarifying question before committing to an answer_type. It returned `refuse_and_verify`. This is
a deliberate schema limitation, not a retrieval or generation failure.

## Where the answer-type misses actually are

Several "misses" are the agent being *more precise* than the Week-2 label, not wrong: rows 17, 19,
21 and 24 are labelled `visa_required` but the agent returns `different_visa_category_required`
(correctly — those travelers can't use an ETA / visa waiver and need a specific subclass or B-1).

## Faithfulness — the axis with the most headroom

The judge's notes are consistent: the generated answers state precise facts the retrieved chunks
don't contain — visa-waiver-program membership ("India is not part of the VWP"), specific fee
amounts, the name and date of a 2026 entry proclamation, where to apply, passport-validity
windows. The model fills gaps from general knowledge despite being told to use only its sources.
A clean faithfulness number is pending a re-measurement (the eval suite stalls under API
rate-limiting).
