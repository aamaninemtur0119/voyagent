# Pack Your Bags

Pack Your Bags is a multi-agent trip-planning orchestrator. Give it a traveler's situation once — nationality, destination, dates, preferences — and an orchestrator coordinates three specialized agents to produce a complete trip briefing, with real state passed between steps, tool-failure recovery, and two independent human-approval gates before either write action (writing deadlines to a calendar, and emailing the finished itinerary to the traveler).

This is a Week 3 ("Build Your AI Agent") project, built on top of the retrieval/live-API infrastructure from a Week 2 RAG project ([crosscheck-travel-agent](https://github.com/aamaninemtur0119/crosscheck-travel-agent)) — the tools are reused; the orchestration layer (state, control flow, failure recovery, human-in-the-loop) is new.

**One-liner**: *Pack Your Bags helps a traveler get a complete, grounded trip plan in a Streamlit app, replacing the need to separately check visa rules, compare flights/hotels, and research restaurants/activities across different tabs and sites. It plans and executes autonomously using 3 specialized agents, hands off to a human before writing anything to Google Calendar, and I'll know it works when a traveler gets a usable plan even when one of the agents' tools fails along the way, not just on the happy path.*

**Full write-up** (use case, happy path, multi-agent design, detailed tool calls, error handling, RAG, eval results, challenges): [`WRITEUP.md`](WRITEUP.md).

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              Orchestrator (LangGraph)         │
                         └─────────────────────────────────────────────┘
START ─► Validate ─► Eligibility Agent ─► (conditional) ─► [Logistics Agent ‖ Experience Agent] ─► Synthesize ─► 🖐 Review ─► Finalize ─► 🖐 Email Approval ─► Send ─► END
             │                   │                Deadlines Agent      (run in parallel,     │              Approve /       │  ▲                                          │
             │                   │                (only if visa/ETA     fan-in to Synthesize) │       Revise / Cancel   writes to Google   Revise-with-feedback loops    emails the
             ▼                   ▼                 needed + start date                        │                        Calendar, or not,    back to re-run Logistics/    itinerary, or not,
   city not in destination   RAG: Pinecone (hybrid                                            ▼                        only on approval      Experience (+ deadlines if     only on approval
   country / unrecognizable  dense+BM25) + rerank,                                    Google Places API                                      dates moved), updated state
   → END + a fix-it message  deterministic evidence gate + topic-mismatch check,      (accommodation, restaurants,                            (Cancel ─► END; capped at
                             then reconciled against live You.com search              places, activities) + MCP                                MAX_REPLANS=2); next screen
                             (official live source wins), sources tagged corpus/live  flight search (Duffel) w/                                shows old → new values
                                                                                     deep-link fallback
```

Before any agent runs, a **validation gate** checks the request is coherent — every destination city has to plausibly belong to the destination country (typos tolerated). An incoherent request (e.g. destination country Japan, city Toronto) routes straight to `END` with a plain-language fix-it message instead of producing a Frankenstein plan that checks a Japan visa while pricing flights to Toronto. It fails open: if the check itself errors, planning proceeds.

Every agent node is wrapped with a retry-once policy. If a node still fails after retrying, the graph does **not** stop — it records the failure, continues to the next agent, and the synthesis step explicitly names what wasn't available rather than silently omitting it. Failure handling is layered rather than all-or-nothing: every LLM structured-output call goes through a shared `structured()` helper (`llm.py`) that retries, then repairs the common "the model returned the whole object as a JSON string" malformation via a raw-JSON parse, and finally falls back to a caller-supplied default — so a garbled hotel-ranking response degrades to the raw (still real) results with flight search untouched, instead of throwing and taking the whole Logistics Agent down with it. Two independent write actions (writing deadlines to Google Calendar, and emailing the finished itinerary to the traveler) each sit behind their own real LangGraph `interrupt()` — the graph pauses, the UI shows exactly what would be written or sent, and execution only resumes on an explicit human decision. Each write also degrades the same way an agent does: retry once, then record the failure and finish rather than crash. The review gate offers three distinct outcomes: **Approve** (write the calendar events, continue to the email step), **Revise** with feedback (e.g. "push the trip back a month", "cheapest hotels only", "Kyoto instead of Tokyo") — an LLM interprets the feedback into a structured delta and the graph loops back to re-run the affected nodes:

- **Dates** → deadline timeline recomputed (a trip pushed back a month gets its visa deadlines pushed back with it), flights re-priced, every flight link's dates updated, recommendation re-run on the new offers.
- **Budget** → hotels re-filtered and re-ranked.
- **Destination** → applied as add / remove / swap *ops* on the existing city list, so "Kyoto instead of Tokyo" replaces only Tokyo and keeps every other city; the result is re-validated against the country (a revise can't turn it into a Japan-visa / Paris-flights mismatch); flights, hotels, and the itinerary re-point to the new primary city.

A real cycle, capped at `MAX_REPLANS` to guarantee termination; the next review screen shows exactly what changed (old → new). **Cancel** routes straight to `END` with nothing written and nothing sent.

## Agents

- **Eligibility Agent** (`agents/eligibility.py`) — visa/entry-requirements RAG: hybrid retrieval (dense + BM25) + rerank, a deterministic confidence gate on the top rerank score (refuses without even calling generation when retrieval is genuinely weak), and an explicit topic-vs-purpose mismatch check (catches a document that shares surface keywords with an off-topic question, which a score threshold alone can miss). The corpus has no automated freshness mechanism, so the corpus answer is then **reconciled against live You.com results**: when a live result from a clearly official source (a government domain / immigration authority / embassy) directly addresses this traveler's nationality + destination + purpose, it's treated as authoritative and drives the answer (with a `divergence_note` explaining what changed); otherwise the corpus answer stands and live only fills gaps. Every run returns a `sources` list tagging each source `corpus` or `live`, and which one drove the answer — rendered in the UI. Carried over from a Week 2 RAG project, including fixes found via real testing (nationality normalization, empty-answer-text fallback).
- **Logistics Agent** (`agents/logistics.py`) — real flight search via an MCP tool (Duffel API, test mode, `mcp_servers/flights_server.py`) with an LLM recommending the best option against budget/cabin preference, falling back to deep links (clearly labeled) if flight search isn't configured or fails after a retry; plus live hotel search (Google Places), curated and ranked by an LLM against budget/preferences.
- **Experience Agent** (`agents/experience.py`) — live restaurant/place/activity search (Google Places), each curated by an LLM against dietary/family/outdoor-seating preferences.
- **Orchestrator** (`graph.py`) — the LangGraph `StateGraph` itself: routing, parallel fan-out/fan-in, retry/failure handling, synthesis, both approval gates, and the adaptive replanning loop.

Results (hotels, restaurants, places, activities, flight offers) are rendered as structured cards directly from state — each with a real Maps/booking link and price/rating — rather than relying on the synthesis LLM to reproduce a URL in prose, which turned out to be unreliable (it would paraphrase or drop links). The synthesized narrative focuses on the visa summary, deadline timeline, and (for a long single-city tourism trip) an explicit, clearly-labeled recommendation to consider a multi-city itinerary — a suggestion, not something the graph actually executes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/aamaninemtur0119/pack-your-bags.git
cd pack-your-bags
uv sync
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
GOOGLE_PLACES_API_KEY=...
DUFFEL_API_KEY=...        # optional — free instant test token at duffel.com; without it, flight search falls back to deep links
YOU_API_KEY=...           # optional — powers the Eligibility Agent's live recency cross-check; skipped if unset

# optional — used only for the second write action (emailing the itinerary). Without these, that
# step reports "not connected yet" instead of failing. Works with any SMTP provider (e.g. a Gmail
# app password: smtp.gmail.com / 587).
SMTP_HOST=...
SMTP_PORT=587
SMTP_USERNAME=...
SMTP_PASSWORD=...
SMTP_FROM=...             # defaults to SMTP_USERNAME when blank
```

(Reuses the same visa-rules corpus and Pinecone index as the Week 2 project — if you haven't ingested it yet: `uv run python -m pack_your_bags.retrieval.ingest`.)

Run it:

```bash
uv run streamlit run app.py
```

Both write actions are optional and degrade gracefully:

- **Google Calendar** — requires your own `google_calendar_credentials.json` (Google Cloud Console → OAuth client, Calendar API enabled) in the project root; without it, the calendar step reports `not_configured` instead of failing.
- **Email** — requires the `SMTP_*` vars above; without them, the email step reports `not_configured`. The approval gate still appears either way, so the human-in-the-loop flow is demonstrable with no email account configured.

## Project Structure

```
app.py                          Streamlit UI — form + approval gates + structured result cards
src/pack_your_bags/
├── graph.py                    The orchestrator: LangGraph StateGraph, retry/failure handling, HITL
├── state.py                    Shared TripState schema
├── llm.py                      Shared LLM instance + resilient structured-output helper (retry → repair → default)
├── agents/
│   ├── eligibility.py          Visa/entry RAG (hybrid retrieval + evidence gate)
│   ├── logistics.py            Flights + accommodation
│   └── experience.py           Restaurants + places + activities
├── tools/
│   ├── google_places.py        Live Google Places API (New) search
│   ├── flights.py               Flight search deep-link builder (generic + per-airline Kayak links)
│   ├── duffel.py                Real flight-offer search via the Duffel API (test mode)
│   ├── calendar_actions.py     Deadline extraction (LLM) + Google Calendar write  [HITL gate #1]
│   └── email_actions.py        Emails the finished itinerary to the traveler (SMTP)  [HITL gate #2]
├── mcp_servers/                flights_server.py — exposes Duffel flight search as an MCP tool
└── retrieval/                  Hybrid dense+sparse retrieval + rerank (shared by Eligibility Agent)
data/rules/                     Visa-requirements corpus (28 files, reused from Week 2)
eval/
├── run_eval.py                 Agentic-behavior suite — control flow, state, tool failure, HITL, robustness
├── run_accuracy_eval.py        Eligibility accuracy vs. golden_set.csv — answer-type, retrieval, faithfulness, reconciliation effect
├── golden_set.csv              25 labeled visa questions (shared with the Week 2 project)
└── README.md                   How to run both, and what each measures
```

## Evals

Two suites in `eval/` (details in `eval/README.md`), both making real LLM + API calls.

**Agentic behavior** — `uv run python -m eval.run_eval` — **18 / 18**. Core (11): control flow, shared state, tool-failure recovery, both HITL gates, the revise cycle, replan-cap termination. Robustness (7): breaking inputs (typos, unknown nationality, city/country mismatch, gibberish) where the bar is graceful behavior; plus corpus-vs-live reconciliation and source tagging. Each check is a full graph run. Note: a single uninterrupted run is unreliable here (occasional MCP-subprocess hang + API rate-limiting), so the harness persists each check's result as it completes and the 18/18 is assembled from that record.

**Eligibility accuracy** — `uv run python -m eval.run_accuracy_eval` — against `golden_set.csv` (25 labeled visa questions). Latest completed run:

| Axis | Result |
|---|---|
| Retrieval hit-rate | **100%** (23/23 rows with a known controlling doc) |
| Answer-type accuracy | **71%** (17/24) — understated; several "misses" are the agent correctly returning `different_visa_category_required` where the coarser Week-2 label says `visa_required` |
| Faithfulness (LLM-judged, corpus answer vs. retrieved chunks) | **52%** — the weak spot; the generation step was stating facts not in its sources (VWP membership, fee amounts, a 2026 proclamation, where to apply) |
| Reconciliation effect | 22 unchanged · 1 fixed a stale corpus answer · 1 broke a correct one · 1 lateral |

**Faithfulness fix** (shipped, re-measurement pending due to rate limits): an in-pipeline **grounding gate** after generation strips any claim the retrieved chunks don't support; the generation and reconciliation prompts got an explicit "do not state X unless a source says it verbatim" list; the harness now judges the corpus answer against its own corpus-derived requirements.
