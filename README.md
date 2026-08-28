# Voyagent

Voyagent is a multi-agent trip-planning orchestrator. Give it a traveler's situation once — nationality, destination, dates, preferences — and an orchestrator coordinates three specialized agents to produce a complete trip briefing, with real state passed between steps, tool-failure recovery, and a human-approval gate before anything gets written to a calendar.

This is a Week 3 ("Build Your AI Agent") project, built on top of the retrieval/live-API infrastructure from a Week 2 RAG project ([crosscheck-travel-agent](https://github.com/aamaninemtur0119/crosscheck-travel-agent)) — the tools are reused; the orchestration layer (state, control flow, failure recovery, human-in-the-loop) is new.

**One-liner**: *Voyagent helps a traveler get a complete, grounded trip plan in a Streamlit app, replacing the need to separately check visa rules, compare flights/hotels, and research restaurants/activities across different tabs and sites. It plans and executes autonomously using 3 specialized agents, hands off to a human before writing anything to Google Calendar, and I'll know it works when a traveler gets a usable plan even when one of the agents' tools fails along the way, not just on the happy path.*

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              Orchestrator (LangGraph)         │
                         └─────────────────────────────────────────────┘
START ──► Eligibility Agent ──► (conditional) ──► Logistics Agent ──► Experience Agent ──► Synthesize ──► 🖐 Human Approval ──► Finalize ──► END
              │                  Deadlines Agent         │                    │                                  │
              │                  (only if visa/ETA        │                    │                          writes to Google
              │                   needed + start date      │                    │                          Calendar, or not,
              │                   given)                   │                    │                          only on approval
              ▼                                            ▼                    ▼
     RAG: Pinecone (hybrid                          Google Places API    Google Places API
     dense+BM25) + rerank,                          (accommodation) +    (restaurants, places,
     deterministic evidence                          flight deep links    activities)
     gate + topic-mismatch
     check
```

Every agent node is wrapped with a retry-once policy. If a node still fails after retrying, the graph does **not** stop — it records the failure, continues to the next agent, and the synthesis step explicitly names what wasn't available rather than silently omitting it. The only write action (Google Calendar) sits behind a real LangGraph `interrupt()` — the graph pauses, the UI shows exactly what would be written, and execution only resumes once a human approves or rejects it.

## Agents

- **Eligibility Agent** (`agents/eligibility.py`) — visa/entry-requirements RAG: hybrid retrieval (dense + BM25) + rerank, a deterministic confidence gate on the top rerank score (refuses without even calling generation when retrieval is genuinely weak), and an explicit topic-vs-purpose mismatch check (catches a document that shares surface keywords with an off-topic question, which a score threshold alone can miss). Carried over from a Week 2 RAG project, including fixes found via real testing (nationality normalization, empty-answer-text fallback).
- **Logistics Agent** (`agents/logistics.py`) — flight search deep links + live hotel search (Google Places), curated by the traveler's stated budget and ranked by rating.
- **Experience Agent** (`agents/experience.py`) — live restaurant/place/activity search (Google Places), curated by dietary and family-friendly preferences.
- **Orchestrator** (`graph.py`) — the LangGraph `StateGraph` itself: routing (does this trip need a deadline check at all?), retry/failure handling, synthesis, and the human-approval gate.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/aamaninemtur0119/voyagent.git
cd voyagent
uv sync
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=...
GOOGLE_PLACES_API_KEY=...
```

(Reuses the same visa-rules corpus and Pinecone index as the Week 2 project — if you haven't ingested it yet: `uv run python -m voyagent.retrieval.ingest`.)

Run it:

```bash
uv run streamlit run app.py
```

Optional — Google Calendar write requires your own `google_calendar_credentials.json` (Google Cloud Console → OAuth client, Calendar API enabled) in the project root; without it, the calendar step correctly reports `not_configured` instead of failing.

## Project Structure

```
app.py                          Streamlit UI — form + live agent-trace panel + approval gate
src/voyagent/
├── graph.py                    The orchestrator: LangGraph StateGraph, retry/failure handling, HITL
├── state.py                    Shared TripState schema
├── agents/
│   ├── eligibility.py          Visa/entry RAG (hybrid retrieval + evidence gate)
│   ├── logistics.py            Flights + accommodation
│   └── experience.py           Restaurants + places + activities
├── tools/
│   ├── google_places.py        Live Google Places API (New) search
│   ├── flights.py               Flight search deep-link builder
│   └── calendar_actions.py     Deadline extraction (LLM) + Google Calendar write
└── retrieval/                  Hybrid dense+sparse retrieval + rerank (shared by Eligibility Agent)
data/rules/                     Visa-requirements corpus (28 files, reused from Week 2)
```
