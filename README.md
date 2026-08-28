# Voyagent

Voyagent is a multi-agent trip-planning orchestrator. Give it a traveler's situation once — nationality, destination, dates, preferences — and an orchestrator coordinates three specialized agents to produce a complete trip briefing, with real state passed between steps, tool-failure recovery, and a human-approval gate before anything gets written to a calendar.

This is a Week 3 ("Build Your AI Agent") project, built on top of the retrieval/live-API infrastructure from a Week 2 RAG project ([crosscheck-travel-agent](https://github.com/aamaninemtur0119/crosscheck-travel-agent)) — the tools are reused; the orchestration layer (state, control flow, failure recovery, human-in-the-loop) is new.

**One-liner**: *Voyagent helps a traveler get a complete, grounded trip plan in a Streamlit app, replacing the need to separately check visa rules, compare flights/hotels, and research restaurants/activities across different tabs and sites. It plans and executes autonomously using 3 specialized agents, hands off to a human before writing anything to Google Calendar, and I'll know it works when a traveler gets a usable plan even when one of the agents' tools fails along the way, not just on the happy path.*

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              Orchestrator (LangGraph)         │
                         └─────────────────────────────────────────────┘
START ──► Eligibility Agent ──► (conditional) ──► [Logistics Agent ‖ Experience Agent] ──► Synthesize ──► 🖐 Calendar Approval ──► Finalize ──► 🖐 Export Approval ──► END
              │                  Deadlines Agent      (run in parallel,      │                                     │  ▲                                              │
              │                  (only if visa/ETA     fan-in to Synthesize)  │                            writes to Google      rejected WITH feedback loops        writes itinerary
              │                   needed + start date                        │                            Calendar, or not,     back to re-run Logistics/           to a file, or not,
              │                   given)                                     │                            only on approval      Experience with updated state       only on approval
              ▼                                                              ▼                             (capped at MAX_REPLANS=2)
     RAG: Pinecone (hybrid                                            Google Places API (accommodation,
     dense+BM25) + rerank,                                            restaurants, places, activities)
     deterministic evidence                                           + MCP-based real flight search
     gate + topic-mismatch                                            (Amadeus) with deep-link fallback
     check
```

Every agent node is wrapped with a retry-once policy. If a node still fails after retrying, the graph does **not** stop — it records the failure, continues to the next agent, and the synthesis step explicitly names what wasn't available rather than silently omitting it. Two independent write actions (Google Calendar, saving the itinerary to a file) each sit behind their own real LangGraph `interrupt()` — the graph pauses, the UI shows exactly what would be written, and execution only resumes once a human approves or rejects it. Rejecting the calendar step *with feedback* (e.g. "push the trip back a month, cheapest hotels only") triggers adaptive replanning: an LLM interprets the feedback, updates state, and the graph loops back to re-run Logistics/Experience with the new input — a real cycle, capped to guarantee termination.

## Agents

- **Eligibility Agent** (`agents/eligibility.py`) — visa/entry-requirements RAG: hybrid retrieval (dense + BM25) + rerank, a deterministic confidence gate on the top rerank score (refuses without even calling generation when retrieval is genuinely weak), and an explicit topic-vs-purpose mismatch check (catches a document that shares surface keywords with an off-topic question, which a score threshold alone can miss). Carried over from a Week 2 RAG project, including fixes found via real testing (nationality normalization, empty-answer-text fallback).
- **Logistics Agent** (`agents/logistics.py`) — real flight search via an MCP tool (Amadeus-backed, `mcp_servers/flights_server.py`) with an LLM recommending the best option against budget/cabin preference, falling back to deep links (clearly labeled) if flight search isn't configured or fails after a retry; plus live hotel search (Google Places), curated and ranked by an LLM against budget/preferences.
- **Experience Agent** (`agents/experience.py`) — live restaurant/place/activity search (Google Places), each curated by an LLM against dietary/family/outdoor-seating preferences.
- **Orchestrator** (`graph.py`) — the LangGraph `StateGraph` itself: routing, parallel fan-out/fan-in, retry/failure handling, synthesis, both approval gates, and the adaptive replanning loop.

Results (hotels, restaurants, places, activities, flight offers) are rendered as structured cards directly from state — each with a real Maps/booking link and price/rating — rather than relying on the synthesis LLM to reproduce a URL in prose, which turned out to be unreliable (it would paraphrase or drop links). The synthesized narrative focuses on the visa summary, deadline timeline, and (for a long single-city tourism trip) an explicit, clearly-labeled recommendation to consider a multi-city itinerary — a suggestion, not something the graph actually executes.

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
AMADEUS_API_KEY=...       # optional — free self-service test account at developers.amadeus.com
AMADEUS_API_SECRET=...    # without these, flight search gracefully falls back to deep links
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
