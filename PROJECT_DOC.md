# Voyagent — Multi-Agent Trip-Planning Orchestrator

Repo: https://github.com/aamaninemtur0119/voyagent
Built on top of: [crosscheck-travel-agent](https://github.com/aamaninemtur0119/crosscheck-travel-agent) (Week 2)

## Project Overview

Voyagent is an agentic system, not a one-shot call or a RAG lookup: given a traveler's situation once, an orchestrator plans and executes a sequence of steps across three specialized agents, carries real state between them, recovers from a tool failure without crashing, and pauses for explicit human approval before the one action that actually writes something (a Google Calendar event).

One-liner: *Voyagent helps a traveler get a complete, grounded trip plan in a Streamlit app, replacing the need to separately check visa rules, compare flights/hotels, and research restaurants/activities across different tabs and sites. It plans and executes autonomously using 3 specialized agents, hands off to a human before writing anything to Google Calendar, and I'll know it works when a traveler gets a usable plan even when one of the agents' tools fails along the way, not just on the happy path.*

**Why build this on top of Week 2 rather than starting from a blank domain**: Week 2's crosscheck project already had real, tested tools (visa RAG, flight links, live restaurant/place/activity/hotel search, a calendar-write action) but orchestrated them with a flat single-turn tool-selector — one message in, the LLM picks a tool, done. That pattern has no persistent plan, no state carried across steps, and no tool-failure recovery (a single tool exception would crash the whole call). Week 3 is graded on exactly those missing properties, so reusing the tool layer and building a genuinely new orchestration layer on top was the highest-leverage use of a 5-day window, rather than re-building a whole new tool set in an unfamiliar domain.

## Datasets Used

Same visa-requirements corpus as Week 2 (`data/rules/`, 28 government-sourced documents, 5 destinations) and the same Pinecone index/namespace — no re-ingestion was needed. Live data sources are unchanged too: Google Places API (New) for restaurants/places/activities/accommodation, and deep-link construction (not a live pricing API) for flights.

## Agent Architecture

Three specialized agents, coordinated by a LangGraph `StateGraph` orchestrator:

1. **Eligibility Agent** — visa/entry-requirements RAG (hybrid dense+BM25 retrieval, rerank, a deterministic evidence gate, a topic-mismatch check).
2. **Logistics Agent** — flight deep links + live hotel search, curated by budget preference.
3. **Experience Agent** — live restaurant/place/activity search, curated by dietary/family preferences.

Control flow: `eligibility → (conditional) deadlines → logistics → experience → synthesize → human_approval (interrupt) → finalize`. The conditional edge after Eligibility only routes to a deadline-extraction step when the visa outcome actually requires one (visa required, or an ETA/ETIAS with its own lead time) *and* the traveler gave a start date — real branching based on what a prior agent actually found, not a fixed sequence.

**State**: a single `TripState` (TypedDict) checkpointed by LangGraph's `MemorySaver`, threaded through every node. `agent_trace` and `errors` use an additive reducer so every node's contribution accumulates across the run rather than overwriting the previous one — this is what the UI's live trace panel renders.

**Tool-failure handling**: every agent node wraps its call in a retry-once helper. If the second attempt also fails, the node does not raise — it returns a state update marking that agent `failed` with the error message, and the graph *continues* to the next node. The synthesis step is explicitly instructed to name any missing section as a tool failure rather than silently omit it. Verified with a genuine simulated failure (see Iterations).

**Human-in-the-loop**: `human_approval_node` calls LangGraph's `interrupt()` with the exact deadlines that would be written, and only proceeds to the actual `write_to_calendar()` call once the graph is resumed with an explicit `Command(resume={"approved": True/False})`. Reads (all three agents) are fully autonomous; this is the only node in the whole graph that can create anything, and it cannot proceed without a human decision.

## Prompts Used During Vibe Coding

The core system-design prompt that shaped this session (paraphrased from the actual conversation, not the code): *"the project handout has a lot of multi-agent examples — can we extend [the Week 2 project] and incorporate MCP and multi-agent [patterns]?"* — this is what led to decomposing a single flat tool-selector into three specialized agents with a real orchestrator, rather than just relabeling the existing agent.

Within the code, the load-bearing prompts are the same discipline as Week 2, reused directly: the Eligibility Agent's generation prompt forbids outside knowledge and requires `refuse_and_verify` when a source is off-topic or unconfirmed; the synthesis prompt explicitly instructs that a missing agent result must be named as a failure, not silently dropped, which is what makes the failure-recovery behavior visible in the final output instead of just in a log.

## Iterations Tried

- **MCP was considered and deliberately deferred.** It's a legitimate way to expose the tool layer (any agent becomes an MCP client discovering tools over a standard protocol instead of importing Python functions directly), but it isn't in this track's listed building blocks (LangChain agents/tools, LangGraph state machines, checkpointers, interrupts, LangSmith), and with a 5-day window the priority was getting state/control-flow/failure-recovery/HITL genuinely solid first. Not built in this pass.
- **Retry-then-continue was tested with a real simulated failure, not assumed to work.** `logistics.run` was monkey-patched to always raise (simulating a Google Places API outage) and the graph was run end-to-end: the Logistics Agent retried once, failed again, was marked `failed` in the trace, and the graph continued through Experience and Synthesis without crashing — the final itinerary explicitly named the missing logistics section as a tool failure. This is the single most important verification in the whole build, given the brief's own framing ("if your agent works on the happy path but falls over on the first tool failure, you have not finished").
- **The human-approval interrupt was tested for a real resume, not just a pause.** A first attempt to test resume across two separate Python processes failed — not a graph bug, but a reminder that `MemorySaver()` is in-process memory only and doesn't survive a process exit. Retested correctly within a single process/graph object: the graph paused at `interrupt()` with the exact deadline payload, and resuming with `Command(resume={"approved": True})` correctly proceeded to the calendar-write step (which itself correctly reported `not_configured`, since this new project doesn't have its own Google Calendar OAuth credentials set up yet).
- **Chose not to convert the Eligibility Agent's internal retrieve-then-generate logic into more graph nodes.** It's already a self-contained, testable pipeline (carried over from Week 2); wrapping it in more LangGraph nodes wouldn't add real state or branching, just structure for its own sake — the same "don't use a framework where a plain function does the job" judgment applied in Week 2's LangGraph removal, applied here to keep each agent itself simple even though the outer orchestration is now a real graph.

## Learnings / Observations

- **The Week 2 correctness fixes were worth carrying forward, not just the tools.** The Eligibility Agent inherits the deterministic evidence gate and topic-mismatch check built in response to Week 2 reviewer feedback ("your refusal relies on a prompt, not a verified evidence gate") — reusing hard-won correctness work, not just reusable code, was a real time-saver.
- **A retry-then-continue policy has to be tested with an actual induced failure, not just written and assumed correct.** It would have been easy to write the try/except wrapper, see the happy path work, and call it done — the monkey-patched-failure test is what actually proved the graph degrades gracefully instead of crashing.
- **In-process checkpointer state is a real operational constraint worth knowing early, not discovering during a demo.** `MemorySaver()` not surviving a process restart is expected behavior, but it would have looked like a broken interrupt/resume feature if not understood before building the UI around it (the UI keeps one graph instance alive via `st.cache_resource` for exactly this reason).
