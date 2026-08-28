# Video Script (5 min max)

**0:00–0:25 — What it is**
"This is Voyagent — a multi-agent trip-planning orchestrator. Give it a traveler's situation once, and three specialized agents — Eligibility, Logistics, and Experience — run under an orchestrator that carries real state between them, recovers from tool failures instead of crashing, and pauses for human approval before it writes anything."

**0:25–0:45 — How it relates to my Week 2 project**
"It's built on top of a RAG project I did last week — reusing its retrieval and live-API tools — but the orchestration layer here is new: real state, control flow, failure recovery, and human-in-the-loop, which is what this week is actually about."

**0:45–2:15 — Live demo: happy path + both approval gates**
- Fill in the form (nationality, destination, dates, preferences) → "Plan My Trip."
- Point at the agent-trace panel as it fills in — narrate: "Eligibility ran the visa check, routed to a deadline check since a visa's required, then Logistics and Experience ran *in parallel* — you can see they don't have a fixed order — then synthesis combined everything."
- Show the calendar approval prompt with the actual deadlines listed. Click **Approve**.
- Point out: it doesn't finish — it pauses on a *second*, independent approval ("Save Itinerary"). "This is a different write action, gated separately, to show the approval pattern generalizes, not a one-off around Calendar." Approve it too.
- Show the final trip briefing.

**2:15–3:15 — Live demo: adaptive replanning**
- Submit a new trip. On the calendar approval step, type feedback instead of just rejecting — e.g. *"push the trip back a month, cheapest hotels only"* — and click Reject.
- Narrate while it runs: "It doesn't just cancel — an LLM interprets that feedback, updates the actual state — new start date, new budget — and re-runs Logistics and Experience with the new input. It's a real cycle in the graph, not a restart." Show the updated dates/budget and the new approval prompt.
- Mention the cap: "This is capped at 2 replans so it can't loop forever — that's tested explicitly, not just hoped."

**3:15–4:00 — Tool-failure recovery (can't trigger a real API outage on demand, so show the eval)**
- Run `uv run python eval/run_eval.py` in a terminal, or show `eval/eval_report.md`.
- Narrate: "This test monkey-patches the Logistics agent to always throw, like a real API outage would. It retries once, fails again, and the graph keeps going — Experience still runs, synthesis still produces a briefing, and the briefing explicitly says the logistics info wasn't available, instead of silently hiding it or crashing. This is the actual point of the whole assignment — an agent that only works on the happy path isn't finished."

**4:00–4:40 — How AI coding tools were used**
- One or two concrete, honest examples: "I asked Claude Code to extend my Week 2 RAG agent into a genuine multi-agent system rather than start from scratch. At one point I asked 'where is multi-agent here?' and it found that two of my three agents had zero LLM calls — they were just sorting API results — and fixed that. I also had it test the human-approval interrupt and the tool-failure recovery with actual induced failures, not just written and assumed to work."

**4:40–5:00 — Wrap-up**
- Repeat the one-liner. Mention the GitHub link and that MCP was considered and deliberately deferred, since it wasn't the priority given the time available — state/control-flow/failure-recovery/HITL came first.
