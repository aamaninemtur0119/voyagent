# Pack Your Bags Agentic-Behavior Eval

**4/4 checks passed** (running — 4/18 complete)

## Core — control flow, state, tool failure, HITL

- ✅ Happy path: all agents succeed, parallel branches both complete, pauses at calendar approval
- ✅ Tool failure recovery: logistics fails twice (retry exhausted), graph continues, experience still runs
- ✅ A malformed structured-output response degrades to the caller's default — never crashes the node
- ✅ Two independent HITL gates: approving calendar leads to a SECOND, separate email approval
- ⏳ … 7 core check(s) not yet run

## Robustness — breaking cases + corpus/live reconciliation

- ⏳ … 7 robustness check(s) not yet run
