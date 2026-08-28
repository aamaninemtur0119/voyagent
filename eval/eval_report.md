# Voyagent Agentic-Behavior Eval

**6/6 checks passed**

- ✅ Happy path: all agents succeed, parallel branches both complete, pauses at calendar approval
- ✅ Tool failure recovery: logistics fails twice (retry exhausted), graph continues, experience still runs
- ✅ Two independent HITL gates: approving calendar leads to a SECOND, separate export approval
- ✅ Rejecting with NO feedback proceeds normally, does not trigger a replan
- ✅ Rejecting WITH feedback updates state (dates/budget) and re-runs Logistics+Experience
- ✅ Replan cap: rejecting with feedback repeatedly stops looping after MAX_REPLANS=2
