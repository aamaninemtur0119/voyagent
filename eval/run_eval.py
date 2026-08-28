"""Eval harness for Voyagent's AGENTIC behavior — control flow, state, tool-failure recovery, and
human-in-the-loop — as code, not just a manually-run smoke test. This deliberately does not
re-test RAG quality (the Eligibility Agent's retrieval/faithfulness eval already exists in the
Week 2 project this was built on); it tests the properties that are specific to being an agentic
system: does it actually recover from a tool failure instead of crashing, does the approval gate
actually block a write, does replanning actually update state and terminate.

Each check is a real graph.invoke()/Command(resume=...) call against a live LLM and live APIs —
not mocked reasoning — except where a tool failure is deliberately simulated to test recovery.
"""

import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from voyagent.agents import logistics as logistics_module
from voyagent.graph import MAX_REPLANS, build_graph

EVAL_DIR = Path(__file__).resolve().parent

BASE_STATE = {
    "nationality": "China",
    "destination_country": "Japan",
    "destination_city": "Tokyo",
    "origin": "New York",
    "purpose": "tourism",
    "duration": "14 days",
    "start_date": "2026-10-01",
    "end_date": "2026-10-15",
    "preferences": {"dietary": "Vegetarian", "family_friendly": True, "outdoor_seating": False, "budget_level": "Any"},
    "agent_trace": [],
    "errors": [],
    "replan_count": 0,
}


def new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def interrupt_type(result: dict) -> str | None:
    if not result.get("__interrupt__"):
        return None
    return result["__interrupt__"][0].value["type"]


class Check:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.detail = ""

    def require(self, condition: bool, detail: str) -> None:
        if not condition:
            self.passed = False
            self.detail = detail


def check_happy_path(graph) -> Check:
    c = Check("Happy path: all agents succeed, parallel branches both complete, pauses at calendar approval")
    result = graph.invoke(BASE_STATE, config=new_config())
    agents_done = {t["agent"] for t in result.get("agent_trace", []) if t["status"] == "done"}
    c.require({"eligibility", "logistics", "experience", "orchestrator"} <= agents_done, f"expected all 4 agents done, got {agents_done}")
    c.require(interrupt_type(result) == "calendar_write_approval", f"expected calendar_write_approval interrupt, got {interrupt_type(result)}")
    return c


def check_tool_failure_recovery(graph) -> Check:
    c = Check("Tool failure recovery: logistics fails twice (retry exhausted), graph continues, experience still runs")
    original = logistics_module.run
    logistics_module.run = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated outage"))
    try:
        result = graph.invoke(BASE_STATE, config=new_config())
    finally:
        logistics_module.run = original

    trace_by_agent = {t["agent"]: t for t in result.get("agent_trace", [])}
    c.require(trace_by_agent.get("logistics", {}).get("status") == "failed", "expected logistics marked 'failed' in trace")
    c.require(trace_by_agent.get("experience", {}).get("status") == "done", "expected experience to still complete despite logistics failing")
    c.require(result.get("logistics") is None, "expected logistics result to be None, not a crashed exception")
    c.require(any("logistics" in e.lower() for e in result.get("errors", [])), "expected the failure recorded in errors")
    c.require(bool(result.get("itinerary")), "expected synthesis to still produce an itinerary despite the failure")
    return c


def check_two_independent_approval_gates(graph) -> Check:
    c = Check("Two independent HITL gates: approving calendar leads to a SECOND, separate export approval")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    after_calendar = graph.invoke(Command(resume={"approved": True}), config=config)
    c.require(interrupt_type(after_calendar) == "export_approval", f"expected export_approval interrupt after calendar approval, got {interrupt_type(after_calendar)}")
    after_export = graph.invoke(Command(resume={"approved": True}), config=config)
    c.require(after_export.get("export_result", {}).get("status") == "written", "expected export_result status 'written' after approving")
    c.require(Path(after_export["export_result"]["path"]).exists(), "expected the exported file to actually exist on disk")
    return c


def check_rejection_without_feedback_does_not_replan(graph) -> Check:
    c = Check("Rejecting with NO feedback proceeds normally, does not trigger a replan")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    result = graph.invoke(Command(resume={"approved": False, "feedback": ""}), config=config)
    c.require(result.get("replan_requested") in (False, None), "expected replan_requested to be falsy with no feedback")
    c.require(interrupt_type(result) == "export_approval", f"expected to proceed to export_approval, got {interrupt_type(result)}")
    return c


def check_replan_updates_state_and_reruns_agents(graph) -> Check:
    c = Check("Rejecting WITH feedback updates state (dates/budget) and re-runs Logistics+Experience")
    config = new_config()
    result = graph.invoke(BASE_STATE, config=config)
    trace_len_before = len(result.get("agent_trace", []))
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "push the trip back to 2026-11-01 and use the cheapest hotels"}),
        config=config,
    )
    c.require(result.get("start_date") == "2026-11-01", f"expected start_date updated to 2026-11-01, got {result.get('start_date')}")
    c.require(result.get("preferences", {}).get("budget_level") == "$", f"expected budget_level updated to '$', got {result.get('preferences', {}).get('budget_level')}")
    c.require(result.get("replan_count") == 1, f"expected replan_count == 1, got {result.get('replan_count')}")
    c.require(len(result.get("agent_trace", [])) > trace_len_before, "expected new trace entries from the re-run")
    c.require(interrupt_type(result) == "calendar_write_approval", "expected a fresh calendar approval pause after replanning")
    return c


def check_replan_cap_terminates(graph) -> Check:
    c = Check(f"Replan cap: rejecting with feedback repeatedly stops looping after MAX_REPLANS={MAX_REPLANS}")
    config = new_config()
    result = graph.invoke(BASE_STATE, config=config)
    for i in range(MAX_REPLANS + 3):  # deliberately exceed the cap
        if not result.get("__interrupt__"):
            break
        result = graph.invoke(Command(resume={"approved": False, "feedback": f"change something round {i}"}), config=config)
    c.require(not result.get("__interrupt__"), "expected the graph to finish (not still interrupted) well within MAX_REPLANS+3 rounds")
    c.require(result.get("replan_count", 0) <= MAX_REPLANS, f"expected replan_count capped at {MAX_REPLANS}, got {result.get('replan_count')}")
    return c


def run() -> None:
    graph = build_graph()
    checks = [
        check_happy_path,
        check_tool_failure_recovery,
        check_two_independent_approval_gates,
        check_rejection_without_feedback_does_not_replan,
        check_replan_updates_state_and_reruns_agents,
        check_replan_cap_terminates,
    ]
    results = []
    for fn in checks:
        print(f"Running: {fn.__name__} ...")
        try:
            results.append(fn(graph))
        except Exception as e:  # noqa: BLE001 - a check itself blowing up is still a result to report
            c = Check(fn.__name__)
            c.passed = False
            c.detail = f"Check raised an exception: {e}"
            results.append(c)

    lines = ["# Voyagent Agentic-Behavior Eval\n"]
    passed = sum(r.passed for r in results)
    lines.append(f"**{passed}/{len(results)} checks passed**\n")
    for r in results:
        icon = "✅" if r.passed else "❌"
        lines.append(f"- {icon} {r.name}" + (f" — {r.detail}" if not r.passed else ""))
    report = "\n".join(lines)
    print("\n" + report)
    (EVAL_DIR / "eval_report.md").write_text(report + "\n")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    run()
