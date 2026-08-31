"""Eval harness for Voyagent's AGENTIC behavior — control flow, state, tool-failure recovery, and
human-in-the-loop — as code, not just a manually-run smoke test. This deliberately does not
re-test RAG quality (the Eligibility Agent's retrieval/faithfulness eval already exists in the
Week 2 project this was built on); it tests the properties that are specific to being an agentic
system: does it actually recover from a tool failure instead of crashing, does the approval gate
actually block a write, does replanning actually update state and terminate.

Two sections:
- CORE — control flow, state, tool failure, HITL.
- ROBUSTNESS — breaking-case inputs (typos, unknown nationality, city/country mismatch,
  gibberish) where the bar is GRACEFUL BEHAVIOR (no crash, sensible degradation, the problem
  flagged) rather than an accurate answer; plus the Eligibility Agent's corpus-vs-live
  reconciliation and source tagging.

Each check is a real graph.invoke()/Command(resume=...) call against a live LLM and live APIs —
not mocked reasoning — except where a failure or a specific live-search result is deliberately
simulated. The robustness section roughly doubles the total runtime.
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
    "traveler_email": "traveler@example.test",  # reserved non-routable TLD; send is stubbed in run() regardless
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


def check_structured_output_failure_degrades_not_crashes(graph) -> Check:
    c = Check("A malformed structured-output response degrades to the caller's default — never crashes the node")
    import voyagent.llm as llm_mod
    from voyagent.agents.logistics import LogisticsCuration

    class _BoomChain:
        def invoke(self, *a, **k):
            raise ValueError("simulated: model returned a malformed tool argument")

    class _FakeLLM:  # swap the whole object — a ChatAnthropic instance (pydantic) rejects attr assignment
        def with_structured_output(self, *a, **k):
            return _BoomChain()

        def invoke(self, *a, **k):  # the raw-JSON fallback path also fails
            raise ValueError("raw-JSON fallback also unavailable")

    real = llm_mod.llm
    llm_mod.llm = _FakeLLM()
    try:
        got = llm_mod.structured(LogisticsCuration, "irrelevant", default=None)
        c.require(got is None, f"expected structured(..., default=None) to return None on total failure, got {got!r}")
        raised = False
        try:
            llm_mod.structured(LogisticsCuration, "irrelevant")  # no default -> must surface the error
        except Exception:
            raised = True
        c.require(raised, "expected structured() with no default to re-raise on total failure")
    finally:
        llm_mod.llm = real
    return c


def check_two_independent_approval_gates(graph) -> Check:
    c = Check("Two independent HITL gates: approving calendar leads to a SECOND, separate email approval")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    after_calendar = graph.invoke(Command(resume={"approved": True}), config=config)
    c.require(interrupt_type(after_calendar) == "email_approval", f"expected email_approval interrupt after calendar approval, got {interrupt_type(after_calendar)}")
    after_email = graph.invoke(Command(resume={"approved": True}), config=config)
    status = (after_email.get("email_result") or {}).get("status")
    # 'sent' if SMTP is configured in this env, 'not_configured' if not, 'failed' on a real send error —
    # any of those means the approved write actually ran; 'skipped'/None means the gate didn't wire through.
    c.require(status in ("sent", "not_configured", "failed"), f"expected the approved email write to run, got email_result={after_email.get('email_result')}")
    c.require(not after_email.get("__interrupt__"), "expected the graph to finish after the second approval")
    return c


def check_email_rejection_does_not_send(graph) -> Check:
    c = Check("Rejecting the email gate records 'skipped' and sends nothing")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    graph.invoke(Command(resume={"approved": True}), config=config)  # approve calendar -> reach email gate
    after_email = graph.invoke(Command(resume={"approved": False}), config=config)
    c.require((after_email.get("email_result") or {}).get("status") == "skipped", f"expected email_result 'skipped', got {after_email.get('email_result')}")
    return c


def check_rejection_without_feedback_does_not_replan(graph) -> Check:
    c = Check("A not-approved decision with NO feedback proceeds normally, does not trigger a replan")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    result = graph.invoke(Command(resume={"approved": False, "feedback": ""}), config=config)
    c.require(result.get("replan_requested") in (False, None), "expected replan_requested to be falsy with no feedback")
    c.require(interrupt_type(result) == "email_approval", f"expected to proceed to email_approval, got {interrupt_type(result)}")
    return c


def check_cancel_ends_the_run(graph) -> Check:
    c = Check("Cancelling at the review gate ends the run — no calendar write, no email gate")
    config = new_config()
    graph.invoke(BASE_STATE, config=config)
    result = graph.invoke(Command(resume={"approved": False, "cancelled": True}), config=config)
    c.require(result.get("cancelled") is True, f"expected cancelled=True, got {result.get('cancelled')}")
    c.require(not result.get("__interrupt__"), f"expected the graph to end (no further interrupt), got {interrupt_type(result)}")
    c.require(not result.get("calendar_result"), f"expected no calendar write on cancel, got {result.get('calendar_result')}")
    c.require(not result.get("email_result"), f"expected no email step on cancel, got {result.get('email_result')}")
    return c


def check_replan_updates_state_and_reruns_agents(graph) -> Check:
    c = Check("Revising WITH feedback updates state (dates/budget) and re-runs Logistics+Experience")
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


def check_date_revision_recomputes_deadlines(graph) -> Check:
    c = Check("Revising the trip dates recomputes the deadline timeline and reports what changed")
    config = new_config()
    result = graph.invoke(BASE_STATE, config=config)  # China->Japan tourism: needs a visa, so deadlines run
    dl_before = [d["date"] for d in (result.get("deadlines") or [])]
    if not dl_before:
        c.require(False, "setup: expected the base run to produce at least one deadline")
        return c
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "move the whole trip to start 2027-03-01, keep it the same length"}),
        config=config,
    )
    dl_after = [d["date"] for d in (result.get("deadlines") or [])]
    c.require(result.get("start_date") == "2027-03-01", f"expected start_date moved to 2027-03-01, got {result.get('start_date')}")
    c.require(result.get("dates_changed") is True, f"expected dates_changed=True, got {result.get('dates_changed')}")
    c.require(bool(dl_after) and dl_after != dl_before, f"expected deadline dates to shift with the trip: {dl_before} -> {dl_after}")
    last_rev = (result.get("__interrupt__") or [None])[0]
    last_rev = last_rev.value.get("last_revision") if last_rev else None
    c.require(bool(last_rev and last_rev.get("changes")), f"expected a last_revision summary of what changed, got {last_rev}")
    return c


def check_destination_revision_reroutes_everything(graph) -> Check:
    c = Check("Revising the destination re-points flights/hotels/itinerary and KEEPS the other cities on a swap")
    config = new_config()
    graph.invoke({**BASE_STATE, "destination_city": "Tokyo", "destination_cities": ["Tokyo", "Osaka"]}, config=config)
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "change the destination to Kyoto from Tokyo"}),
        config=config,
    )
    # swap Tokyo -> Kyoto, Osaka untouched
    c.require(result.get("destination_cities") == ["Kyoto", "Osaka"], f"expected ['Kyoto', 'Osaka'] (Osaka kept), got {result.get('destination_cities')}")
    c.require(result.get("destination_city") == "Kyoto", f"expected primary 'Kyoto', got {result.get('destination_city')}")
    logi = result.get("logistics") or {}
    c.require(set((logi.get("accommodation") or {}).keys()) == {"Kyoto", "Osaka"}, f"expected hotels for Kyoto AND Osaka, got {list((logi.get('accommodation') or {}).keys())}")
    c.require("NRT" not in ((logi.get("flight_offers") or [{}])[0]).get("search_link", ""), "expected flight links to no longer point to Tokyo Narita")
    last_rev = (result.get("__interrupt__") or [None])[0]
    last_rev = last_rev.value.get("last_revision") if last_rev else None
    c.require(bool(last_rev and any("Destination" in ch for ch in last_rev.get("changes", []))), f"expected a 'Destination: ... → ...' change note, got {last_rev}")
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


# ---------------------------------------------------------------------------------------------
# Robustness / breaking-case checks. The bar here is GRACEFUL BEHAVIOR — the graph must not crash,
# must degrade sensibly, and should flag the problem — NOT that it produces an accurate visa
# answer for a made-up country. Each is a full graph.invoke against live APIs, so this section
# roughly doubles the suite's runtime.
# ---------------------------------------------------------------------------------------------
def _reached_a_stopping_point(result: dict) -> bool:
    """The graph reached a clean end state — an approval gate, a final itinerary, or a validation
    stop — rather than crashing, hanging, or returning nothing."""
    return bool(result) and bool(
        result.get("__interrupt__") or result.get("itinerary") or result.get("input_error")
    )


def _no_agents_ran(result: dict) -> bool:
    agents = {t["agent"] for t in result.get("agent_trace", [])}
    return not (agents - {"validator"})


def _adv_state(**overrides) -> dict:
    state = {k: v for k, v in BASE_STATE.items()}
    state["preferences"] = dict(BASE_STATE["preferences"])
    state.update(overrides)
    return state


def check_typos_are_tolerated(graph) -> Check:
    c = Check("Breaking case: typos ('Chinna' / 'Tokoy' in Japan / 'New Yrok') are understood — planning proceeds, no false validation stop")
    state = _adv_state(nationality="Chinna", destination_city="Tokoy", destination_cities=["Tokoy"], origin="New Yrok")
    result = graph.invoke(state, config=new_config())
    c.require(not result.get("input_error"), f"expected typos NOT to trip the validation gate, got input_error={result.get('input_error')!r}")
    c.require(_reached_a_stopping_point(result), "expected an approval gate or an itinerary despite the typos")
    c.require(
        not any(t["agent"] == "orchestrator" and t["status"] == "failed" for t in result.get("agent_trace", [])),
        "expected synthesis not to hard-fail on typo'd input",
    )
    return c


def check_unknown_nationality(graph) -> Check:
    c = Check("Breaking case: unknown nationality ('Wakanda') — eligibility degrades to refuse_and_verify, graph continues")
    result = graph.invoke(_adv_state(nationality="Wakanda"), config=new_config())
    c.require(_reached_a_stopping_point(result), "expected the graph to complete despite an unknown nationality")
    c.require(not result.get("input_error"), "nationality is not the validator's concern — it should not stop the run")
    elig = result.get("eligibility")
    c.require(
        elig is None or elig.get("answer_type") == "refuse_and_verify",
        f"expected refuse_and_verify (or a clean eligibility failure), got {elig and elig.get('answer_type')}",
    )
    return c


def check_city_country_mismatch_is_caught_early(graph) -> Check:
    c = Check("Breaking case: city ('Toronto') not in the destination country ('Japan') — validation gate stops the run before any agent, with a fix-it message")
    state = _adv_state(origin="New York", destination_country="Japan", destination_city="Toronto", destination_cities=["Toronto"])
    result = graph.invoke(state, config=new_config())
    err = (result.get("input_error") or "").lower()
    c.require(bool(err), "expected input_error to be set for a city/country mismatch")
    c.require("toronto" in err or "canada" in err, f"expected the message to name the mismatch, got {result.get('input_error')!r}")
    c.require(not result.get("__interrupt__") and not result.get("itinerary"), "expected the graph to stop, not produce a plan")
    c.require(result.get("eligibility") is None and result.get("logistics") is None, "expected NO agent to have run")
    c.require(_no_agents_ran(result), f"expected only the validator in the trace, got {[t['agent'] for t in result.get('agent_trace', [])]}")
    return c


def check_gibberish_city_is_caught_early(graph) -> Check:
    c = Check("Breaking case: unrecognizable city ('xqzzy 42') — validation gate stops the run with a fix-it message")
    state = _adv_state(destination_city="xqzzy 42", destination_cities=["xqzzy 42"])
    result = graph.invoke(state, config=new_config())
    c.require(bool(result.get("input_error")), f"expected input_error for an unrecognizable city, got {result.get('input_error')!r}")
    c.require(not result.get("__interrupt__") and not result.get("itinerary"), "expected the graph to stop, not produce a plan")
    return c


# ---------------------------------------------------------------------------------------------
# Eligibility: reconcile the visa corpus against live search, and tag sources.
# ---------------------------------------------------------------------------------------------
def check_eligibility_reports_tagged_sources(graph) -> Check:
    c = Check("Eligibility returns a tagged `sources` list (corpus|live) and a primary_source")
    result = graph.invoke(BASE_STATE, config=new_config())
    elig = result.get("eligibility") or {}
    srcs = elig.get("sources")
    c.require(isinstance(srcs, list) and len(srcs) > 0, f"expected a non-empty sources list, got {srcs!r}")
    c.require(all(s.get("type") in ("corpus", "live") for s in srcs), f"every source must be tagged corpus|live: {srcs}")
    c.require(elig.get("primary_source") in ("corpus", "live", "both"), f"expected a primary_source, got {elig.get('primary_source')}")
    return c


def check_official_live_source_can_override_corpus(graph) -> Check:
    # Integration-level and LLM-dependent: relies on the reconciliation LLM judging gov.uk as an
    # official source that contradicts the corpus. If the corpus already agrees with "ETA required"
    # this asserts 'both' instead.
    c = Check("Eligibility: an official live source contradicting the corpus drives the answer (primary_source live/both)")
    import voyagent.agents.eligibility as elig_mod

    def fake_search(query, max_results=5):
        return [{
            "url": "https://www.gov.uk/eta",
            "title": "Get an electronic travel authorisation (ETA) to visit the UK - GOV.UK",
            "snippets": [
                "As of 2026, citizens of many visa-exempt countries now need an ETA to visit the UK for tourism.",
                "You must apply for and receive an ETA before you travel to the UK.",
            ],
        }]

    real = elig_mod.you_search
    elig_mod.you_search = fake_search
    try:
        state = _adv_state(nationality="Canada", destination_country="UK", destination_city="London", destination_cities=["London"])
        result = graph.invoke(state, config=new_config())
    finally:
        elig_mod.you_search = real
    elig = result.get("eligibility") or {}
    c.require(elig.get("primary_source") in ("live", "both"), f"expected primary_source live/both, got {elig.get('primary_source')}")
    c.require(
        any(s.get("type") == "live" and "gov.uk" in s.get("url", "") for s in elig.get("sources", [])),
        f"expected the live gov.uk source tagged in sources, got {elig.get('sources')}",
    )
    return c


def check_live_search_outage_falls_back_to_corpus(graph) -> Check:
    c = Check("Eligibility: a live-search outage falls back to the corpus answer, sources still returned")
    import voyagent.agents.eligibility as elig_mod

    real = elig_mod.you_search
    elig_mod.you_search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated search outage"))
    try:
        result = graph.invoke(BASE_STATE, config=new_config())
    finally:
        elig_mod.you_search = real
    elig = result.get("eligibility") or {}
    c.require(result.get("eligibility") is not None, "expected eligibility to survive a live-search outage")
    c.require(str(elig.get("cross_check_status", "")).startswith("failed"), f"expected cross_check_status 'failed: ...', got {elig.get('cross_check_status')}")
    c.require(elig.get("primary_source") == "corpus", "expected fallback to the corpus answer")
    c.require(any(s.get("type") == "corpus" for s in elig.get("sources", [])), "expected corpus sources still listed")
    return c


CORE_CHECKS = [
    check_happy_path,
    check_tool_failure_recovery,
    check_structured_output_failure_degrades_not_crashes,
    check_two_independent_approval_gates,
    check_email_rejection_does_not_send,
    check_rejection_without_feedback_does_not_replan,
    check_cancel_ends_the_run,
    check_replan_updates_state_and_reruns_agents,
    check_date_revision_recomputes_deadlines,
    check_destination_revision_reroutes_everything,
    check_replan_cap_terminates,
]

ROBUSTNESS_CHECKS = [
    check_typos_are_tolerated,
    check_unknown_nationality,
    check_city_country_mismatch_is_caught_early,
    check_gibberish_city_is_caught_early,
    check_eligibility_reports_tagged_sources,
    check_official_live_source_can_override_corpus,
    check_live_search_outage_falls_back_to_corpus,
]


def _render_report(core, robustness, core_total: int, rob_total: int) -> str:
    done = core + robustness
    passed = sum(r.passed for r in done)
    lines = [
        "# Voyagent Agentic-Behavior Eval\n",
        f"**{passed}/{len(done)} checks passed** "
        + (f"(running — {len(done)}/{core_total + rob_total} complete)\n" if len(done) < core_total + rob_total else "\n"),
        "## Core — control flow, state, tool failure, HITL\n",
    ]
    for r in core:
        lines.append(f"- {'✅' if r.passed else '❌'} {r.name}" + (f" — {r.detail}" if not r.passed else ""))
    if len(core) < core_total:
        lines.append(f"- ⏳ … {core_total - len(core)} core check(s) not yet run")
    lines.append("\n## Robustness — breaking cases + corpus/live reconciliation\n")
    for r in robustness:
        lines.append(f"- {'✅' if r.passed else '❌'} {r.name}" + (f" — {r.detail}" if not r.passed else ""))
    if len(robustness) < rob_total:
        lines.append(f"- ⏳ … {rob_total - len(robustness)} robustness check(s) not yet run")
    return "\n".join(lines) + "\n"


def run(sections: tuple = ("core", "robustness")) -> None:
    # NEVER send real email from the eval — SMTP may be configured in .env. Stub the sender for the
    # whole suite so an "approve the email gate" check exercises the graph path without delivering
    # anything (a bad recipient would otherwise bounce into the real inbox on every run).
    import time

    import voyagent.graph as graph_mod
    graph_mod.send_itinerary_email = lambda to_address, subject, body: {
        "status": "sent", "to": to_address, "message": "(eval stub — not actually sent)",
    }

    graph = build_graph()
    plan = []
    if "core" in sections:
        plan += [("core", fn) for fn in CORE_CHECKS]
    if "robustness" in sections:
        plan += [("robustness", fn) for fn in ROBUSTNESS_CHECKS]
    core_total = sum(1 for s, _ in plan if s == "core")
    rob_total = sum(1 for s, _ in plan if s == "robustness")

    core, robustness = [], []
    for section, fn in plan:
        print(f"[{len(core) + len(robustness) + 1}/{len(plan)}] {fn.__name__} ...", flush=True)
        t0 = time.monotonic()
        try:
            c = fn(graph)
        except Exception as e:  # noqa: BLE001 - a check itself blowing up is still a result to report
            c = Check(fn.__name__)
            c.passed, c.detail = False, f"Check raised an exception: {e}"
        dt = time.monotonic() - t0
        print(f"    {'PASS' if c.passed else 'FAIL: ' + c.detail}  ({dt:.0f}s)", flush=True)
        (core if section == "core" else robustness).append(c)
        # persist a partial report after EVERY check so progress is always inspectable
        (EVAL_DIR / "eval_report.md").write_text(_render_report(core, robustness, core_total, rob_total))

    report = _render_report(core, robustness, core_total, rob_total)
    print("\n" + report)
    (EVAL_DIR / "eval_report.md").write_text(report)
    if sum(r.passed for r in core + robustness) < len(plan):
        sys.exit(1)


if __name__ == "__main__":
    args = {a.lower() for a in sys.argv[1:]}
    secs = tuple(s for s in ("core", "robustness") if s in args) or ("core", "robustness")
    run(secs)
