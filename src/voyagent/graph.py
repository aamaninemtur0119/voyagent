"""The orchestrator: a LangGraph StateGraph coordinating three specialized agents (Eligibility,
Logistics, Experience) with real state passed between them, tool-failure recovery (retry once,
then continue without that piece and say so — never a hard stop), two independently human-approved
write actions (Google Calendar, and saving the itinerary to a file), and an adaptive replanning
loop: rejecting the calendar approval WITH feedback loops back to re-run Logistics/Experience with
updated dates/budget rather than just stopping, capped at MAX_REPLANS to guarantee termination.

Reads (all three agents) are autonomous; both write actions are not — each sits behind its own
interrupt(), independently approved.
"""

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from voyagent.agents import eligibility, experience, logistics
from voyagent.config import settings
from voyagent.state import TripState
from voyagent.tools.calendar_actions import Deadline, extract_deadlines, write_to_calendar

_llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)

MAX_REPLANS = 2
EXPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "exports"


def _with_retry(fn, *args, attempts: int = 2, **kwargs):
    """Retry once on failure, then let the exception surface to the caller (the node), which is
    responsible for catching it and turning it into a graceful, visible degradation rather than a
    crashed graph run."""
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is the generic retry boundary
            last_exc = e
    raise last_exc


# ---------------------------------------------------------------------------
# Eligibility Agent node
# ---------------------------------------------------------------------------
def eligibility_node(state: TripState) -> dict:
    try:
        result = _with_retry(
            eligibility.run, state["nationality"], state["destination_country"], state["purpose"], state["duration"],
        )
    except Exception as e:
        return {
            "eligibility": None,
            "agent_trace": [{"agent": "eligibility", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Eligibility Agent failed after retry: {e}"],
        }
    return {
        "eligibility": result,
        "agent_trace": [{"agent": "eligibility", "status": "done", "detail": result["answer_type"]}],
    }


NEEDS_DEADLINE_CHECK = {
    "visa_required", "different_visa_category_required", "visa_free_waiver_with_ETA", "visa_free_waiver_with_ETIAS",
}


def route_after_eligibility(state: TripState):
    elig = state.get("eligibility")
    if elig and elig["answer_type"] in NEEDS_DEADLINE_CHECK and state.get("start_date"):
        return "deadlines"
    return ["logistics", "experience"]  # no deadline check needed — fan out straight to both


# ---------------------------------------------------------------------------
# Deadline extraction (conditional — only when eligibility found something time-sensitive)
# ---------------------------------------------------------------------------
def deadlines_node(state: TripState) -> dict:
    elig = state.get("eligibility") or {}
    chunks = elig.get("retrieved_chunks", [])
    situation = elig.get("situation", {})
    try:
        extracted = _with_retry(extract_deadlines, chunks, situation, state.get("start_date"))
        deadlines = [d.model_dump() for d in extracted.deadlines]
    except Exception as e:
        return {
            "deadlines": None,
            "agent_trace": [{"agent": "deadlines", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Deadline extraction failed after retry: {e}"],
        }
    return {
        "deadlines": deadlines,
        "agent_trace": [{"agent": "deadlines", "status": "done", "detail": f"{len(deadlines)} deadline(s) identified"}],
    }


# ---------------------------------------------------------------------------
# Logistics + Experience run in PARALLEL — they don't depend on each other, only on eligibility
# having (maybe) run first. Both fan into synthesize, which LangGraph automatically waits on.
# ---------------------------------------------------------------------------
def logistics_node(state: TripState) -> dict:
    prefs = state.get("preferences") or {}
    try:
        result = _with_retry(
            logistics.run,
            state["origin"], state["destination_city"], state["destination_country"],
            state.get("start_date"), state.get("end_date"),
            prefs.get("budget_level", "Any"), prefs,
        )
    except Exception as e:
        return {
            "logistics": None,
            "agent_trace": [{"agent": "logistics", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Logistics Agent failed after retry: {e}"],
        }
    return {
        "logistics": result,
        "agent_trace": [{"agent": "logistics", "status": "done", "detail": f"{len(result['accommodation'])} hotel option(s) found"}],
    }


def experience_node(state: TripState) -> dict:
    prefs = state.get("preferences") or {}
    try:
        result = _with_retry(experience.run, state["destination_city"], state["destination_country"], prefs)
    except Exception as e:
        return {
            "experience": None,
            "agent_trace": [{"agent": "experience", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Experience Agent failed after retry: {e}"],
        }
    return {
        "experience": result,
        "agent_trace": [{"agent": "experience", "status": "done", "detail": "Restaurants/places/activities curated"}],
    }


# ---------------------------------------------------------------------------
# Synthesis — combine whatever succeeded into one briefing, naming what didn't
# ---------------------------------------------------------------------------
class Itinerary(BaseModel):
    briefing: str = Field(
        description=(
            "A narrative trip briefing in markdown: visa/entry summary, the deadline timeline (if "
            "any), a one-line flight-search status note, and — if applicable — a multi-destination "
            "recommendation. Do NOT itemize every hotel/restaurant/place/activity/flight option "
            "with its own link and price here — those appear directly below this briefing as their "
            "own section in the app, so just don't mention them at all rather than describing "
            "where they are. Never use internal/technical terms like 'logistics data', 'structured "
            "cards', 'the state', or 'provided separately' — write only what a traveler would "
            "actually want to read, in plain language. For flights specifically: state in one "
            "plain sentence whether a live price was found or not (e.g. 'A flight search wasn't "
            "available this time, but search links are below' or 'A recommended flight was found "
            "below') — do not describe HOW or WHERE it's shown beyond that. If a section is null/"
            "missing, say so in plain language (e.g. 'Accommodation info wasn't available this "
            "time due to a tool error') rather than pretending it was never asked for."
        )
    )


MULTI_CITY_MIN_DAYS = 10


def _multi_city_note_instruction(state: TripState) -> str:
    duration_text = state.get("duration", "")
    single_city = len(state.get("destination_cities") or []) <= 1
    purpose_is_tourism = (state.get("purpose") or "").strip().lower() == "tourism"
    # Cheap heuristic, not a parsed date range - just enough to catch "14 days" / "18 days" etc.
    long_trip = any(str(n) in duration_text for n in range(MULTI_CITY_MIN_DAYS, 60))
    if single_city and purpose_is_tourism and long_trip:
        return (
            "\n\nThe traveler only listed one destination city for a tourism trip of "
            f"'{duration_text}'. At the END of the briefing, under a '💡 Recommendation' heading, "
            "suggest — clearly framed as a suggestion, not a fact — that they consider splitting "
            "their time across more than one city within the destination country instead of "
            "staying in just one place that whole time. You may name well-known specific cities "
            "if you're confident they're genuinely popular multi-stop destinations for that "
            "country; otherwise keep it general rather than guessing at a specific itinerary."
        )
    return ""


def synthesize_node(state: TripState) -> dict:
    prompt = (
        "Combine these trip-planning results into a narrative markdown briefing for the traveler "
        "(see the format instructions on the output field — do not itemize individual hotels/"
        "restaurants/places/activities/flights here, those are shown separately). Use ONLY what's "
        "actually present below — if a section is null/missing, say so explicitly rather than "
        "pretending it was never asked for."
        f"{_multi_city_note_instruction(state)}\n\n"
        f"Eligibility (visa/entry): {state.get('eligibility')}\n\n"
        f"Deadlines: {state.get('deadlines')}\n\n"
        f"Logistics (flights/accommodation): {state.get('logistics')}\n\n"
        f"Experience (restaurants/places/activities): {state.get('experience')}\n\n"
        f"Any errors encountered: {state.get('errors')}"
    )
    result = _llm.with_structured_output(Itinerary).invoke(prompt)
    return {
        "itinerary": result.briefing,
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "Itinerary synthesized"}],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop #1: calendar write, OR reject-with-feedback to trigger a replan.
# ---------------------------------------------------------------------------
class ReplanUpdate(BaseModel):
    start_date: str | None = Field(None, description="New ISO date (YYYY-MM-DD) if the feedback implies changing the start date, else null.")
    end_date: str | None = Field(None, description="New ISO end date if implied, else null.")
    budget_level: str | None = Field(None, description="New budget level (Any, $, $$, $$$, $$$$) if implied, else null.")
    summary: str = Field(description="One short line describing what's changing and why, for the trace log.")


def human_approval_node(state: TripState) -> dict:
    deadlines = state.get("deadlines") or []
    replans_so_far = state.get("replan_count", 0)
    decision = interrupt({
        "type": "calendar_write_approval",
        "message": (
            f"{len(deadlines)} deadline(s) identified. Approve writing these to Google Calendar, "
            "or describe a change (dates, budget) and I'll replan instead."
        ) if deadlines else "Review the itinerary. Approve to continue, or describe a change and I'll replan.",
        "deadlines": deadlines,
        "itinerary": state.get("itinerary"),
        "replans_so_far": replans_so_far,
        "replans_remaining": max(0, MAX_REPLANS - replans_so_far),
    })
    approved = bool(decision.get("approved"))
    feedback = (decision.get("feedback") or "").strip()

    if approved or not feedback or replans_so_far >= MAX_REPLANS:
        return {"calendar_approved": approved, "replan_requested": False}

    update = _llm.with_structured_output(ReplanUpdate).invoke(
        f'The traveler rejected a trip plan draft and said: "{feedback}". Current start_date='
        f"{state.get('start_date')}, end_date={state.get('end_date')}, budget_level="
        f"{(state.get('preferences') or {}).get('budget_level')}. Extract exactly what should "
        "change — only set a field if the feedback actually implies changing it; leave the rest null."
    )
    updates: dict = {
        "calendar_approved": False,
        "replan_requested": True,
        "replan_count": replans_so_far + 1,
        "agent_trace": [{"agent": "orchestrator", "status": "retrying", "detail": f"Replanning: {update.summary}"}],
    }
    if update.start_date:
        updates["start_date"] = update.start_date
    if update.end_date:
        updates["end_date"] = update.end_date
    if update.budget_level:
        prefs = dict(state.get("preferences") or {})
        prefs["budget_level"] = update.budget_level
        updates["preferences"] = prefs
    return updates


def route_after_approval(state: TripState):
    if state.get("replan_requested"):
        return ["logistics", "experience"]  # real cycle: re-run the affected agents, not a restart
    return "finalize"


def finalize_node(state: TripState) -> dict:
    if not state.get("calendar_approved") or not state.get("deadlines"):
        return {
            "calendar_result": {"status": "skipped"},
            "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "No calendar write performed"}],
        }
    deadline_objs = [Deadline(**d) for d in state["deadlines"]]
    result = write_to_calendar(deadline_objs)
    return {
        "calendar_result": result,
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": f"Calendar: {result['status']}"}],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop #2: a second, independent write action — saving the itinerary to a file.
# Shows the approval pattern generalizes rather than being a one-off built around Calendar.
# ---------------------------------------------------------------------------
def export_approval_node(state: TripState) -> dict:
    decision = interrupt({
        "type": "export_approval",
        "message": "Save a copy of this itinerary as a file?",
        "preview": (state.get("itinerary") or "")[:300],
    })
    return {"export_approved": bool(decision.get("approved"))}


def finalize_export_node(state: TripState) -> dict:
    if not state.get("export_approved"):
        return {
            "export_result": {"status": "skipped"},
            "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "Itinerary not saved to file"}],
        }
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_city = "".join(c if c.isalnum() else "_" for c in state.get("destination_city", "trip"))
    path = EXPORTS_DIR / f"{safe_city}_itinerary.md"
    path.write_text(state.get("itinerary") or "")
    return {
        "export_result": {"status": "written", "path": str(path)},
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": f"Itinerary saved to {path.name}"}],
    }


def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("deadlines", deadlines_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("experience", experience_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("export_approval", export_approval_node)
    graph.add_node("finalize_export", finalize_export_node)

    graph.add_edge(START, "eligibility")
    graph.add_conditional_edges("eligibility", route_after_eligibility)
    graph.add_edge("deadlines", "logistics")
    graph.add_edge("deadlines", "experience")
    graph.add_edge("logistics", "synthesize")  # fan-in: synthesize waits for both branches
    graph.add_edge("experience", "synthesize")
    graph.add_edge("synthesize", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_approval)  # may cycle back
    graph.add_edge("finalize", "export_approval")
    graph.add_edge("export_approval", "finalize_export")
    graph.add_edge("finalize_export", END)

    return graph.compile(checkpointer=MemorySaver())
