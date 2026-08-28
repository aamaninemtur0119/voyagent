"""The orchestrator: a LangGraph StateGraph coordinating three specialized agents (Eligibility,
Logistics, Experience) with real state passed between them, tool-failure recovery (retry once,
then continue without that piece and say so — never a hard stop), and a human-in-the-loop
approval gate (via interrupt()) around the one real write action: writing deadlines to Google
Calendar. Reads (all three agents) are autonomous; the write is not.
"""

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


def route_after_eligibility(state: TripState) -> str:
    elig = state.get("eligibility")
    if elig and elig["answer_type"] in NEEDS_DEADLINE_CHECK and state.get("start_date"):
        return "deadlines"
    return "logistics"


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
# Logistics Agent node
# ---------------------------------------------------------------------------
def logistics_node(state: TripState) -> dict:
    prefs = state.get("preferences") or {}
    try:
        result = _with_retry(
            logistics.run,
            state["origin"], state["destination_city"], state["destination_country"],
            state.get("start_date"), state.get("end_date"),
            prefs.get("budget_level", "Any"),
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


# ---------------------------------------------------------------------------
# Experience Agent node
# ---------------------------------------------------------------------------
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
            "A complete, well-organized trip briefing in markdown combining all available agent "
            "results. If any section is missing/None, explicitly say that piece wasn't available "
            "(a tool failed after retrying) rather than silently omitting it."
        )
    )


def synthesize_node(state: TripState) -> dict:
    prompt = (
        "Combine these trip-planning results into one clear, organized markdown briefing for the "
        "traveler. Use ONLY what's actually present below — if a section is null/missing, say so "
        "explicitly (e.g. 'Accommodation info wasn't available this time due to a tool error') "
        "rather than pretending it was never asked for.\n\n"
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
# Human-in-the-loop: the ONLY write action (calendar) requires explicit approval.
# ---------------------------------------------------------------------------
def human_approval_node(state: TripState) -> dict:
    deadlines = state.get("deadlines") or []
    if not deadlines:
        return {"calendar_approved": False}
    decision = interrupt({
        "type": "calendar_write_approval",
        "message": f"{len(deadlines)} deadline(s) identified. Write these to Google Calendar?",
        "deadlines": deadlines,
    })
    return {"calendar_approved": bool(decision.get("approved"))}


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


def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("deadlines", deadlines_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("experience", experience_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "eligibility")
    graph.add_conditional_edges("eligibility", route_after_eligibility, {"deadlines": "deadlines", "logistics": "logistics"})
    graph.add_edge("deadlines", "logistics")
    graph.add_edge("logistics", "experience")
    graph.add_edge("experience", "synthesize")
    graph.add_edge("synthesize", "human_approval")
    graph.add_edge("human_approval", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())
