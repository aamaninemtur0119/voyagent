"""The orchestrator: a LangGraph StateGraph coordinating three specialized agents (Eligibility,
Logistics, Experience) with real state passed between them, tool-failure recovery (retry once,
then continue without that piece and say so — never a hard stop), two independently human-approved
write actions (writing deadlines to Google Calendar, and emailing the finished itinerary to the
traveler), and an adaptive replanning loop: rejecting the calendar approval WITH feedback loops
back to re-run Logistics/Experience with updated dates/budget rather than just stopping, capped at
MAX_REPLANS to guarantee termination.

Reads (all three agents) are autonomous; both write actions are not — each sits behind its own
interrupt(), independently approved, and each degrades gracefully (retry once, then record the
failure and finish) if the underlying send/write fails.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from pack_your_bags.agents import eligibility, experience, logistics
from pack_your_bags.llm import structured
from pack_your_bags.state import TripState
from pack_your_bags.tools.calendar_actions import Deadline, extract_deadlines, write_to_calendar
from pack_your_bags.tools.email_actions import send_itinerary_email

MAX_REPLANS = 2


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
# Validation gate — the first node. Catches an incoherent request (a destination city that isn't
# in the destination country, an unrecognizable city) BEFORE any agent runs, so the traveler gets
# "fix the city/country" instead of a plan that checks a Japan visa while pricing flights to
# Toronto. Fails open: if the check itself errors, planning proceeds.
# ---------------------------------------------------------------------------
class SituationCheck(BaseModel):
    coherent: bool = Field(
        description=(
            "True if every listed destination city plausibly belongs to the destination country "
            "(minor misspellings / alternate spellings are fine). False if any city is clearly in "
            "a different country, or is not identifiable as a real place in that country."
        )
    )
    issue: str = Field(
        default="",
        description="If not coherent: ONE plain sentence naming the mismatch, e.g. 'Toronto is in Canada, not Japan.' Empty if coherent.",
    )
    fix: str = Field(
        default="",
        description="If not coherent: a short suggested fix, e.g. 'Change the country to Canada, or pick a city in Japan such as Tokyo or Osaka.' Empty if coherent.",
    )


def _check_situation(country: str, cities: list[str]) -> SituationCheck:
    """Shared by the START validation gate and a mid-run revision that changes the destination."""
    return structured(
        SituationCheck,
        f"Destination country: {country}\n"
        f"Destination cities the traveler entered: {cities}\n\n"
        "Does each city plausibly belong to that country? Minor misspellings are fine "
        "(e.g. 'Tokoy' -> Tokyo). Flag it only if a city is clearly in a different country or "
        "isn't a recognizable place in this one.",
        default=SituationCheck(coherent=True),  # fail open — a validator hiccup must not block planning
    )


def validate_node(state: TripState) -> dict:
    cities = state.get("destination_cities") or [state.get("destination_city", "")]
    check = _check_situation(state.get("destination_country", ""), cities)
    if check.coherent:
        return {"agent_trace": [{"agent": "validator", "status": "done", "detail": "destination city/country check passed"}]}
    message = " ".join(p for p in (check.issue.strip(), check.fix.strip()) if p) or (
        "The destination city and country don't match. Please correct them and try again."
    )
    return {
        "input_error": message,
        "agent_trace": [{"agent": "validator", "status": "failed", "detail": check.issue or "city/country mismatch"}],
    }


def route_after_validate(state: TripState):
    return END if state.get("input_error") else "eligibility"


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
    status = result.get("cross_check_status", "skipped")
    primary = result.get("primary_source", "corpus")
    detail = result["answer_type"]
    if status == "diverged":
        detail += f" (⚠️ corpus/live differ — using {primary})"
    elif status == "reconciled":
        detail += f" (corpus + live checked, {len(result.get('sources', []))} source(s))"
    elif status.startswith("failed") or status == "not_configured":
        detail += " (corpus only — live cross-check unavailable)"
    return {
        "eligibility": result,
        "agent_trace": [{"agent": "eligibility", "status": "done", "detail": detail}],
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
    cities = state.get("destination_cities") or [state["destination_city"]]
    try:
        result = _with_retry(
            logistics.run,
            state["origin"], cities, state["destination_country"],
            state.get("start_date"), state.get("end_date"),
            prefs.get("budget_level", "Any"), prefs,
        )
    except Exception as e:
        return {
            "logistics": None,
            "agent_trace": [{"agent": "logistics", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Logistics Agent failed after retry: {e}"],
        }
    n_hotels = sum(len(v) for v in (result.get("accommodation") or {}).values())
    return {
        "logistics": result,
        "agent_trace": [{"agent": "logistics", "status": "done", "detail": f"{n_hotels} hotel option(s) across {len(cities)} city(ies)"}],
    }


def experience_node(state: TripState) -> dict:
    prefs = state.get("preferences") or {}
    cities = state.get("destination_cities") or [state["destination_city"]]
    try:
        result = _with_retry(experience.run, cities, state["destination_country"], prefs)
    except Exception as e:
        return {
            "experience": None,
            "agent_trace": [{"agent": "experience", "status": "failed", "detail": f"Gave up after retrying: {e}"}],
            "errors": [f"Experience Agent failed after retry: {e}"],
        }
    return {
        "experience": result,
        "agent_trace": [{"agent": "experience", "status": "done", "detail": f"Restaurants/places/activities curated for {', '.join(cities)}"}],
    }


# ---------------------------------------------------------------------------
# Synthesis — combine whatever succeeded into one briefing, naming what didn't
# ---------------------------------------------------------------------------
class Itinerary(BaseModel):
    main_summary: str = Field(
        description=(
            "ONLY the visa/entry summary and the deadline timeline (if any). Close with ONE "
            "plain-language sentence noting that flight, hotel, restaurant, place, and activity "
            "options are shown below — do not itemize, describe, or summarize the content of any "
            "of those here (no 'restaurants emphasize vegetarian spots'-style recaps); that's what "
            "structured cards below this text are for, and what the recommendation field covers. "
            "Never say something is 'not rendered' or 'not available' if it actually exists below "
            "— say where it is. Never use internal/technical terms like 'logistics data', "
            "'structured cards', or 'the state'. If eligibility or deadlines themselves are "
            "genuinely null/missing (a tool failed), say so in plain language. If the eligibility "
            "data has a non-empty 'divergence_note' (the visa corpus and a live official source "
            "disagreed), state plainly which one the answer follows and why; otherwise don't "
            "mention sources at all — they're listed separately below."
        )
    )
    recommendation: str = Field(
        default="",  # optional: an empty string is valid when there's genuinely nothing useful to add
        # (e.g. a tool failed and there's little curated content to recommend from). Making this
        # required let the synthesis LLM crash the node by omitting it — see synthesize_node.
        description=(
            "Under a '💡 Recommendation' heading, plain language: (1) If the traveler listed only "
            "one destination city for a long tourism trip, suggest — clearly framed as a "
            "suggestion, not a fact — splitting time across more cities in the destination "
            "country; name specific well-known cities if you're confident they're genuinely "
            "popular pairings. (2) One or two genuinely distinctive notes about the SPECIFIC real "
            "places actually curated below (pick something notable from the actual restaurant/"
            "place/activity names given — not generic travel-guide filler). (3) A short suggested "
            "itinerary that names specific real restaurants/places/activities from what's curated "
            "below, organized by city (and loosely by day, if it fits naturally) for every city "
            "the traveler actually listed. If the traveler listed only one city, you may extend "
            "the itinerary with 1-2 more cities in the destination country using your own general "
            "knowledge — but clearly label that portion as a general suggestion, not verified "
            "data, since those cities weren't actually searched."
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
        "Build the two-part trip output described in the schema below from these results. Ground "
        "the recommendation's itinerary in the ACTUAL restaurant/place/activity names in the "
        "Experience data — never invent a name for a city that was actually searched."
        f"{_multi_city_note_instruction(state)}\n\n"
        f"Destination country: {state.get('destination_country')}\n\n"
        f"Destination cities the traveler listed: {state.get('destination_cities')}\n\n"
        f"Eligibility (visa/entry): {state.get('eligibility')}\n\n"
        f"Deadlines: {state.get('deadlines')}\n\n"
        f"Logistics (flights/accommodation) status: {state.get('logistics')}\n\n"
        f"Experience (restaurants/places/activities), keyed by city: {state.get('experience')}\n\n"
        f"Any errors encountered: {state.get('errors')}"
    )
    result = structured(Itinerary, prompt, default=None)  # retries + raw-JSON repair, then None
    if result is None:  # synthesis degrades like every other node — it must not hard-stop the graph
        answer_type = (state.get("eligibility") or {}).get("answer_type", "unavailable")
        return {
            "itinerary": (
                "A full written briefing couldn't be assembled this time "
                f"(visa/entry status: {answer_type}). The flight, hotel, and activity options that "
                "were found are shown below."
            ),
            "errors": ["Synthesis failed after retries, used a minimal briefing."],
            "agent_trace": [{"agent": "orchestrator", "status": "failed", "detail": "Synthesis degraded to a minimal briefing"}],
        }
    rec = (result.recommendation or "").strip()
    combined = result.main_summary.strip() + (f"\n\n---\n\n{rec}" if rec else "")
    return {
        "itinerary": combined,
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "Itinerary synthesized"}],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop #1: calendar write, OR reject-with-feedback to trigger a replan.
# ---------------------------------------------------------------------------
class ReplanUpdate(BaseModel):
    start_date: str | None = Field(None, description="New ISO date (YYYY-MM-DD) if the feedback implies changing the start date, else null.")
    end_date: str | None = Field(None, description="New ISO end date if implied, else null.")
    budget_level: str | None = Field(None, description="New budget level (Any, $, $$, $$$, $$$$) if implied, else null.")
    cities_add: list[str] = Field(
        default_factory=list,
        description="Cities to ADD to the trip's existing list ('also do Nara', or the new city in "
        "'Kyoto instead of Tokyo'). Empty if the feedback adds nowhere.",
    )
    cities_remove: list[str] = Field(
        default_factory=list,
        description="Cities to REMOVE from the trip's existing list. Use this for 'skip Osaka' or "
        "for a true replacement ('Kyoto INSTEAD OF Tokyo' -> remove Tokyo). Do NOT remove a city "
        "the feedback says to KEEP (e.g. 'swap Tokyo out as main stop but keep Tokyo and Osaka' -> "
        "remove nothing). Empty if nothing is removed.",
    )
    set_primary: str | None = Field(
        None,
        description="The city that should become the PRIMARY destination (first in the list — what "
        "flights are priced to). Use it for 'make Kyoto the main destination', 'Kyoto as the main "
        "stop', 'base it in Kyoto'. The city stays in the list; it just moves to the front. Null if "
        "the feedback doesn't change which city is primary.",
    )
    cities_replace_all: list[str] | None = Field(
        None,
        description="ONLY when the traveler clearly wants to discard the WHOLE current city list and "
        "go somewhere entirely different ('scrap it, just Kyoto now'). Null for ordinary adds, "
        "removes, swaps, or primary changes. Every city (here or in cities_add) must be in the SAME "
        "country — if the feedback implies a different country, set none of these and say so in summary.",
    )
    summary: str = Field(description="One short plain-language line describing what's changing and why — shown to the traveler on the next review screen.")


def human_approval_node(state: TripState) -> dict:
    deadlines = state.get("deadlines") or []
    replans_so_far = state.get("replan_count", 0)
    decision = interrupt({
        "type": "calendar_write_approval",
        "message": (
            f"{len(deadlines)} deadline(s) identified. Approve to add them to your calendar and "
            "continue to emailing your itinerary, ask for a revision, or cancel."
        ) if deadlines else "Review the plan. Approve to continue, ask for a revision, or cancel.",
        "deadlines": deadlines,
        "itinerary": state.get("itinerary"),
        "replans_so_far": replans_so_far,
        "replans_remaining": max(0, MAX_REPLANS - replans_so_far),
        "last_revision": state.get("last_revision"),  # what the previous revision actually changed
    })
    approved = bool(decision.get("approved"))
    cancelled = bool(decision.get("cancelled"))
    feedback = (decision.get("feedback") or "").strip()

    if cancelled:
        return {
            "calendar_approved": False,
            "replan_requested": False,
            "cancelled": True,
            "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "Cancelled by the traveler — nothing written, nothing sent"}],
        }

    if approved or not feedback or replans_so_far >= MAX_REPLANS:
        return {"calendar_approved": approved, "replan_requested": False}

    old_start = state.get("start_date")
    old_end = state.get("end_date")
    old_budget = (state.get("preferences") or {}).get("budget_level")
    country = state.get("destination_country", "")
    old_cities = state.get("destination_cities") or ([state["destination_city"]] if state.get("destination_city") else [])
    # If the feedback can't be parsed into a structured delta, degrade to "re-run unchanged and
    # say so" rather than crashing the revise.
    update = structured(
        ReplanUpdate,
        f'The traveler asked to revise a trip plan draft and said: "{feedback}". '
        f"Current start_date={old_start}, end_date={old_end}, budget_level={old_budget}, "
        f"destination_country={country}, destination_cities={old_cities}. Extract exactly what "
        "should change — only set a field if the feedback actually implies changing it; leave the "
        "rest null/empty. For the destination: 'X INSTEAD OF Y' -> cities_remove=[Y], cities_add=[X]. "
        "'make X the MAIN/primary destination, keeping the others' -> cities_add=[X] (if not already "
        "listed) + set_primary=X, and remove NOTHING. 'also add Z' -> cities_add=[Z]. 'skip W' -> "
        "cities_remove=[W]. Only use cities_replace_all to abandon the whole current list.",
        default=ReplanUpdate(summary="Couldn't interpret the requested change — re-ran the plan unchanged."),
    )

    dates_changed = bool(
        (update.start_date and update.start_date != old_start)
        or (update.end_date and update.end_date != old_end)
    )
    changes: list[str] = []
    if update.start_date and update.start_date != old_start:
        changes.append(f"Start date: {old_start} → {update.start_date}")
    if update.end_date and update.end_date != old_end:
        changes.append(f"End date: {old_end} → {update.end_date}")
    if update.budget_level and update.budget_level != old_budget:
        changes.append(f"Budget: {old_budget} → {update.budget_level}")
    if dates_changed and state.get("deadlines") is not None:
        changes.append("Visa deadline timeline recalculated from the new dates")

    # Destination change: apply add/remove/replace as OPS on the existing list (a swap keeps every
    # other city), then re-validate the result is in the same country before applying — so a revise
    # can't quietly turn the trip into a Frankenstein (Japan visa, flights to Paris) or silently
    # drop cities the traveler never mentioned.
    new_cities: list[str] | None = None
    removed = {c.strip().lower() for c in update.cities_remove if c.strip()}
    kept = [c for c in old_cities if c.lower() not in removed]
    added = [c.strip() for c in update.cities_add if c.strip() and c.strip().lower() not in {k.lower() for k in kept}]
    set_primary = (update.set_primary or "").strip()
    if update.cities_replace_all:
        proposed = [c.strip() for c in update.cities_replace_all if c.strip()]
    elif added or removed or set_primary:
        proposed = kept + added
        primary = set_primary
        if not primary and bool(old_cities) and old_cities[0].lower() in removed and added:
            primary = added[0]  # old primary was replaced — the new city leads
        if primary:
            proposed = [primary] + [c for c in proposed if c.lower() != primary.lower()]
    else:
        proposed = []
    if proposed and proposed != old_cities:
        if _check_situation(country, proposed).coherent:
            new_cities = proposed
            changes.append(f"Destination: {', '.join(old_cities) or '—'} → {', '.join(new_cities)}")
        else:
            changes.append(f"Destination change ignored — those cities aren't in {country or 'the destination country'}")

    if not changes:
        changes.append("No concrete change was detected — re-ran the search anyway")

    updates: dict = {
        "calendar_approved": False,
        "replan_requested": True,
        "replan_count": replans_so_far + 1,
        "dates_changed": dates_changed,
        "last_revision": {"summary": update.summary, "changes": changes},
        "agent_trace": [{"agent": "orchestrator", "status": "retrying", "detail": f"Revising: {update.summary}"}],
    }
    if update.start_date:
        updates["start_date"] = update.start_date
    if update.end_date:
        updates["end_date"] = update.end_date
    if update.budget_level:
        prefs = dict(state.get("preferences") or {})
        prefs["budget_level"] = update.budget_level
        updates["preferences"] = prefs
    if new_cities:
        updates["destination_cities"] = new_cities
        updates["destination_city"] = new_cities[0]
    return updates


def route_after_approval(state: TripState):
    if state.get("cancelled"):
        return END  # traveler cancelled at the review gate — no calendar write, no email
    if state.get("replan_requested"):
        # Dates moved AND there was a deadline timeline → re-run deadlines first (it recomputes the
        # timeline from the new start date using the chunks already in state), then its existing
        # edges fan out to logistics + experience → synthesize. Budget-only revisions skip it.
        if state.get("dates_changed") and state.get("deadlines") is not None:
            return "deadlines"
        return ["logistics", "experience"]  # real cycle: re-run the affected agents, not a restart
    return "finalize"


def finalize_node(state: TripState) -> dict:
    if not state.get("calendar_approved") or not state.get("deadlines"):
        return {
            "calendar_result": {"status": "skipped"},
            "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "No calendar write performed"}],
        }
    deadline_objs = [Deadline(**d) for d in state["deadlines"]]
    try:
        result = _with_retry(write_to_calendar, deadline_objs)
    except Exception as e:  # noqa: BLE001 - an approved write that fails degrades gracefully, like every other tool
        return {
            "calendar_result": {"status": "failed", "message": f"Calendar write failed after retrying: {e}"},
            "errors": [f"Calendar write failed after retry: {e}"],
            "agent_trace": [{"agent": "orchestrator", "status": "failed", "detail": f"Calendar write failed: {e}"}],
        }
    return {
        "calendar_result": result,
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": f"Calendar: {result['status']}"}],
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop #2: a second, independent write action — emailing the finished itinerary to
# the traveler. Separate interrupt, separate failure handling: shows the approval pattern
# generalizes rather than being a one-off built around Calendar.
# ---------------------------------------------------------------------------
def _itinerary_email_body(state: TripState) -> str:
    """Plain-text email body: the synthesized narrative, then the concrete options (flights,
    hotels, and per-city restaurants/places/activities) that otherwise only live in the UI cards,
    each with its booking/maps link so the email is self-contained."""
    parts: list[str] = [(state.get("itinerary") or "").strip(), ""]

    logistics = state.get("logistics") or {}
    offers = logistics.get("flight_offers") or []
    if offers:
        parts.append("\n" + "-" * 40 + "\nFLIGHTS")
        if logistics.get("flight_recommendation"):
            parts.append(f"Recommended: {logistics['flight_recommendation']}")
        for o in offers:
            parts.append(f"- {o['airline']} — {o['price_total']} {o['currency']} — {o['stops']} stop(s) — {o['duration']}")
            if o.get("search_link"):
                parts.append(f"  {o['search_link']}")

    accommodation = logistics.get("accommodation") or {}
    multi_stay = len(accommodation) > 1
    for city, hotels in accommodation.items():
        if hotels:
            heading = f"HOTELS — {city}" if multi_stay else "HOTELS"
            parts.append("\n" + "-" * 40 + f"\n{heading}")
            for h in hotels:
                bits = [h["name"]]
                if h.get("rating") is not None:
                    bits.append(f"{h['rating']}★ ({h.get('review_count', 0)})")
                if h.get("price_level"):
                    bits.append(h["price_level"])
                parts.append("- " + " — ".join(bits))
                if h.get("maps_url"):
                    parts.append(f"  {h['maps_url']}")

    experience = state.get("experience") or {}
    multi_city = len(experience) > 1
    for city, city_exp in experience.items():
        for label, key in [("RESTAURANTS", "restaurants"), ("PLACES TO VISIT", "places_to_visit"), ("ACTIVITIES", "activities")]:
            items = (city_exp or {}).get(key) or []
            if items:
                heading = f"{label} — {city}" if multi_city else label
                parts.append("\n" + "-" * 40 + f"\n{heading}")
                for item in items:
                    line = f"- {item['name']}"
                    if item.get("rating") is not None:
                        line += f" — {item['rating']}★ ({item.get('review_count', 0)})"
                    parts.append(line)
                    if item.get("maps_url"):
                        parts.append(f"  {item['maps_url']}")

    return "\n".join(parts).strip() + "\n"


def email_approval_node(state: TripState) -> dict:
    to_address = (state.get("traveler_email") or "").strip()
    if not to_address:
        return {
            "email_approved": False,
            "email_result": {"status": "skipped", "message": "No email address was given, so there is nothing to send."},
            "agent_trace": [{"agent": "orchestrator", "status": "skipped", "detail": "No email address given — itinerary not sent"}],
        }
    decision = interrupt({
        "type": "email_approval",
        "message": f"Send this itinerary to {to_address}?",
        "recipient": to_address,
        "preview": (state.get("itinerary") or "")[:300],
    })
    return {"email_approved": bool(decision.get("approved"))}


def route_after_email_approval(state: TripState):
    # No recipient -> email_approval_node already recorded a 'skipped' result and did not interrupt;
    # nothing left to do. Otherwise go run the send.
    return "send_email" if (state.get("traveler_email") or "").strip() else END


def send_email_node(state: TripState) -> dict:
    if not state.get("email_approved"):
        return {
            "email_result": {"status": "skipped", "message": "Not sent — you declined."},
            "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": "Itinerary not emailed (declined)"}],
        }
    to_address = (state.get("traveler_email") or "").strip()
    subject = f"Your trip plan — {state.get('destination_city', 'trip')}"
    try:
        result = _with_retry(send_itinerary_email, to_address, subject, _itinerary_email_body(state))
    except Exception as e:  # noqa: BLE001 - an approved send that fails degrades gracefully, like every other tool
        return {
            "email_result": {"status": "failed", "message": f"Email send failed after retrying: {e}"},
            "errors": [f"Itinerary email failed after retry: {e}"],
            "agent_trace": [{"agent": "orchestrator", "status": "failed", "detail": f"Email send failed: {e}"}],
        }
    return {
        "email_result": result,
        "agent_trace": [{"agent": "orchestrator", "status": "done", "detail": f"Email: {result['status']}"}],
    }


def build_graph():
    graph = StateGraph(TripState)
    graph.add_node("validate", validate_node)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("deadlines", deadlines_node)
    graph.add_node("logistics", logistics_node)
    graph.add_node("experience", experience_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("email_approval", email_approval_node)
    graph.add_node("send_email", send_email_node)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", route_after_validate)  # incoherent input -> straight to END
    graph.add_conditional_edges("eligibility", route_after_eligibility)
    graph.add_edge("deadlines", "logistics")
    graph.add_edge("deadlines", "experience")
    graph.add_edge("logistics", "synthesize")  # fan-in: synthesize waits for both branches
    graph.add_edge("experience", "synthesize")
    graph.add_edge("synthesize", "human_approval")
    graph.add_conditional_edges("human_approval", route_after_approval)  # may cycle back
    graph.add_edge("finalize", "email_approval")
    graph.add_conditional_edges("email_approval", route_after_email_approval)  # skip send if no recipient
    graph.add_edge("send_email", END)

    return graph.compile(checkpointer=MemorySaver())
