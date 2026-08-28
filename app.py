"""Voyagent — a multi-agent trip-planning orchestrator. One graph, three specialized agents
(Eligibility, Logistics, Experience), real state passed between them, tool-failure recovery, and a
human-in-the-loop approval gate before the one write action (Google Calendar). This UI is built to
make that control flow visible, not hidden behind a single final answer — the agent-trace panel
below the form shows exactly what ran, in what order, and what (if anything) failed."""

import uuid
from datetime import date, timedelta

import streamlit as st
from langgraph.types import Command

from voyagent.graph import build_graph

st.set_page_config(page_title="Voyagent", page_icon="🧭", layout="wide")

DESTINATIONS = ["Japan", "USA", "UK", "Schengen", "Australia"]
PURPOSES = ["Tourism", "Business meeting", "Transit", "Study"]

STATUS_ICON = {"done": "✅", "failed": "⚠️", "running": "⏳", "retrying": "🔁", "skipped": "⏭️"}


def md_safe(text) -> str:
    """Escape '$' before handing text to any Streamlit markdown-rendering call. Streamlit's
    markdown renderer treats a pair of '$' as LaTeX math-mode, which silently mangles any text
    containing two or more dollar signs — collapsing whitespace and italicizing everything
    between them (found live: a flight recommendation mentioning a price came out as an
    unreadable run-on). This is a travel app — prices/fees show up constantly in LLM-generated
    text and API data — so this is applied broadly rather than patched per call site."""
    return str(text).replace("$", "\\$")


@st.cache_resource
def get_graph():
    # One graph + one in-memory checkpointer for the life of the server process — each browser
    # session gets its own thread_id, so concurrent users don't share state, but a single user's
    # session correctly resumes across reruns (Streamlit reruns the whole script on every
    # interaction, so the graph object itself must NOT be rebuilt each time).
    return build_graph()


def render_trace(trace: list[dict]) -> None:
    for entry in trace:
        icon = STATUS_ICON.get(entry["status"], "•")
        st.markdown(f"{icon} **{entry['agent']}** — {md_safe(entry['detail'])}")


def _place_card(item: dict) -> None:
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{md_safe(item['name'])}**")
            badges = []
            if item.get("rating") is not None:
                badges.append(f"⭐ {item['rating']} ({item.get('review_count', 0)} reviews)")
            if item.get("price_level"):
                badges.append(f"💰 {md_safe(item['price_level'])}")
            if badges:
                st.caption(" · ".join(badges))
            if item.get("why"):
                st.caption(md_safe(item["why"]))
        with col2:
            if item.get("maps_url"):
                st.link_button("View / Book", item["maps_url"], use_container_width=True)


def render_result_cards(result: dict) -> None:
    """Structured cards rendered directly from state — not from the LLM's synthesized prose —
    so a link or price can never be dropped/paraphrased away by the synthesis step."""
    logistics = result.get("logistics") or {}
    experience = result.get("experience") or {}

    if logistics:
        st.markdown("### ✈️ Flights")
        status = logistics.get("flight_search_status", "not_attempted")
        if status == "ok" and logistics.get("flight_offers"):
            st.caption("⚠️ Sandbox data (Duffel test mode) — real request/response, not live market fares.")
            if logistics.get("flight_recommendation"):
                st.markdown(
                    f'<div style="border-left: 4px solid #F5A623; background: rgba(245,166,35,0.12); '
                    f'padding: 0.75rem 1rem; border-radius: 4px; margin-bottom: 0.5rem;">'
                    f'<b>🏆 Recommended:</b> {md_safe(logistics["flight_recommendation"])}</div>',
                    unsafe_allow_html=True,
                )
            for o in logistics["flight_offers"]:
                oc1, oc2 = st.columns([4, 1])
                oc1.caption(md_safe(
                    f"{o['airline']} — {o['price_total']} {o['currency']} — {o['stops']} stop(s) — {o['duration']}"
                ))
                if o.get("search_link"):
                    oc2.link_button("Search", o["search_link"], use_container_width=True)
            st.caption("Not this exact sandbox flight (it isn't real) — searches for that airline/route on Google Flights. Or compare more broadly:")
        else:
            reason = "real-time flight search isn't connected yet" if status == "not_configured" else f"flight search hit an issue ({status})"
            st.caption(f"No live flight prices this time — {reason}. Search directly:")
        link_cols = st.columns(len(logistics.get("flight_links", {})) or 1)
        for col, (label, url) in zip(link_cols, logistics.get("flight_links", {}).items()):
            col.link_button(label, url, use_container_width=True)

        if logistics.get("accommodation"):
            st.markdown("### 🏨 Accommodation")
            for h in logistics["accommodation"]:
                _place_card(h)

    for label, key in [("🍽️ Restaurants", "restaurants"), ("📍 Places to Visit", "places_to_visit"), ("🎯 Activities", "activities")]:
        items = experience.get(key)
        if items:
            st.markdown(f"### {label}")
            for item in items:
                _place_card(item)


st.title("🧭 Voyagent")
st.caption(
    "A multi-agent trip-planning orchestrator — Eligibility, Logistics, and Experience agents "
    "coordinated by an orchestrator, with real state, tool-failure recovery, and a human approval "
    "gate before anything gets written to your calendar."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "trip_result" not in st.session_state:
    st.session_state.trip_result = None
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

form_col, trace_col = st.columns([2, 1])

with form_col:
    st.subheader("Plan a trip")
    c1, c2 = st.columns(2)
    with c1:
        nationality = st.text_input("Nationality", placeholder="e.g. China, India, Canada")
        destination_country = st.selectbox("Destination country", DESTINATIONS)
        destination_city = st.text_input("Primary destination city", placeholder="e.g. Tokyo")
        other_cities = st.text_input(
            "Other cities also in your plan (optional, comma-separated)",
            placeholder="e.g. Kyoto, Osaka — leave blank if it's a single-city trip",
        )
        origin = st.text_input("Flying from (city or airport)", placeholder="e.g. New York")
    with c2:
        purpose = st.selectbox("Purpose", PURPOSES)
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=60))
        end_date = st.date_input("End date", value=date.today() + timedelta(days=74))
        # Duration is derived from the dates, not asked separately — a free-text duration field
        # alongside start/end dates let them silently contradict each other (e.g. "14 days"
        # stated while the actual date range was 100+ days), feeding the wrong duration to the
        # Eligibility Agent's visa reasoning while flights/hotels reflected the real range. Single
        # source of truth instead: the dates are what's real, duration is just their difference.
        trip_days = (end_date - start_date).days
        duration = f"{trip_days} days" if trip_days > 0 else ""
        st.caption(f"Duration: **{trip_days} days**" if trip_days > 0 else "⚠️ End date must be after start date.")

    st.caption("Preferences (optional — used by the Experience and Logistics agents)")
    p1, p2, p3 = st.columns(3)
    with p1:
        dietary = st.selectbox("Dietary", ["No preference", "Vegetarian", "Vegan"])
    with p2:
        family_friendly = st.checkbox("Family-friendly")
        outdoor_seating = st.checkbox("Outdoor seating")
    with p3:
        budget_level = st.selectbox("Budget", ["Any", "$", "$$", "$$$", "$$$$"])

    plan_clicked = st.button("Plan My Trip", type="primary", use_container_width=True)

with trace_col:
    st.subheader("Agent trace")
    trace_placeholder = st.container()

if plan_clicked:
    if not (nationality and destination_city and origin and duration):
        st.warning("Please fill in nationality, destination city, and origin, and make sure the end date is after the start date.")
    else:
        st.session_state.thread_id = str(uuid.uuid4())  # fresh run each time the form is submitted
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {
            "nationality": nationality,
            "destination_country": destination_country,
            "destination_city": destination_city,
            "destination_cities": [destination_city] + [c.strip() for c in other_cities.split(",") if c.strip()],
            "origin": origin,
            "purpose": purpose,
            "duration": duration,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "preferences": {
                "dietary": dietary,
                "family_friendly": family_friendly,
                "outdoor_seating": outdoor_seating,
                "budget_level": budget_level,
            },
            "agent_trace": [],
            "errors": [],
        }
        with st.spinner("Running the agent pipeline (Eligibility → Logistics → Experience → synthesis)..."):
            try:
                result = graph.invoke(initial_state, config=config)
            except Exception as e:  # noqa: BLE001 - surface in the UI rather than crashing the app
                st.error(f"The orchestrator itself hit an unrecoverable error: {e}")
                result = None

        if result is not None:
            st.session_state.trip_result = result
            st.session_state.awaiting_approval = bool(result.get("__interrupt__"))

if st.session_state.trip_result:
    result = st.session_state.trip_result
    with trace_col:
        render_trace(result.get("agent_trace", []))
        if result.get("errors"):
            with st.expander(f"Errors encountered ({len(result['errors'])})"):
                for e in result["errors"]:
                    st.caption(md_safe(f"⚠️ {e}"))

    st.divider()

    if st.session_state.awaiting_approval:
        interrupt_payload = result["__interrupt__"][0].value
        itype = interrupt_payload["type"]

        def resume(decision: dict) -> None:
            with st.spinner("Resuming..."):
                try:
                    final = graph.invoke(Command(resume=decision), config=config)
                except Exception as e:  # noqa: BLE001 - surface visibly rather than silently doing nothing
                    st.error(f"Resuming the graph failed: {e}")
                    return
            st.session_state.trip_result = final
            st.session_state.awaiting_approval = bool(final.get("__interrupt__"))
            st.rerun()

        if itype == "calendar_write_approval":
            st.subheader("🖐️ Human approval needed — Calendar")
            st.write(interrupt_payload["message"])
            if interrupt_payload.get("itinerary"):
                with st.container(border=True):
                    st.markdown(md_safe(interrupt_payload["itinerary"]))
            for d in interrupt_payload.get("deadlines", []):
                st.markdown(f"- **{md_safe(d['title'])}** — {d['date']} _( {d['basis']} )_ — {md_safe(d['reason'])}")
            render_result_cards(result)

            feedback_key = f"feedback_{interrupt_payload.get('replans_so_far', 0)}"
            feedback = ""
            if interrupt_payload.get("replans_remaining", 0) > 0:
                feedback = st.text_input(
                    "Or describe a change instead (e.g. 'push the trip back a month', 'cheapest hotels only') "
                    f"— {interrupt_payload['replans_remaining']} replan(s) left",
                    key=feedback_key,
                )
                if feedback.strip():
                    st.caption("Rejecting will replan using this feedback instead of just cancelling.")
            else:
                st.caption("Replan limit reached — approve or reject only.")

            # Stable labels + explicit keys on purpose: a button's identity defaults to its label
            # text, so a label that changes based on other widget state (e.g. appending "— replan"
            # only once feedback is non-empty) can make Streamlit lose track of a click between
            # the render that showed it and the one processing it. Found live: a real click that
            # silently did nothing.
            ac1, ac2 = st.columns(2)
            if ac1.button("✅ Approve — write to Google Calendar", type="primary", use_container_width=True, key="approve_btn"):
                resume({"approved": True})
            if ac2.button("❌ Reject", use_container_width=True, key="reject_btn"):
                resume({"approved": False, "feedback": st.session_state.get(feedback_key, "")})

        elif itype == "export_approval":
            st.subheader("🖐️ Human approval needed — Save Itinerary")
            st.write(interrupt_payload["message"])
            st.caption(md_safe(interrupt_payload.get("preview", "")) + "...")

            ec1, ec2 = st.columns(2)
            if ec1.button("✅ Approve — save to file", type="primary", use_container_width=True):
                resume({"approved": True})
            if ec2.button("❌ Reject — don't save", use_container_width=True):
                resume({"approved": False})
    else:
        if result.get("itinerary"):
            st.subheader("📋 Trip Briefing")
            st.markdown(md_safe(result["itinerary"]))
        render_result_cards(result)
        if result.get("calendar_result"):
            cr = result["calendar_result"]
            st.caption(md_safe(f"Calendar: {cr.get('status')} — {cr.get('message', '')}"))
        if result.get("export_result"):
            er = result["export_result"]
            st.caption(md_safe(f"Export: {er.get('status')}" + (f" — saved to `{er['path']}`" if er.get("path") else "")))
