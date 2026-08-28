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
        st.markdown(f"{icon} **{entry['agent']}** — {entry['detail']}")


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
        destination_city = st.text_input("Destination city", placeholder="e.g. Tokyo")
        origin = st.text_input("Flying from (city or airport)", placeholder="e.g. New York")
    with c2:
        purpose = st.selectbox("Purpose", PURPOSES)
        duration = st.text_input("Duration of stay", placeholder="e.g. 14 days")
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=60))
        end_date = st.date_input("End date", value=date.today() + timedelta(days=74))

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
        st.warning("Please fill in nationality, destination city, origin, and duration.")
    else:
        st.session_state.thread_id = str(uuid.uuid4())  # fresh run each time the form is submitted
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {
            "nationality": nationality,
            "destination_country": destination_country,
            "destination_city": destination_city,
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
                    st.caption(f"⚠️ {e}")

    st.divider()

    if st.session_state.awaiting_approval:
        interrupt_payload = result["__interrupt__"][0].value
        st.subheader("🖐️ Human approval needed")
        st.write(interrupt_payload["message"])
        for d in interrupt_payload["deadlines"]:
            st.markdown(f"- **{d['title']}** — {d['date']} _( {d['basis']} )_ — {d['reason']}")

        ac1, ac2 = st.columns(2)
        if ac1.button("✅ Approve — write to Google Calendar", type="primary", use_container_width=True):
            with st.spinner("Resuming graph, writing to calendar..."):
                final = graph.invoke(Command(resume={"approved": True}), config=config)
            st.session_state.trip_result = final
            st.session_state.awaiting_approval = False
            st.rerun()
        if ac2.button("❌ Reject — don't write anything", use_container_width=True):
            with st.spinner("Resuming graph..."):
                final = graph.invoke(Command(resume={"approved": False}), config=config)
            st.session_state.trip_result = final
            st.session_state.awaiting_approval = False
            st.rerun()
    else:
        if result.get("itinerary"):
            st.subheader("📋 Trip Briefing")
            st.markdown(result["itinerary"])
        if result.get("calendar_result"):
            cr = result["calendar_result"]
            st.caption(f"Calendar: {cr.get('status')} — {cr.get('message', '')}")
