"""Pack Your Bags — a multi-agent trip-planning orchestrator. One graph, three specialized agents
(Eligibility, Logistics, Experience), real state passed between them, tool-failure recovery, and
two independent human-in-the-loop approval gates before either write action (writing deadlines to
Google Calendar, and emailing the finished itinerary to the traveler)."""

import uuid
from datetime import date, timedelta

import streamlit as st
from langgraph.types import Command

from voyagent.graph import build_graph

st.set_page_config(page_title="Pack Your Bags", page_icon="🧭", layout="wide")

# --- ambience: a drifting aurora gradient behind a dark UI, frosted-glass panels on top.
#     Pure CSS (no external images) so it always renders and never blocks on a CDN. ---
st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(1100px 700px at 12% -10%, rgba(70,130,255,0.20), transparent 60%),
          radial-gradient(1000px 800px at 108% 8%, rgba(0,220,190,0.16), transparent 55%),
          radial-gradient(900px 900px at 50% 120%, rgba(245,166,35,0.14), transparent 55%),
          linear-gradient(160deg, #0d0f13 0%, #14161a 45%, #10131a 100%);
        background-attachment: fixed;
      }
      .stApp::before {
        content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
        background:
          radial-gradient(38% 32% at 20% 25%, rgba(88,150,255,0.22), transparent 70%),
          radial-gradient(42% 34% at 82% 30%, rgba(0,224,196,0.18), transparent 70%),
          radial-gradient(50% 40% at 55% 95%, rgba(245,166,35,0.16), transparent 70%);
        filter: blur(30px);
        animation: drift 26s ease-in-out infinite alternate;
      }
      @keyframes drift {
        0%   { transform: translate3d(-3%, -2%, 0) scale(1.02); }
        100% { transform: translate3d(4%, 3%, 0) scale(1.08); }
      }
      @media (prefers-reduced-motion: reduce) { .stApp::before { animation: none; } }
      [data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
      [data-testid="stHeader"] { background: transparent; }

      h1 {
        background: linear-gradient(92deg, #ffd27a, #7fe9d4 60%, #8fb8ff);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        letter-spacing: -0.01em;
      }
      /* frosted glass on bordered containers (the plan/result cards) */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.28);
      }
      [data-testid="stTextInput"] input, [data-testid="stDateInput"] input, div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
      }
      .stButton > button { border-radius: 10px; }
      .stButton > button[kind="primary"] { box-shadow: 0 4px 16px rgba(245,166,35,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

DESTINATIONS = ["Japan", "USA", "UK", "Schengen", "Australia"]
PURPOSES = ["Tourism", "Business meeting", "Transit", "Study"]


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


def render_visa_sources(result: dict) -> None:
    """The visa answer's sources, plus a note when a live official source overrode the reference
    data — rendered straight from eligibility state, not the synthesized prose."""
    elig = result.get("eligibility") or {}
    sources = elig.get("sources") or []
    if not sources and not elig.get("divergence_note"):
        return
    st.markdown("### 🛂 Visa information — sources")
    if elig.get("divergence_note"):
        st.warning(
            f"A more current official source overrode the reference data: {md_safe(elig['divergence_note'])}"
        )
    for s in sources:
        name = md_safe(s.get("name") or s.get("url") or "source")
        st.markdown(f"- [{name}]({s['url']})" if s.get("url") else f"- {name}")


def render_result_cards(result: dict) -> None:
    """Structured cards rendered directly from state — not from the LLM's synthesized prose —
    so a link or price can never be dropped/paraphrased away by the synthesis step."""
    logistics = result.get("logistics") or {}
    experience = result.get("experience") or {}

    if logistics:
        route = logistics.get("flight_route")
        st.markdown("### ✈️ Flights" + (f" &nbsp;<span style='font-weight:400;opacity:.65'>{md_safe(route)}</span>" if route else ""), unsafe_allow_html=True)
        if logistics.get("flight_destination_note"):
            st.caption(f"ℹ️ {md_safe(logistics['flight_destination_note'])}")
        status = logistics.get("flight_search_status", "not_attempted")
        if status == "ok" and logistics.get("flight_offers"):
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
                    airline_filtered = o.get("airline_iata") and o["airline_iata"] != "ZZ"
                    label = f"Search {o['airline']}" if airline_filtered else "Search route"
                    oc2.link_button(label, o["search_link"], use_container_width=True)
        elif status == "ok":
            st.caption("No flight offers came back for this route this time — use the search links below.")
        elif status == "not_configured":
            st.caption("Real-time flight search isn't connected — use the search links below.")
        else:
            st.caption(f"Flight search hit an issue ({status}) — use the search links below.")
        link_cols = st.columns(len(logistics.get("flight_links", {})) or 1)
        for col, (label, url) in zip(link_cols, logistics.get("flight_links", {}).items()):
            col.link_button(label, url, use_container_width=True)

        # accommodation is keyed by city: {"Tokyo": [...], "Kyoto": [...]} — every city the
        # traveler listed gets its own hotel options, not just the primary one.
        accommodation = logistics.get("accommodation") or {}
        multi_city_stay = len(accommodation) > 1
        for city, hotels in accommodation.items():
            if hotels:
                st.markdown("### 🏨 Accommodation" + (f" — {city}" if multi_city_stay else ""))
                for h in hotels:
                    _place_card(h)

    # experience is keyed by city: {"Tokyo": {"restaurants": [...], ...}, "Kyoto": {...}} — every
    # city the traveler listed gets real, searched data, not just the primary one.
    for city, city_experience in experience.items():
        multi_city = len(experience) > 1
        for label, key in [("🍽️ Restaurants", "restaurants"), ("📍 Places to Visit", "places_to_visit"), ("🎯 Activities", "activities")]:
            items = (city_experience or {}).get(key)
            if items:
                st.markdown(f"### {label}" + (f" — {city}" if multi_city else ""))
                for item in items:
                    _place_card(item)


st.title("🧭 Pack Your Bags")
st.caption(
    "Tell it where you're headed once. Get back visa eligibility, the deadlines that come with it, "
    "real flights and hotels, and a day-by-day plan."
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "trip_result" not in st.session_state:
    st.session_state.trip_result = None
if "awaiting_approval" not in st.session_state:
    st.session_state.awaiting_approval = False

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

form_col = st.container()

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

    traveler_email = st.text_input(
        "Your email",
        placeholder="you@example.com",
        help="Required — the finished itinerary is emailed here once you approve it at the end.",
    )

    plan_clicked = st.button("Plan My Trip", type="primary", use_container_width=True)

if plan_clicked:
    if not (nationality and destination_city and origin and duration and traveler_email.strip()):
        st.warning("Please fill in nationality, destination city, origin, and your email, and make sure the end date is after the start date.")
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
            "traveler_email": traveler_email.strip(),
            "preferences": {
                "dietary": dietary,
                "family_friendly": family_friendly,
                "outdoor_seating": outdoor_seating,
                "budget_level": budget_level,
            },
            "agent_trace": [],
            "errors": [],
        }
        with st.spinner("Putting your trip together…"):
            try:
                result = graph.invoke(initial_state, config=config)
            except Exception as e:  # noqa: BLE001 - surface in the UI rather than crashing the app
                st.error(f"The orchestrator itself hit an unrecoverable error: {e}")
                result = None

        if result is not None:
            st.session_state.trip_result = result
            st.session_state.awaiting_approval = bool(result.get("__interrupt__"))

if st.session_state.trip_result and st.session_state.trip_result.get("input_error"):
    st.warning(f"⚠️ {md_safe(st.session_state.trip_result['input_error'])}")
    st.caption("Nothing was planned — fix the fields above and hit **Plan My Trip** again.")

elif st.session_state.trip_result:
    result = st.session_state.trip_result
    if result.get("errors"):
        with st.expander(f"⚠️ {len(result['errors'])} issue(s) hit while planning — the plan continued without them"):
            for e in result["errors"]:
                st.caption(md_safe(f"• {e}"))

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
            st.subheader("📋 Review your plan")
            last_rev = interrupt_payload.get("last_revision")
            if last_rev:
                lines = "\n".join(f"- {md_safe(ch)}" for ch in last_rev.get("changes", []))
                st.success(f"✏️ **Revised:** {md_safe(last_rev.get('summary', ''))}\n\n{lines}")
            st.write(interrupt_payload["message"])

            # --- decision block FIRST so it's visible without scrolling past the whole plan,
            #     especially right after a revise. Plan details are shown below for reference. ---
            replans_left = interrupt_payload.get("replans_remaining", 0)
            feedback_key = f"feedback_{interrupt_payload.get('replans_so_far', 0)}"
            if replans_left > 0:
                feedback = st.text_input(
                    "Want changes? Describe them, then hit Revise "
                    "(e.g. 'push the trip back a month', 'cheapest hotels only', 'Kyoto instead of Tokyo') "
                    f"— {replans_left} revision(s) left",
                    key=feedback_key,
                )
            else:
                feedback = ""
                st.caption("No revisions left — approve the plan or cancel.")

            # Three explicit outcomes, stable labels + explicit keys: a button whose label changes
            # with other widget state can make Streamlit lose a click between the render that showed
            # it and the one processing it (found live). Revise is disabled until there's feedback to
            # act on, so it can never be an ambiguous no-op.
            ac1, ac2, ac3 = st.columns(3)
            if ac1.button("✅ Approve plan", type="primary", use_container_width=True, key="approve_btn"):
                resume({"approved": True})
            if ac2.button(
                "✏️ Revise", use_container_width=True, key="revise_btn",
                disabled=not (replans_left > 0 and feedback.strip()),
            ):
                resume({"approved": False, "feedback": st.session_state.get(feedback_key, "")})
            if ac3.button("✖️ Cancel", use_container_width=True, key="cancel_btn"):
                resume({"approved": False, "cancelled": True})
            st.caption(
                "**Approve** adds the visa deadlines to your calendar, then moves on to emailing your "
                "itinerary. **Revise** re-plans with your changes. **Cancel** stops here — nothing is "
                "added to your calendar or sent."
            )

            st.divider()
            st.markdown("#### Your plan")
            if interrupt_payload.get("itinerary"):
                with st.container(border=True):
                    st.markdown(md_safe(interrupt_payload["itinerary"]))
            for d in interrupt_payload.get("deadlines", []):
                st.markdown(f"- **{md_safe(d['title'])}** — {d['date']} _( {d['basis']} )_ — {md_safe(d['reason'])}")
            render_visa_sources(result)
            render_result_cards(result)

        elif itype == "email_approval":
            st.subheader("📧 Your call — send this to your inbox?")
            st.write(interrupt_payload["message"])
            st.caption(f"To: **{md_safe(interrupt_payload.get('recipient', ''))}**")
            with st.container(border=True):
                st.markdown(md_safe(interrupt_payload.get("preview", "")) + "…")

            ec1, ec2 = st.columns(2)
            if ec1.button("✅ Approve — send the email", type="primary", use_container_width=True, key="email_approve_btn"):
                resume({"approved": True})
            if ec2.button("❌ Reject — don't send", use_container_width=True, key="email_reject_btn"):
                resume({"approved": False})
    elif result.get("cancelled"):
        st.warning("Planning cancelled — nothing was added to your calendar or sent to your inbox. Adjust the form above and plan again.")
    else:
        if result.get("itinerary"):
            st.subheader("📋 Trip Briefing")
            st.markdown(md_safe(result["itinerary"]))
        render_visa_sources(result)
        render_result_cards(result)
        if result.get("calendar_result"):
            cr = result["calendar_result"]
            st.caption(md_safe(f"Calendar: {cr.get('status')} — {cr.get('message', '')}"))
        if result.get("email_result"):
            er = result["email_result"]
            st.caption(md_safe(f"Email: {er.get('status')} — {er.get('message', '')}"))
