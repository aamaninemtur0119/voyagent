"""Flight search — tool-call based, not RAG. No live pricing API is connected (that would need
a paid provider and an API key, a decision for the user to make), so instead of fabricating or
guessing prices, this builds pre-filled deep links to real aggregators that already do live price
comparison across airlines. Clicking through gets real, current prices.

Origin/destination accept a city name, airport name, or IATA code — resolve_airport_code() uses
an LLM to normalize whatever was typed into the IATA code the deep-link URLs actually need, since
a static lookup table can't cover every city/airport name a user might type.

URL formats confirmed via web search on 2026-08-23. Skyscanner and Kayak have documented,
reasonably stable route+date URL patterns. Google Flights' precise parameter encoding (a binary
`tfs` field) isn't publicly documented, so its officially-supported natural-language query mode is
used instead of guessing that encoding. Stops/layover preference isn't reliably encodable as a URL
parameter for any of these three sites, so it's surfaced as guidance to apply on the results page
rather than promised as a pre-applied filter that might silently not work.
"""

from datetime import date
from urllib.parse import quote

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from voyagent.config import settings

_llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)

SKYSCANNER_CABIN = {
    "Economy": "economy",
    "Premium Economy": "premiumeconomy",
    "Business": "business",
    "First": "first",
}


class ResolvedAirport(BaseModel):
    iata_code: str = Field(
        description="The 3-letter IATA airport code. If the input names a city with multiple airports, "
        "pick that city's primary/largest international airport."
    )


def resolve_airport_code(text: str) -> str:
    text = text.strip()
    if len(text) == 3 and text.isalpha():
        return text.upper()  # already looks like a code — skip the LLM call
    result = _llm.with_structured_output(ResolvedAirport).invoke(
        f"What is the 3-letter IATA airport code for this airport, city, or place: '{text}'? "
        "If it names a city with multiple airports, pick the primary international one."
    )
    return result.iata_code.upper()


def build_flight_links(
    origin: str,
    destination: str,
    start_date: date,
    end_date: date,
    cabin_class: str,
    stops_preference: str,
) -> dict[str, str]:
    origin = resolve_airport_code(origin)
    destination = resolve_airport_code(destination)
    start_iso, end_iso = start_date.isoformat(), end_date.isoformat()

    stops_phrase = "nonstop " if stops_preference == "Nonstop only" else ""
    query = (
        f"Flights from {origin} to {destination} on {start_iso} returning {end_iso}, "
        f"{stops_phrase}{cabin_class.lower()} class"
    )
    google_flights_url = f"https://www.google.com/travel/flights?q={quote(query)}"

    # Skyscanner requires YYMMDD dates specifically — YYYY-MM-DD 404s (verified 2026-08-23).
    sky_start, sky_end = start_date.strftime("%y%m%d"), end_date.strftime("%y%m%d")
    sky_cabin = SKYSCANNER_CABIN.get(cabin_class, "economy")
    skyscanner_url = (
        f"https://www.skyscanner.com/transport/flights/{origin}/{destination}/"
        f"{sky_start}/{sky_end}/?adultsv2=1&cabinclass={sky_cabin}"
    )

    kayak_url = f"https://www.kayak.com/flights/{origin}-{destination}/{start_iso}/{end_iso}?sort=price_a"

    return {
        "Google Flights": google_flights_url,
        "Skyscanner": skyscanner_url,
        "Kayak": kayak_url,
    }


def build_airline_search_link(
    origin_code: str, destination_code: str, start_date: str, end_date: str, cabin_class: str, airline_name: str,
) -> str:
    """A per-offer search link nudging toward the airline a sandbox Duffel offer named. Not a
    link to that exact flight — Duffel's sandbox prices/schedules aren't real bookable flights,
    so there's nothing genuine to deep-link to the way a hotel's real Google Maps listing is.
    Google Flights' natural-language query mode is the only one of the three aggregators where an
    airline name can be included at all; Skyscanner/Kayak's URL schemes don't have an easy
    per-airline parameter for a deep link like this."""
    query = f"Flights from {origin_code} to {destination_code} on {start_date} returning {end_date}, {cabin_class.lower()} class on {airline_name}"
    return f"https://www.google.com/travel/flights?q={quote(query)}"
