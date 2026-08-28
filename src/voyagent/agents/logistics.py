"""Logistics Agent — flights (deep links) + accommodation (live Google Places), with a curation
step: filter/rank accommodation results against the traveler's stated budget preference rather than
just handing back whatever the API returned in whatever order."""

from datetime import date

from voyagent.tools.flights import build_flight_links
from voyagent.tools.google_places import search_accommodation


def run(
    origin: str,
    destination_city: str,
    destination_country: str,
    start_date: str,
    end_date: str,
    budget_level: str = "Any",
    cabin_class: str = "Economy",
    stops_preference: str = "Any",
) -> dict:
    flight_links = build_flight_links(
        origin, destination_city, date.fromisoformat(start_date), date.fromisoformat(end_date),
        cabin_class, stops_preference,
    )

    hotels = search_accommodation(destination_city, destination_country)
    if budget_level != "Any":
        # Keep anything at/under the requested level, or with no price data at all rather than
        # discarding it outright (Google frequently omits price_level for hotels).
        levels = ["$", "$$", "$$$", "$$$$"]
        max_idx = levels.index(budget_level) if budget_level in levels else len(levels) - 1
        filtered = [h for h in hotels if h["price_level"] is None or levels.index(h["price_level"]) <= max_idx]
        hotels = filtered or hotels  # never over-filter down to nothing

    hotels_ranked = sorted(hotels, key=lambda h: (h["rating"] or 0), reverse=True)[:5]

    return {"flight_links": flight_links, "accommodation": hotels_ranked}
