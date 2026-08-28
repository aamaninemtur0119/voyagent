"""Logistics Agent — flights (deep links) + accommodation (live Google Places), with a genuine
judgment step: an LLM picks and justifies the best hotel options for this specific traveler,
grounded only in the actual returned fields (rating, review count, price level, family-friendly) —
never inventing an amenity or fact the data doesn't contain. Not just sorted by rating."""

from datetime import date

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from voyagent.config import settings
from voyagent.tools.flights import build_flight_links
from voyagent.tools.google_places import search_accommodation

_llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)


class HotelPick(BaseModel):
    name: str = Field(description="Must exactly match a 'name' field from the candidate list — never invent a hotel.")
    reason: str = Field(
        description=(
            "Why this hotel suits this traveler, grounded ONLY in the fields provided for it "
            "(rating, review count, price level, family-friendly, address). Never state an amenity, "
            "view, or feature that isn't in the provided data."
        )
    )


class LogisticsCuration(BaseModel):
    picks: list[HotelPick] = Field(
        description="3-5 best options for this traveler, best first. Empty list if no candidates were provided."
    )


def _curate_hotels(candidates: list[dict], budget_level: str, preferences: dict) -> list[dict]:
    if not candidates:
        return []
    listing = "\n".join(
        f"- {h['name']} | rating: {h['rating']} ({h['review_count']} reviews) | price: {h['price_level']} | "
        f"family-friendly: {h['family_friendly']} | {h['address']}"
        for h in candidates
    )
    prompt = (
        "You are curating hotel options for a traveler. Pick and rank the best 3-5 from the "
        "candidates below, weighing rating, review count, and price level against this traveler's "
        f"stated budget ({budget_level}) and preferences ({preferences}). Justify each pick using "
        "ONLY the fields shown — do not invent amenities, views, or anything not listed.\n\n"
        f"Candidates:\n{listing}"
    )
    curation = _llm.with_structured_output(LogisticsCuration).invoke(prompt)

    by_name = {h["name"]: h for h in candidates}
    picked = []
    for pick in curation.picks:
        hotel = by_name.get(pick.name)
        if hotel:  # defensive: ignore a name that doesn't match a real candidate
            picked.append({**hotel, "why": pick.reason})
    return picked


def run(
    origin: str,
    destination_city: str,
    destination_country: str,
    start_date: str,
    end_date: str,
    budget_level: str = "Any",
    preferences: dict | None = None,
    cabin_class: str = "Economy",
    stops_preference: str = "Any",
) -> dict:
    flight_links = build_flight_links(
        origin, destination_city, date.fromisoformat(start_date), date.fromisoformat(end_date),
        cabin_class, stops_preference,
    )

    hotels = search_accommodation(destination_city, destination_country)
    if budget_level != "Any":
        levels = ["$", "$$", "$$$", "$$$$"]
        max_idx = levels.index(budget_level) if budget_level in levels else len(levels) - 1
        filtered = [h for h in hotels if h["price_level"] is None or levels.index(h["price_level"]) <= max_idx]
        hotels = filtered or hotels  # never over-filter down to nothing

    candidates = sorted(hotels, key=lambda h: (h["rating"] or 0), reverse=True)[:8]
    curated = _curate_hotels(candidates, budget_level, preferences or {})

    return {"flight_links": flight_links, "accommodation": curated or candidates[:5]}
