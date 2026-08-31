"""Experience Agent — restaurants, places to visit, and activities via live Google Places, with a
genuine judgment step per category: an LLM picks and justifies the best options for this specific
traveler's stated preferences, grounded only in the actual returned fields — never inventing a dish,
view, or amenity the data doesn't contain."""

from pydantic import BaseModel, Field

from voyagent.llm import structured
from voyagent.tools.google_places import search_activities, search_places_to_visit, search_restaurants


class Pick(BaseModel):
    name: str = Field(description="Must exactly match a 'name' field from the candidate list — never invent a place.")
    reason: str = Field(
        description=(
            "Why this suits the traveler, grounded ONLY in the fields provided (rating, review "
            "count, price level, family-friendly, outdoor seating, vegetarian options). Never state "
            "a cuisine, dish, view, or feature that isn't in the provided data."
        )
    )


class Curation(BaseModel):
    picks: list[Pick] = Field(description="3-5 best options, best first. Empty list if no candidates were provided.")


def _curate(category: str, candidates: list[dict], preferences: dict) -> list[dict]:
    if not candidates:
        return []
    listing = "\n".join(
        f"- {r['name']} | rating: {r['rating']} ({r['review_count']} reviews) | price: {r.get('price_level')} | "
        f"family-friendly: {r.get('family_friendly')} | outdoor seating: {r.get('outdoor_seating')} | "
        f"vegetarian options: {r.get('vegetarian_options')}"
        for r in candidates
    )
    prompt = (
        f"You are curating {category} for a traveler. Pick and rank the best 3-5 from the "
        f"candidates below against this traveler's stated preferences ({preferences}). Justify "
        "each pick using ONLY the fields shown — do not invent cuisine types, dishes, views, or "
        "anything not listed.\n\n"
        f"Candidates:\n{listing}"
    )
    # default=None so a malformed LLM response degrades to "no curation" — the caller falls back to
    # the top few raw results — rather than throwing and nulling this city's whole Experience block.
    curation = structured(Curation, prompt, default=None)
    if curation is None:
        return []

    by_name = {r["name"]: r for r in candidates}
    picked = []
    for pick in curation.picks:
        item = by_name.get(pick.name)
        if item:
            picked.append({**item, "why": pick.reason})
    return picked


def _run_one_city(city: str, country: str, preferences: dict) -> dict:
    pref_phrases = []
    if preferences.get("dietary") in ("Vegetarian", "Vegan"):
        pref_phrases.append(f"{preferences['dietary'].lower()} friendly")
    if preferences.get("family_friendly"):
        pref_phrases.append("family-friendly")
    if preferences.get("outdoor_seating"):
        pref_phrases.append("outdoor seating")
    pref_text = " ".join(pref_phrases)

    restaurants = search_restaurants(city, country, pref_text)[:8]
    places = search_places_to_visit(city, country)[:8]
    activities = search_activities(city, country)[:8]

    return {
        "restaurants": _curate("restaurants", restaurants, preferences) or restaurants[:5],
        "places_to_visit": _curate("places to visit", places, preferences) or places[:5],
        "activities": _curate("activities", activities, preferences) or activities[:5],
    }


def run(cities: list[str] | str, country: str, preferences: dict | None = None) -> dict:
    """Searches EVERY city the traveler listed, not just the primary one — a city they explicitly
    named deserves real, curated data, not a general-knowledge stand-in. Returns
    {city_name: {restaurants, places_to_visit, activities}}. A single string is still accepted for
    backward compatibility and returns a one-city dict."""
    preferences = preferences or {}
    city_list = [cities] if isinstance(cities, str) else cities
    return {city: _run_one_city(city, country, preferences) for city in city_list}
