"""Experience Agent — restaurants, places to visit, and activities via live Google Places, curated
against the traveler's stated preferences (dietary, family-friendly, outdoor seating) rather than
just returning raw results."""

from voyagent.tools.google_places import search_activities, search_places_to_visit, search_restaurants


def run(city: str, country: str, preferences: dict | None = None) -> dict:
    preferences = preferences or {}
    pref_phrases = []
    if preferences.get("dietary") in ("Vegetarian", "Vegan"):
        pref_phrases.append(f"{preferences['dietary'].lower()} friendly")
    if preferences.get("family_friendly"):
        pref_phrases.append("family-friendly")
    if preferences.get("outdoor_seating"):
        pref_phrases.append("outdoor seating")
    pref_text = " ".join(pref_phrases)

    restaurants = search_restaurants(city, country, pref_text)
    places = search_places_to_visit(city, country)
    activities = search_activities(city, country)

    def curate(results: list[dict], n: int = 5) -> list[dict]:
        if preferences.get("dietary") in ("Vegetarian", "Vegan"):
            veg = [r for r in results if r.get("vegetarian_options")]
            if veg:
                results = veg + [r for r in results if r not in veg]
        if preferences.get("family_friendly"):
            fam = [r for r in results if r.get("family_friendly")]
            if fam:
                results = fam + [r for r in results if r not in fam]
        return sorted(results, key=lambda r: (r["rating"] or 0), reverse=True)[:n]

    return {
        "restaurants": curate(restaurants),
        "places_to_visit": curate(places),
        "activities": curate(activities),
    }
