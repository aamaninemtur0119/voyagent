"""Real place search via Google Places API (New) — actual ratings, review counts, price
levels, and amenity flags, not RAG. This is a different kind of grounding than the rest of the
project: instead of citing a fetched document, every field here comes directly from a live API
response, field by field. A field is simply omitted (not guessed) when Google doesn't have data
for a given place — never filled in with a default.

Four categories share the same underlying search: restaurants, tourist attractions/landmarks
("places to visit"), things-to-do/experiences ("activities"), and lodging ("accommodation"). The
category only changes the text-search query framing and, for restaurants, adds two
restaurant-specific amenity fields that don't make sense for a landmark, experience provider, or
hotel."""

import requests

from pack_your_bags.config import settings

SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

BASE_FIELDS = [
    "places.displayName",
    "places.formattedAddress",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.goodForChildren",
    "places.googleMapsUri",
]

RESTAURANT_ONLY_FIELDS = ["places.outdoorSeating", "places.servesVegetarianFood"]

PRICE_LEVEL_LABELS = {
    "PRICE_LEVEL_FREE": "Free",
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

QUERY_TEMPLATES = {
    "restaurants": "restaurants in {city}, {country}",
    "places": "tourist attractions and landmarks in {city}, {country}",
    "activities": "tours, classes, and outdoor experiences in {city}, {country}",
    "stay": "hotels in {city}, {country}",
}


def _search(category: str, city: str, country: str, preferences: str = "") -> list[dict]:
    query = QUERY_TEMPLATES[category].format(city=city, country=country)
    if preferences:
        query = f"{preferences} {query}"

    fields = BASE_FIELDS + RESTAURANT_ONLY_FIELDS if category == "restaurants" else BASE_FIELDS

    response = requests.post(
        SEARCH_URL,
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": settings.google_places_api_key,
            "X-Goog-FieldMask": ",".join(fields),
        },
        json={"textQuery": query, "maxResultCount": 10},
        timeout=15,
    )
    response.raise_for_status()
    places = response.json().get("places", [])

    results = []
    for p in places:
        entry = {
            "name": p.get("displayName", {}).get("text", "Unknown"),
            "address": p.get("formattedAddress"),
            "rating": p.get("rating"),
            "review_count": p.get("userRatingCount"),
            "price_level": PRICE_LEVEL_LABELS.get(p.get("priceLevel", ""), None),
            "family_friendly": p.get("goodForChildren"),
            "maps_url": p.get("googleMapsUri"),
        }
        if category == "restaurants":
            entry["outdoor_seating"] = p.get("outdoorSeating")
            entry["vegetarian_options"] = p.get("servesVegetarianFood")
        results.append(entry)
    return results


def search_restaurants(city: str, country: str, preferences: str = "") -> list[dict]:
    return _search("restaurants", city, country, preferences)


def search_places_to_visit(city: str, country: str, preferences: str = "") -> list[dict]:
    return _search("places", city, country, preferences)


def search_activities(city: str, country: str, preferences: str = "") -> list[dict]:
    return _search("activities", city, country, preferences)


def search_accommodation(city: str, country: str, preferences: str = "") -> list[dict]:
    return _search("stay", city, country, preferences)
