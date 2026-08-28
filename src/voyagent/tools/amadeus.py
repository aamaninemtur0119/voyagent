"""Real flight-offer search via the Amadeus Self-Service API (test environment) — actual prices,
airlines, stop counts, and durations, not a deep link. Raises on any failure (missing credentials,
network error, bad response) rather than swallowing it — the caller (the MCP server wrapping this,
and beyond that the Logistics Agent's retry/failure handling) is responsible for deciding what to
do about it, matching how every other tool in this project handles failure."""

import time

import requests

from voyagent.config import settings

TOKEN_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
SEARCH_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

CABIN_MAP = {"Economy": "ECONOMY", "Premium Economy": "PREMIUM_ECONOMY", "Business": "BUSINESS", "First": "FIRST"}

_cached_token: dict = {"value": None, "expires_at": 0.0}


class AmadeusNotConfigured(RuntimeError):
    """Raised when no Amadeus credentials are set — distinct from a real API failure, so callers
    can report 'not connected yet' rather than 'the flight search broke'."""


def _get_token() -> str:
    if not settings.amadeus_api_key or not settings.amadeus_api_secret:
        raise AmadeusNotConfigured("AMADEUS_API_KEY / AMADEUS_API_SECRET not set in .env")

    if _cached_token["value"] and time.time() < _cached_token["expires_at"]:
        return _cached_token["value"]

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": settings.amadeus_api_key,
            "client_secret": settings.amadeus_api_secret,
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    _cached_token["value"] = data["access_token"]
    _cached_token["expires_at"] = time.time() + data["expires_in"] - 30  # refresh a bit early
    return _cached_token["value"]


def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    cabin_class: str = "Economy",
    max_results: int = 5,
) -> list[dict]:
    """origin/destination are IATA airport codes. Dates are YYYY-MM-DD. Returns real Amadeus
    offers — price, currency, airline, stop count, duration — or raises on any failure."""
    token = _get_token()
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": departure_date,
        "returnDate": return_date,
        "adults": 1,
        "travelClass": CABIN_MAP.get(cabin_class, "ECONOMY"),
        "currencyCode": "USD",
        "max": max_results,
    }
    response = requests.get(SEARCH_URL, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=20)
    response.raise_for_status()
    offers = response.json().get("data", [])

    results = []
    for offer in offers:
        price = offer.get("price", {})
        itinerary = offer["itineraries"][0]
        segments = itinerary["segments"]
        results.append({
            "price_total": price.get("total"),
            "currency": price.get("currency"),
            "airline": segments[0].get("carrierCode"),
            "stops": len(segments) - 1,
            "duration": itinerary.get("duration"),
            "departure": segments[0]["departure"]["at"],
            "arrival": segments[-1]["arrival"]["at"],
        })
    return results
