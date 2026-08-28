"""Real flight-offer search via the Duffel API — a genuine live request/response round-trip
against a real airline-data API, not fabricated data. Uses Duffel's free, instantly-self-service
test mode (a `duffel_test_...` token, no approval process) — Amadeus and Kiwi's equivalent free
self-service tiers were both discontinued in 2026 (confirmed live: Amadeus shut theirs down
entirely in July 2026, Kiwi's Tequila API is now partner-approval-only).

Honest caveat, disclosed in the UI rather than hidden: test mode returns offers modeled on real
airlines/routes/schedules against Duffel's own sandbox economics (verified live: real IATA codes,
real flight-number formats, a real multi-offer response), not live market fares. This is a real
API integration exercising a real data contract — not an LLM inventing a price — but the specific
numbers aren't bookable market prices, and callers should say so.

Raises on any failure rather than swallowing it — the caller (the MCP server wrapping this, and
beyond that the Logistics Agent's retry/failure handling) decides what to do about it, matching
how every other tool in this project handles failure.
"""

import requests

from voyagent.config import settings

SEARCH_URL = "https://api.duffel.com/air/offer_requests"
DUFFEL_VERSION = "v2"

CABIN_MAP = {"Economy": "economy", "Premium Economy": "premium_economy", "Business": "business", "First": "first"}


class DuffelNotConfigured(RuntimeError):
    """Raised when no Duffel token is set — distinct from a real API failure, so callers can
    report 'not connected yet' rather than 'the flight search broke'."""


def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    cabin_class: str = "Economy",
    max_results: int = 5,
) -> list[dict]:
    """origin/destination are IATA airport codes. Dates are YYYY-MM-DD. Returns real Duffel
    sandbox offers — price, currency, airline, stop count, duration — cheapest first, or raises
    on any failure."""
    if not settings.duffel_api_key:
        raise DuffelNotConfigured("DUFFEL_API_KEY not set in .env")

    response = requests.post(
        SEARCH_URL,
        params={"return_offers": "true"},
        headers={
            "Authorization": f"Bearer {settings.duffel_api_key}",
            "Duffel-Version": DUFFEL_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "data": {
                "slices": [
                    {"origin": origin, "destination": destination, "departure_date": departure_date},
                    {"origin": destination, "destination": origin, "departure_date": return_date},
                ],
                "passengers": [{"type": "adult"}],
                "cabin_class": CABIN_MAP.get(cabin_class, "economy"),
            }
        },
        timeout=30,
    )
    response.raise_for_status()
    offers = response.json().get("data", {}).get("offers", [])

    results = []
    for offer in sorted(offers, key=lambda o: float(o["total_amount"]))[:max_results]:
        outbound = offer["slices"][0]
        results.append({
            "price_total": offer["total_amount"],
            "currency": offer["total_currency"],
            "airline": offer.get("owner", {}).get("name", "Unknown"),
            "stops": len(outbound["segments"]) - 1,
            "duration": outbound.get("duration"),
            "departure": outbound["segments"][0]["departing_at"],
            "arrival": outbound["segments"][-1]["arriving_at"],
        })
    return results
