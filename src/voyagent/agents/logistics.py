"""Logistics Agent — flights (real search via an MCP tool, Duffel-backed test/sandbox mode) +
accommodation (live Google Places), each with a genuine judgment step: an LLM picks and justifies
the best option(s) for this specific traveler, grounded only in the actual returned fields — never
inventing a price, amenity, or fact the data doesn't contain. Note: Duffel's free test mode returns
real API responses over real airline/route data, but sandbox prices, not live market fares —
disclosed as such in the UI, not presented as real live pricing.

Flights are the one tool in Voyagent exposed via the Model Context Protocol rather than a direct
Python import (see mcp_servers/flights_server.py) — new capability, added specifically to
demonstrate that pattern. A tool-level MCP failure surfaces as a text-content error block, not a
raised Python exception (verified empirically, not assumed) — this module translates that into a
real exception so it flows through the exact same retry/graceful-degradation path every other tool
in this project uses, rather than needing its own special case. If flight search still isn't
available after retrying, Logistics falls back to deep links only, clearly labeled as such — never
silently drops the flights section.
"""

import asyncio
import json
from datetime import date

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field

from voyagent.llm import structured
from voyagent.tools.flights import build_airline_search_link, build_flight_links, resolve_airport
from voyagent.tools.google_places import search_accommodation

_MCP_SERVERS = {
    "flights": {
        "command": "uv",
        "args": ["run", "--directory", str(__import__("pathlib").Path(__file__).resolve().parents[3]),
                 "python", "-m", "voyagent.mcp_servers.flights_server"],
        "transport": "stdio",
    }
}


async def _call_flight_mcp_tool(**kwargs) -> list[dict]:
    client = MultiServerMCPClient(_MCP_SERVERS)
    tools = await client.get_tools()
    tool = next(t for t in tools if t.name == "search_flights")
    result = await tool.ainvoke(kwargs)

    if not isinstance(result, list) or not result:
        return []

    # MCP surfaces a tool-level failure as a single text-content error block, not a raised
    # exception — translate it into a real one so the caller's normal exception handling applies.
    first_text = result[0].get("text", "")
    if first_text.startswith("Error executing tool"):
        raise RuntimeError(first_text)

    # A successful list[dict] return comes back as ONE text-content block PER LIST ITEM
    # (verified empirically, not assumed) — not one block containing the whole JSON array.
    return [json.loads(block["text"]) for block in result]


MCP_CALL_TIMEOUT_SECONDS = 25


def search_flights_via_mcp(
    origin_code: str, destination_code: str, departure_date: str, return_date: str, cabin_class: str, max_results: int = 5,
) -> list[dict]:
    """Sync wrapper — the rest of the graph is sync, so the async MCP client call is run to
    completion here rather than converting the whole graph to async execution for one tool.
    Bounded with an explicit timeout: a spawned subprocess (the MCP server) can hang rather than
    fail fast (observed live — a run that never returned, distinct from the clean, fast exceptions
    every other failure mode produces), and an unbounded hang defeats the retry-then-continue
    policy the rest of this project relies on — retrying twice is pointless if each attempt can
    block forever."""
    async def _run_with_timeout():
        return await asyncio.wait_for(
            _call_flight_mcp_tool(
                origin=origin_code, destination=destination_code, departure_date=departure_date,
                return_date=return_date, cabin_class=cabin_class, max_results=max_results,
            ),
            timeout=MCP_CALL_TIMEOUT_SECONDS,
        )
    return asyncio.run(_run_with_timeout())


class FlightRecommendation(BaseModel):
    summary: str = Field(
        description=(
            "One recommended option and why, grounded ONLY in the fields provided (price, "
            "currency, airline, stops, duration) — weigh against the traveler's stated budget/"
            "cabin preference. Never invent a price or airline not in the data."
        )
    )


def _recommend_flight(offers: list[dict], cabin_class: str, budget_level: str) -> str | None:
    if not offers:
        return None
    listing = "\n".join(
        f"- {o['airline']} | {o['price_total']} {o['currency']} | {o['stops']} stop(s) | {o['duration']}"
        for o in offers
    )
    prompt = (
        f"Recommend the single best flight option for a traveler with cabin preference "
        f"'{cabin_class}' and budget level '{budget_level}', from these real offers. Ground your "
        f"summary ONLY in the fields shown.\n\nOffers:\n{listing}"
    )
    rec = structured(FlightRecommendation, prompt, default=None)
    return rec.summary if rec else None  # no recommendation blurb is fine — the offers still render


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
    # default=None so a malformed LLM response degrades to "no curation" (the caller falls back to
    # the rating-sorted candidates) instead of throwing and taking the whole Logistics Agent —
    # flight search included — down with it.
    curation = structured(LogisticsCuration, prompt, default=None)
    if curation is None:
        return []

    by_name = {h["name"]: h for h in candidates}
    picked = []
    for pick in curation.picks:
        hotel = by_name.get(pick.name)
        if hotel:  # defensive: ignore a name that doesn't match a real candidate
            picked.append({**hotel, "why": pick.reason})
    return picked


def run(
    origin: str,
    cities: list[str],
    destination_country: str,
    start_date: str,
    end_date: str,
    budget_level: str = "Any",
    preferences: dict | None = None,
    cabin_class: str = "Economy",
    stops_preference: str = "Any",
) -> dict:
    """`cities` is every destination city the traveler listed. Flights are priced to the first
    (primary) city; accommodation is searched and curated for EVERY city, returned keyed by city."""
    primary_city = cities[0]
    flight_links = build_flight_links(
        origin, primary_city, date.fromisoformat(start_date), date.fromisoformat(end_date),
        cabin_class, stops_preference,
    )
    origin_airport = resolve_airport(origin)
    destination_airport = resolve_airport(primary_city)
    origin_code, destination_code = origin_airport.iata_code, destination_airport.iata_code

    flight_offers: list[dict] = []
    flight_recommendation: str | None = None
    flight_search_status = "not_attempted"
    last_exc: Exception | None = None
    for _ in range(2):  # retry once before falling back to links-only, same policy as every other tool
        try:
            flight_offers = search_flights_via_mcp(origin_code, destination_code, start_date, end_date, cabin_class)
            for offer in flight_offers:
                offer["search_link"] = build_airline_search_link(
                    origin_code, destination_code, start_date, end_date, cabin_class,
                    airline_iata=offer.get("airline_iata") or "",
                    airline_name=offer["airline"],
                )
            flight_recommendation = _recommend_flight(flight_offers, cabin_class, budget_level)
            flight_search_status = "ok"
            last_exc = None
            break
        except Exception as e:  # noqa: BLE001 - real flight search is best-effort; links are the guaranteed fallback
            last_exc = e
    if last_exc is not None:
        flight_search_status = "not_configured" if "DUFFEL_API_KEY" in str(last_exc) else f"failed: {last_exc}"

    levels = ["$", "$$", "$$$", "$$$$"]
    accommodation: dict[str, list[dict]] = {}
    for city in cities:
        hotels = search_accommodation(city, destination_country)
        if budget_level != "Any":
            max_idx = levels.index(budget_level) if budget_level in levels else len(levels) - 1
            filtered = [h for h in hotels if h["price_level"] is None or levels.index(h["price_level"]) <= max_idx]
            hotels = filtered or hotels  # never over-filter down to nothing
        candidates = sorted(hotels, key=lambda h: (h["rating"] or 0), reverse=True)[:8]
        curated = _curate_hotels(candidates, budget_level, preferences or {})
        accommodation[city] = curated or candidates[:5]

    return {
        "flight_offers": flight_offers,
        "flight_recommendation": flight_recommendation,
        "flight_search_status": flight_search_status,
        "flight_links": flight_links,
        "flight_route": f"{origin_code} → {destination_code}",
        "flight_destination_note": destination_airport.airport_note,  # e.g. "Kyoto has no airport — routes via KIX"
        "accommodation": accommodation,  # {city: [hotels]}
    }
