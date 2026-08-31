"""MCP server exposing real flight search as a tool, over stdio. Run as a subprocess by the
Logistics Agent's MCP client (see agents/logistics.py) — this is the one tool in Pack Your Bags exposed
via the Model Context Protocol rather than a direct Python import, since it's new capability being
added specifically to demonstrate that pattern, not a retrofit of the already-tested tools.
"""

from mcp.server.fastmcp import FastMCP

from pack_your_bags.tools.duffel import search_flights as _search_flights_impl

mcp = FastMCP("pack-your-bags-flights")


@mcp.tool()
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    cabin_class: str = "Economy",
    max_results: int = 5,
) -> list[dict]:
    """Search real flight offers via Duffel (test/sandbox mode). origin/destination are IATA
    airport codes (e.g. JFK, NRT). Dates are YYYY-MM-DD. cabin_class one of: Economy, Premium
    Economy, Business, First. Returns a list of real offers with price, currency, airline, stop
    count, and duration, cheapest first — raises if Duffel isn't configured or the request fails.
    Note: sandbox prices, not live market fares."""
    return _search_flights_impl(origin, destination, departure_date, return_date, cabin_class, max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
