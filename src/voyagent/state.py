"""Shared state schema for the trip-planning graph. Plain fields are last-write-wins (each is
written by exactly one node); agent_trace and errors use an additive reducer since multiple nodes
append to them over the course of a run and we want the full history, not just the last entry."""

import operator
from typing import Annotated, TypedDict


class TraceEntry(TypedDict):
    agent: str
    status: str  # "running" | "done" | "retrying" | "failed" | "skipped"
    detail: str


class TripState(TypedDict, total=False):
    # --- input, set once at graph start ---
    nationality: str
    destination_country: str
    destination_city: str  # primary city — drives the actual Logistics/Experience agent calls
    destination_cities: list[str]  # all cities the traveler mentioned, including destination_city;
    # length > 1 means they've already indicated a multi-city plan, which suppresses the
    # "consider more than one city" recommendation nudge in synthesis
    origin: str
    purpose: str
    duration: str
    start_date: str | None  # ISO date, optional
    end_date: str | None
    preferences: dict  # dietary, family_friendly, outdoor_seating, budget_level

    # --- accumulated across nodes ---
    eligibility: dict | None
    deadlines: list[dict] | None
    logistics: dict | None
    experience: dict | None
    itinerary: str | None

    # --- control flow / observability (what the UI's agent-trace panel renders) ---
    agent_trace: Annotated[list[TraceEntry], operator.add]
    errors: Annotated[list[str], operator.add]

    # --- human-in-the-loop: two independent write actions, each gated separately ---
    calendar_approved: bool | None
    calendar_result: dict | None
    export_approved: bool | None
    export_result: dict | None

    # --- adaptive replanning: a rejection with feedback loops back to Logistics/Experience
    # rather than just stopping, capped to prevent an infinite loop ---
    replan_requested: bool
    replan_count: int
