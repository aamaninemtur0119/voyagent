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
    destination_city: str
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

    # --- human-in-the-loop, only around the one real write action ---
    calendar_approved: bool | None
    calendar_result: dict | None
