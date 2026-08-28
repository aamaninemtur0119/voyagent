import json
import re
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from voyagent.config import settings

_llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CREDENTIALS_PATH = PROJECT_ROOT / "google_calendar_credentials.json"
TOKEN_PATH = PROJECT_ROOT / "google_calendar_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Deadline(BaseModel):
    title: str = Field(description="Short, stern, imperative action — a command, e.g. 'Pay ETA fee and apply', not a description")
    date: str = Field(
        description="ISO date YYYY-MM-DD if determinable, otherwise a plain-language estimate with a note on why it's approximate"
    )
    reason: str = Field(description="One blunt line on why this matters, grounded in the retrieved source. No soft explanation.")
    basis: str = Field(
        default="grounded",
        description="'grounded' if this exact date/lead-time is stated in the sources; 'estimated' if the sources only imply "
        "the ordering (e.g. 'apply well in advance') and you computed a reasonable buffer yourself",
    )


class ExtractedDeadlines(BaseModel):
    deadlines: list[Deadline] = Field(
        description="Only deadlines actually grounded in the retrieved sources. Empty list if none apply."
    )


def extract_deadlines(retrieved_chunks: list[dict], situation: dict, travel_start_date: str | None = None) -> ExtractedDeadlines:
    context = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in retrieved_chunks)

    if travel_start_date:
        date_instruction = (
            f"The trip starts on {travel_start_date}. Build a REAL TIMELINE working backward from that date "
            "— do not put every deadline on today's date or on the travel date itself, that defeats the "
            "purpose of a schedule. For each action, figure out how much lead time it actually needs and "
            "place it that far before the travel date:\n"
            "- If the source states an explicit wait time or lead time (e.g. 'processing takes 4-8 weeks', "
            "'must be valid 6 months beyond stay'), compute the exact date from that and set basis='grounded'.\n"
            "- If the source doesn't give a number but the nature of the step implies real lead time is needed "
            "(e.g. visa interview appointments often have long wait lists; document prep takes time), use a "
            "sensible buffer of your own judgment (commonly 6-12 weeks before travel for an interview-based "
            "visa, 2-4 weeks for a fast online ETA/ETIAS) and set basis='estimated' — say so honestly in the "
            "reason, don't present a guessed buffer as if it were a stated fact.\n"
            "- If the source cautions against an action until something else happens first (e.g. 'do not "
            "book tickets until the visa is issued'), extract that as its own deadline: place it AFTER your "
            "estimated visa-approval date but with enough buffer before travel to still get reasonable fares "
            "— do not just skip this kind of conditional guidance because it isn't a fixed date.\n"
            "- Passport renewal or validity-window checks should be computed from the END of the trip "
            f"(start date + duration from the traveler situation), not the start date.\n"
            "Never leave a date as a vague phrase like 'before the trip' when a start date is available — "
            "always resolve it to an actual YYYY-MM-DD date, spread sensibly across the lead-up to travel."
        )
    else:
        date_instruction = (
            "No travel start date was provided, so if a deadline is only expressible relative to travel "
            "dates (not a fixed external date), use a plain-language estimate instead of guessing an ISO date."
        )

    prompt = (
        "Extract time-sensitive action items the traveler must do before their trip — e.g. pay a visa/ETA/"
        "ETIAS fee and get an appointment, submit the application, renew a passport, get a required "
        "vaccination, or timing guidance on booking flights/tickets. Only include items actually grounded "
        "in the sources below; do not invent generic travel advice not stated in the sources. If a fee "
        "applies, the title should say to pay it and get an appointment/apply, not just describe that a fee "
        "exists. Write titles as short, stern, imperative commands ('Pay ETA fee and apply', 'Get visa "
        f"approved before departure') — not descriptions or soft suggestions. Keep the reason to one blunt "
        f"line. {date_instruction}\n\n"
        f"Traveler situation: {situation}\n\nRetrieved sources:\n{context}\n\n"
        "Respond with ONLY a JSON object of this exact shape, no other text: "
        '{"deadlines": [{"title": "...", "date": "...", "reason": "...", "basis": "grounded|estimated"}]}. '
        'Use "deadlines": [] if none apply.'
    )
    response = _llm.invoke(prompt)
    if isinstance(response.content, list):
        text = "".join(block.get("text", "") for block in response.content if isinstance(block, dict))
    else:
        text = response.content
    match = re.search(r"\{.*\}", text, re.DOTALL)
    payload = json.loads(match.group(0)) if match else {"deadlines": []}
    return ExtractedDeadlines(**payload)


def _get_calendar_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def write_to_calendar(deadlines: list[Deadline]) -> dict:
    if not deadlines:
        return {"status": "no_deadlines", "message": "No time-sensitive deadlines were identified.", "deadlines": []}

    if not CREDENTIALS_PATH.exists():
        return {
            "status": "not_configured",
            "message": (
                "Google Calendar isn't connected for this agent yet — no credentials are configured at "
                f"{CREDENTIALS_PATH.name}. The following deadlines were identified and would be written "
                "once calendar access is set up:"
            ),
            "deadlines": [d.model_dump() for d in deadlines],
        }

    service = _get_calendar_service()
    created, skipped = [], []
    for d in deadlines:
        if not ISO_DATE.match(d.date):
            skipped.append({"title": d.title, "date": d.date, "reason": "date not in a clean YYYY-MM-DD format"})
            continue
        event = {
            "summary": d.title,
            "description": d.reason,
            "start": {"date": d.date},
            "end": {"date": d.date},
        }
        created_event = service.events().insert(calendarId="primary", body=event).execute()
        created.append({"title": d.title, "date": d.date, "link": created_event.get("htmlLink")})

    return {
        "status": "written" if created else "no_valid_dates",
        "message": f"Created {len(created)} calendar event(s) on your primary Google Calendar.",
        "created": created,
        "skipped": skipped,
        "deadlines": [d.model_dump() for d in deadlines],
    }
