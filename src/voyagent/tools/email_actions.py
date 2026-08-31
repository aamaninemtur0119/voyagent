"""Email the finished itinerary to the traveler — the graph's second independent write action,
gated behind its own human approval (separate from the Google Calendar gate).

Contract mirrors write_to_calendar in calendar_actions.py:
- Missing configuration is NOT an error: returns {"status": "not_configured", ...} so the UI can
  say "not connected yet" rather than "the send broke". Same for a missing recipient
  ({"status": "skipped"}).
- A genuine SMTP failure (auth rejected, connection refused, timeout) is raised, so the calling
  node's retry-once-then-degrade policy handles it the same way it handles every other tool.

Plain stdlib smtplib + STARTTLS, so it works against any standard SMTP provider (a Gmail app
password, SendGrid, Mailgun, Postmark, ...) without pulling in a vendor SDK.
"""

import smtplib
import ssl
from email.message import EmailMessage

from voyagent.config import settings


def _configured() -> bool:
    return bool(settings.smtp_host and settings.smtp_username and settings.smtp_password)


def send_itinerary_email(to_address: str, subject: str, body: str) -> dict:
    to_address = (to_address or "").strip()
    if not to_address:
        return {"status": "skipped", "message": "No email address was provided, so there is nothing to send."}

    if not _configured():
        return {
            "status": "not_configured",
            "to": to_address,
            "message": f"No SMTP account is configured, so the itinerary wasn't sent to {to_address}. Add SMTP_* to .env to enable it.",
        }

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from or settings.smtp_username
    msg["To"] = to_address
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
        server.starttls(context=context)
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(msg)

    return {"status": "sent", "to": to_address, "message": f"Itinerary emailed to {to_address}."}
