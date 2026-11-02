"""
Generic Gmail retrieval utilities.

Works for any use case: submission responses, Evite invitations, Paperless Post,
calendar-related emails, etc. Nothing here is domain-specific.

Usage:
    from gmail_tools import get_service, build_query, fetch_messages, message_text
    from gmail_tools.auth import get_creds

    service = get_service(token_path=..., credentials_path=...)
    msgs = fetch_messages(service, build_query(domains=["evite.com"], after_date="2026-01-01"))
    for m in msgs:
        subject, body = message_text(service, m["id"])
"""

import base64
import html as html_lib
import re
from pathlib import Path

from googleapiclient.discovery import build

from .auth import get_creds

GMAIL_READONLY = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_service(
    token_path: Path,
    credentials_path: Path,
    scopes: list[str] = GMAIL_READONLY,
):
    """Return an authenticated Gmail API service."""
    creds = get_creds(scopes=scopes, token_path=token_path, credentials_path=credentials_path)
    return build("gmail", "v1", credentials=creds)


def build_query(
    domains: list[str] | None = None,
    after_date: str | None = None,
    subject_contains: str | None = None,
    extra: str | None = None,
) -> str:
    """
    Build a Gmail search query string.

    domains: sender domains to match (partial match on from: field)
    after_date: ISO date string YYYY-MM-DD — only emails after this date
    subject_contains: keyword that must appear in subject
    extra: any additional raw Gmail search clause to append
    """
    parts = []
    if domains:
        domain_clauses = " OR ".join(f"from:{d}" for d in domains)
        parts.append(f"({domain_clauses})")
    if after_date:
        parts.append(f"after:{after_date.replace('-', '/')}")
    if subject_contains:
        parts.append(f"subject:{subject_contains}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def fetch_messages(service, query: str, max_results: int = 50) -> list[dict]:
    """
    Return a list of message metadata dicts matching the query.
    Each dict has at minimum an 'id' key — pass to message_text() to get content.
    """
    result = service.users().messages().list(
        userId="me", q=query, maxResults=max_results
    ).execute()
    return result.get("messages", [])


def message_text(service, msg_id: str, body_limit: int = 3000) -> tuple[str, str, str]:
    """
    Fetch a message and return (subject, body, received_date).
    received_date: ISO YYYY-MM-DD from Gmail's internalDate (when the email arrived).
    body_limit: max characters of body to return (keeps LLM calls cheap).
    """
    from datetime import datetime, timezone
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    subject = next(
        (h["value"] for h in msg["payload"].get("headers", []) if h["name"].lower() == "subject"),
        "",
    )
    body = _extract_body(msg["payload"])
    received_date = datetime.fromtimestamp(
        int(msg["internalDate"]) / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d")
    return subject, body[:body_limit], received_date


def message_full(service, msg_id: str, body_limit: int = 3000, sender_name: str | None = None) -> dict:
    """
    Like message_text(), but also returns From/To/Cc — added for use cases that need
    who a message was with, not just its content (e.g. classifying sent-mail recipients
    into relationship categories). A separate function, not an extended message_text(),
    so existing callers that unpack message_text()'s 3-tuple (e.g. poetry-tools'
    sync_emails.py) are untouched.

    sender_name: passed straight to strip_signature() — the account holder's display
    name, to cut their own signature block out of the body. Optional; omit to leave
    bodies as-is (just quote-stripped).

    Returns: {subject, body, date, from, to: [...], cc: [...]}
    date: ISO YYYY-MM-DD from Gmail's internalDate.
    """
    from datetime import datetime, timezone
    msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}

    def _addrs(header_value: str) -> list[str]:
        if not header_value:
            return []
        return [a.strip() for a in re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", header_value)]

    body = strip_signature(strip_quoted(_extract_body(msg["payload"])), sender_name=sender_name)
    date = datetime.fromtimestamp(
        int(msg["internalDate"]) / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d")
    return {
        "subject": headers.get("subject", ""),
        "body": body[:body_limit],
        "date": date,
        "from": _addrs(headers.get("from", "")),
        "to": _addrs(headers.get("to", "")),
        "cc": _addrs(headers.get("cc", "")),
    }


def send_email(service, to: str, subject: str, body_html: str) -> str:
    """
    Send an HTML email from the authenticated Gmail account.
    Requires gmail.compose or gmail.send scope.
    Returns the sent message ID.
    """
    import base64
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["To"] = to
    msg.attach(MIMEText(body_html, "html"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return result["id"]


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text from a Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")

    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _extract_body(part)
            if text:
                return text

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            raw_html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            # Block-level tags become newlines *before* every other tag is
            # flattened to a space — otherwise a whole message (quoted
            # history included) collapses onto one line, and anything
            # depending on line-anchored matching (strip_quoted's "wrote:"
            # marker) can never find it. A mobile Mail.app reply, which is
            # often HTML-only, was going through unstripped because of this.
            raw_html = re.sub(r"<(br|/div|/p|/li|/tr)\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", raw_html)
            return html_lib.unescape(text)

    return ""


_QUOTE_MARKERS = re.compile(
    r"""
    ^On\ .{0,80}\ wrote:\s*$            |  # "On Mon, Jan 5, 2026 ... X wrote:"
    ^-{2,}\s*Original\ Message\s*-{2,}  |  # "----- Original Message -----"
    ^From:\s.*$                            # a quoted header block starting a forward
    """,
    re.VERBOSE | re.MULTILINE | re.IGNORECASE,
)


def strip_quoted(body: str) -> str:
    """
    Cut a message body at the first quoted-reply marker, keeping only the
    new text someone actually typed. A reply's plain-text body normally
    includes the entire prior thread below the new text ("On ... wrote:"
    plus '>'-prefixed history) — without stripping that, a context window
    or char limit mostly captures old quoted messages, not what was
    actually written this time.
    """
    m = _QUOTE_MARKERS.search(body)
    cut = body[: m.start()] if m else body
    # Drop any trailing '>'-quoted lines the marker regex didn't catch
    lines = [ln for ln in cut.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines).strip()


def strip_signature(body: str, sender_name: str | list[str] | None = None) -> str:
    """
    Cut a message body at the start of the sender's own signature block —
    most mail clients insert it right after a line containing just the
    sender's name. Without this, a signature (title, links, bio) gets read
    as if it were context about the *recipient*, not a fixed block that
    appears identically in every message regardless of who it's to.

    sender_name: the account holder's display name(s), exactly as they
    appear on their own line above a signature — a single string, or a
    list when someone has more than one signature variant (e.g. a full
    "Jane Smith" one and a shorter "Jane" one; real-world accounts often
    have both, and missing either lets that variant's signature leak
    through unstripped). Matches whichever variant's marker comes first.
    Without this, there's nothing generic enough to detect a signature
    by, so the body is returned unchanged — this function doesn't guess
    whose signature it's looking for.
    """
    if not sender_name:
        return body
    names = [sender_name] if isinstance(sender_name, str) else sender_name
    earliest = None
    for name in names:
        marker = re.compile(rf"^{re.escape(name)}\s*$", re.MULTILINE)
        m = marker.search(body)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    return body[:earliest].rstrip() if earliest is not None else body
