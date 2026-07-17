from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


CALENDAR_READONLY_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
SCOPES = [CALENDAR_READONLY_SCOPE]


def _google_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Calendar dependencies are missing. Run: pip install -e .[bot]"
        ) from exc
    return Request, Credentials, InstalledAppFlow, build


def authorize_google_calendar(credentials_file: Path, token_file: Path) -> Path:
    """Open Google's desktop OAuth flow and save a reusable read-only token."""
    Request, Credentials, InstalledAppFlow, _ = _google_imports()
    if not credentials_file.is_file():
        raise FileNotFoundError(f"Google OAuth client file is missing: {credentials_file}")

    credentials = None
    if token_file.is_file():
        credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
            credentials = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(credentials.to_json(), encoding="utf-8")
    return token_file


def _load_credentials(credentials_file: Path, token_file: Path) -> Any:
    Request, Credentials, _, _ = _google_imports()
    if not credentials_file.is_file():
        raise FileNotFoundError(f"Google OAuth client file is missing: {credentials_file}")
    if not token_file.is_file():
        raise FileNotFoundError(
            "Google Calendar is not authorized. Run: personal-ai authorize-calendar"
        )
    credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        token_file.write_text(credentials.to_json(), encoding="utf-8")
    if not credentials.valid:
        raise RuntimeError(
            "Google Calendar authorization is invalid. Delete the token and run "
            "personal-ai authorize-calendar again."
        )
    return credentials


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 datetime: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Calendar datetime must include a timezone: {value}")
    return parsed


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_time(value: dict[str, Any]) -> str:
    return str(value.get("dateTime") or value.get("date") or "")


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    merged: list[list[datetime]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def _local_interval(
    start: datetime,
    end: datetime,
    local_zone: ZoneInfo,
) -> dict[str, str]:
    return {
        "start": start.astimezone(local_zone).isoformat(),
        "end": end.astimezone(local_zone).isoformat(),
    }


def _query_events(
    service: Any,
    calendar_ids: list[str],
    start: datetime,
    end: datetime,
    query: str,
    limit: int,
    time_zone: str,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for calendar_id in calendar_ids:
        parameters: dict[str, Any] = {
            "calendarId": calendar_id,
            "timeMin": _rfc3339(start),
            "timeMax": _rfc3339(end),
            "singleEvents": True,
            "orderBy": "startTime",
            "maxResults": limit,
            "timeZone": time_zone,
        }
        if query:
            parameters["q"] = query
        response = service.events().list(**parameters).execute()
        for event in response.get("items", []):
            if event.get("status") == "cancelled":
                continue
            events.append(
                {
                    "calendar_id": calendar_id,
                    "summary": event.get("summary", "(без названия)"),
                    "start": _event_time(event.get("start", {})),
                    "end": _event_time(event.get("end", {})),
                    "location": event.get("location", ""),
                    "description": str(event.get("description", ""))[:500],
                }
            )
    events.sort(key=lambda event: event["start"])
    return {
        "action": "events",
        "time_zone": time_zone,
        "events": events[:limit],
    }


def _query_free_time(
    service: Any,
    calendar_ids: list[str],
    start: datetime,
    end: datetime,
    minimum_free_minutes: int,
    time_zone: str,
) -> dict[str, Any]:
    response = (
        service.freebusy()
        .query(
            body={
                "timeMin": _rfc3339(start),
                "timeMax": _rfc3339(end),
                "timeZone": time_zone,
                "items": [{"id": calendar_id} for calendar_id in calendar_ids],
            }
        )
        .execute()
    )
    intervals: list[tuple[datetime, datetime]] = []
    errors: list[dict[str, Any]] = []
    for calendar_id, calendar in response.get("calendars", {}).items():
        errors.extend(calendar.get("errors", []))
        for busy in calendar.get("busy", []):
            busy_start = max(start, _parse_datetime(busy["start"]))
            busy_end = min(end, _parse_datetime(busy["end"]))
            if busy_end > busy_start:
                intervals.append((busy_start, busy_end))
    if errors:
        raise RuntimeError(f"Google Calendar free/busy query failed: {errors}")

    merged = _merge_intervals(intervals)
    free: list[tuple[datetime, datetime]] = []
    cursor = start
    minimum = timedelta(minutes=max(1, min(minimum_free_minutes, 1440)))
    for busy_start, busy_end in merged:
        if busy_start - cursor >= minimum:
            free.append((cursor, busy_start))
        cursor = max(cursor, busy_end)
    if end - cursor >= minimum:
        free.append((cursor, end))

    local_zone = ZoneInfo(time_zone)
    return {
        "action": "free_time",
        "time_zone": time_zone,
        "busy": [_local_interval(a, b, local_zone) for a, b in merged],
        "free": [_local_interval(a, b, local_zone) for a, b in free],
    }


def query_google_calendar(
    credentials_file: Path,
    token_file: Path,
    calendar_ids: list[str],
    action: str,
    start: str,
    end: str,
    *,
    query: str = "",
    minimum_free_minutes: int = 30,
    limit: int = 20,
    time_zone: str = "Europe/Kyiv",
) -> dict[str, Any]:
    """Read events or availability from the authorized Google calendars."""
    start_at, end_at = _parse_datetime(start), _parse_datetime(end)
    if end_at <= start_at:
        raise ValueError("Calendar end must be later than start")
    if not calendar_ids:
        raise ValueError("At least one Google Calendar ID is required")
    _, _, _, build = _google_imports()
    service = build(
        "calendar",
        "v3",
        credentials=_load_credentials(credentials_file, token_file),
        cache_discovery=False,
    )
    bounded_limit = max(1, min(int(limit), 50))
    if action == "events":
        return _query_events(
            service,
            calendar_ids,
            start_at,
            end_at,
            query.strip(),
            bounded_limit,
            time_zone,
        )
    if action == "free_time":
        return _query_free_time(
            service,
            calendar_ids,
            start_at,
            end_at,
            int(minimum_free_minutes),
            time_zone,
        )
    raise ValueError("Calendar action must be 'events' or 'free_time'")
