"""Timezone-safe date-range helpers.

Timestamps are STORED in UTC. Human-facing boundaries (e.g. "election day") are
defined in a local timezone, then converted to UTC for querying.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DateRange:
    """Half-open interval [start, end) in UTC."""

    start: datetime
    end: datetime
    name: str = ""

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("DateRange boundaries must be timezone-aware")
        if self.end <= self.start:
            raise ValueError(f"Empty or inverted range: {self.start} -> {self.end}")

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400

    def as_params(self) -> dict[str, str]:
        """ISO strings for Cypher ``datetime($start)`` / ``datetime($end)``."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end


def parse_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def local_midnight_utc(day: date, tz: str) -> datetime:
    """Midnight of ``day`` in timezone ``tz``, expressed in UTC."""
    return datetime.combine(day, time(0, 0), tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def day_range(start_day: str | date, end_day_inclusive: str | date, tz: str = "UTC",
              name: str = "") -> DateRange:
    """Range covering whole local days from ``start_day`` to ``end_day_inclusive``."""
    s = local_midnight_utc(parse_date(start_day), tz)
    e = local_midnight_utc(parse_date(end_day_inclusive) + timedelta(days=1), tz)
    return DateRange(s, e, name)


def offset_range(anchor_day: str | date, start_offset: int, end_offset: int,
                 tz: str = "UTC", name: str = "") -> DateRange:
    """Range [anchor+start_offset, anchor+end_offset) in local days, as UTC."""
    anchor = parse_date(anchor_day)
    s = local_midnight_utc(anchor + timedelta(days=start_offset), tz)
    e = local_midnight_utc(anchor + timedelta(days=end_offset), tz)
    return DateRange(s, e, name)
