"""Configurable election windows (before / election day / after)."""
from __future__ import annotations

from typing import Any

from graph_enhanced_lm.temporal.date_filters import DateRange, offset_range


def build_windows(temporal_cfg: dict[str, Any], election_date: str | None = None) -> list[DateRange]:
    """Create the windows listed in configs/temporal.yaml.

    ``election_date`` overrides the config value (e.g. from a CLI flag). The date must
    come from the user or a verified source; it is never assumed.
    """
    e_date = election_date or temporal_cfg.get("election_date")
    if not e_date:
        raise ValueError("No election date configured (configs/temporal.yaml or --election-date)")
    tz = temporal_cfg.get("timezone", "UTC")
    windows = [offset_range(e_date, int(w["start_offset"]), int(w["end_offset"]), tz, w["name"])
               for w in temporal_cfg["windows"]]
    names = [w.name for w in windows]
    if len(set(names)) != len(names):
        raise ValueError(f"Window names must be unique: {names}")
    return windows


def nearest_pre_windows(windows: list[DateRange]) -> tuple[DateRange, DateRange] | None:
    """Return (far, near) - the two pre-election windows closest to election day.

    Used for the 'pre_election_growth' component of community ranking.
    """
    pre = sorted([w for w in windows if w.name.startswith("pre")], key=lambda w: w.end)
    if len(pre) < 2:
        return None
    return pre[-2], pre[-1]
