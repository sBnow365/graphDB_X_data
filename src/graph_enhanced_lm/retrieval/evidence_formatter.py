"""Turn graph rows into numbered, citable evidence for the LLM.

* Every item gets a stable id E1..En and keeps the record identifiers it came from.
* Items are marked (computed) for database statistics vs (record) for raw records.
* Post text is truncated (privacy + prompt length).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ID_KEYS = ("post_id", "user_id", "community", "amplifier", "author", "connected_user_id", "window")
TEXT_KEYS = ("text",)


@dataclass
class EvidenceItem:
    eid: str
    kind: str                     # 'computed' | 'record'
    content: str
    record_ids: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    items: list[EvidenceItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source_query: str | None = None
    truncated: bool = False

    def is_empty(self) -> bool:
        return not self.items

    def ids(self) -> list[str]:
        return [i.eid for i in self.items]

    def to_prompt_text(self) -> str:
        lines = [f"[{i.eid}] ({i.kind}) {i.content}" for i in self.items]
        if self.truncated:
            lines.append("(Evidence list truncated.)")
        lines += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, Any]]:
        return [{"id": i.eid, "kind": i.kind, "content": i.content, **i.record_ids} for i in self.items]


def _fmt_value(key: str, value: Any, max_chars: int) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        shown = ", ".join(map(str, value[:5]))
        return f"[{shown}{', ...' if len(value) > 5 else ''}]"
    text = str(value)
    if key in TEXT_KEYS and len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def format_rows(rows: list[dict[str, Any]], kind: str = "record", max_items: int = 40,
                max_text_chars: int = 160, start_index: int = 1, prefix: str = "") -> Evidence:
    """Format result rows into an :class:`Evidence` object."""
    ev = Evidence(truncated=len(rows) > max_items)
    for n, row in enumerate(rows[:max_items], start=start_index):
        parts = [f"{k}={_fmt_value(k, v, max_text_chars)}" for k, v in row.items() if v is not None]
        ev.items.append(EvidenceItem(eid=f"E{n}", kind=kind, content=(prefix + "; ".join(parts)),
                                     record_ids={k: row[k] for k in ID_KEYS if k in row}))
    return ev


def merge_evidence(*parts: Evidence) -> Evidence:
    """Concatenate evidence blocks and renumber ids sequentially."""
    out = Evidence()
    n = 1
    for part in parts:
        for item in part.items:
            out.items.append(EvidenceItem(f"E{n}", item.kind, item.content, item.record_ids))
            n += 1
        out.notes += part.notes
        out.truncated = out.truncated or part.truncated
    return out
