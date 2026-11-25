"""Rule-based question understanding (labels, dates, communities, users, intent).

Rules first, LLM second: deterministic parsing is testable and cannot hallucinate.
Questions it cannot classify fall back to Text-to-Cypher.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from graph_enhanced_lm.temporal.date_filters import DateRange, day_range, offset_range

LABEL_PATTERNS = [
    (re.compile(r"\bhate(ful)?\b|\bhate[- ]?(speech|labell?ed)\b", re.I), "HATE"),
    (re.compile(r"\bspam(my|-labell?ed)?\b", re.I), "SPAM"),
    (re.compile(r"\boffensive\b|\babusive\b", re.I), "OFFENSIVE"),
    (re.compile(r"community[- ]targeted|\btargeted\b", re.I), "COMMUNITY_TARGETED"),
]
ISO = r"(\d{4}-\d{2}-\d{2})"
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14,
                "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90}
NUM = r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")"


def _to_int(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]


@dataclass
class Intent:
    question: str
    intent: str | None = None
    labels: list[str] = field(default_factory=list)
    date_range: DateRange | None = None
    community_id: int | None = None
    user_id: str | None = None
    notes: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        p: dict[str, Any] = {}
        if self.date_range:
            p.update(self.date_range.as_params())
        if self.labels:
            p["label"] = self.labels[0]
        if self.community_id is not None:
            p["community_id"] = self.community_id
        if self.user_id:
            p["user_id"] = self.user_id
        return p


def parse_date_range(q: str, election_date: str | None, tz: str) -> tuple[DateRange | None, str | None]:
    m = re.search(rf"(?:from|between)\s+{ISO}\s+(?:to|and|until)\s+{ISO}", q, re.I)
    if m:
        return day_range(m.group(1), m.group(2), tz, "explicit"), None
    if election_date:
        m = re.search(NUM + r"\s+days?\s+before\s+(?:the\s+)?(?:configured\s+)?election", q, re.I)
        if m:
            n = _to_int(m.group(1))
            return offset_range(election_date, -n, 0, tz, f"pre_{n}"), None
        m = re.search(NUM + r"\s+days?\s+after\s+(?:the\s+)?(?:configured\s+)?election", q, re.I)
        if m:
            n = _to_int(m.group(1))
            return offset_range(election_date, 1, n + 1, tz, f"post_{n}"), None
        if re.search(r"\belection day\b|\bon the election\b", q, re.I):
            return offset_range(election_date, 0, 1, tz, "election_day"), None
        if re.search(r"\b(before|leading up to|run[- ]up to)\b.*\belection\b", q, re.I):
            return offset_range(election_date, -30, 0, tz, "pre_30"), "Assumed 30 days before election"
        if re.search(r"\bafter\b.*\belection\b", q, re.I):
            return offset_range(election_date, 1, 11, tz, "post_10"), "Assumed 10 days after election"
    elif re.search(r"\belection\b", q, re.I):
        return None, "No election date configured"
    return None, None


def classify_intent(q: str, has_community: bool, has_user: bool, labels: list[str]) -> str | None:
    ql = q.lower()
    if re.search(r"how did .*(change|evolve)|before, during,? and after|over time|trend|increas|changed", ql):
        return "temporal_comparison"
    if has_community and re.search(r"evidence|explain|support|why|posts? (in|from)", ql):
        return "community_evidence"
    if has_community and re.search(r"influen|central|pagerank|structural|important|key users", ql):
        return "top_users_in_community"
    if re.search(r"amplif|repeatedly|re-?shar", ql):
        return "repeated_amplifiers"
    if re.search(r"communit", ql) and (labels or re.search(r"proportion|share|high|most", ql)):
        return "community_stats"
    if has_user and re.search(r"connect|interact|neighbo", ql):
        return "connected_users"
    if re.search(r"most active|posted the most|most posts", ql):
        return "posts_per_user"
    if re.search(r"most retweeted|retweets", ql):
        return "most_retweeted_users"
    if re.search(r"\busers?\b|\baccounts?\b|who posted|which users", ql) and labels:
        return "users_posting_label_in_range"
    return None


def parse_question(question: str, election_date: str | None, tz: str = "UTC") -> Intent:
    labels = [lab for pat, lab in LABEL_PATTERNS if pat.search(question)]
    dr, note = parse_date_range(question, election_date, tz)
    cm = re.search(r"\bcommunity\s*(?:id\s*)?#?(\d+)\b", question, re.I)
    um = re.search(r"\buser\s+([A-Za-z0-9_]+)\b", question, re.I)
    user_id = um.group(1) if um and um.group(1).lower() not in {"id", "ids", "who", "with"} else None
    intent = Intent(question=question, labels=labels, date_range=dr,
                    community_id=int(cm.group(1)) if cm else None, user_id=user_id)
    if note:
        intent.notes.append(note)
    intent.intent = classify_intent(question, intent.community_id is not None, bool(user_id), labels)
    return intent
