"""Template-first graph retrieval for a parsed :class:`Intent`."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from graph_enhanced_lm.gds.statistics import fetch_community_stats, score_communities
from graph_enhanced_lm.graph.cypher_queries import get_template
from graph_enhanced_lm.retrieval.intent_parser import Intent
from graph_enhanced_lm.temporal.election_windows import build_windows
from graph_enhanced_lm.temporal.trend_analysis import (
    CAUSALITY_NOTE, community_window_activity_neo4j, window_comparison_neo4j,
)

logger = logging.getLogger(__name__)
WIDE_RANGE = {"start": "1970-01-01T00:00:00+00:00", "end": "2100-01-01T00:00:00+00:00"}


@dataclass
class RetrievalResult:
    method: str                              # 'template:<name>' | 'analysis:<name>' | 'none'
    rows: list[dict[str, Any]] = field(default_factory=list)
    cypher: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    computed: bool = False
    notes: list[str] = field(default_factory=list)


class GraphRetriever:
    def __init__(self, client: Any, community_property: str, temporal_cfg: dict[str, Any],
                 priority_weights: dict[str, float], min_users: int = 5, limit: int = 50):
        self.client = client
        self.prop = community_property
        self.temporal_cfg = temporal_cfg
        self.weights = priority_weights
        self.min_users = min_users
        self.limit = limit

    def _template(self, name: str, params: dict[str, Any], computed: bool = False) -> RetrievalResult:
        t = get_template(name)
        p = {k: params.get(k) for k in t.params}
        if "limit" in t.params:
            p["limit"] = params.get("limit", self.limit)
        rows = self.client.run_read(t.cypher, p)
        return RetrievalResult(method=f"template:{name}", rows=rows, cypher=t.cypher, params=p,
                               computed=computed)

    def retrieve(self, intent: Intent) -> RetrievalResult | None:
        """Return results for known intents, or None so the caller can use Text-to-Cypher."""
        p = {**WIDE_RANGE, **intent.params(), "community_property": self.prop,
             "min_users": self.min_users, "limit": self.limit, "min_count": 2,
             "score_property": "pagerank"}
        p.setdefault("label", None)
        kind = intent.intent
        if kind == "users_posting_label_in_range":
            return self._template(kind, p)
        if kind in ("posts_per_user", "most_retweeted_users", "repeated_amplifiers",
                    "connected_users"):
            return self._template(kind, p, computed=kind != "connected_users")
        if kind == "top_users_in_community":
            return self._template(kind, p, computed=True)
        if kind == "community_evidence":
            if intent.date_range is None:
                p["start"] = p["end"] = None
            return self._template(kind, p)
        if kind == "community_stats":
            stats = fetch_community_stats(self.client, self.prop, self.min_users)
            if stats.empty:
                return RetrievalResult(method="analysis:community_stats", computed=True,
                                       notes=["No community property found: run scripts/run_graph_algorithms.py first."])
            scored = score_communities(stats, self.weights)
            keep = ["community", "users", "posts", "hate_posts", "spam_posts", "offensive_posts",
                    "hate_share", "spam_share", "hateful_users", "annotated_users", "hateful_user_share",
                    "internal_ratio", "priority_score", "priority_rank"]
            rows = scored[[c for c in keep if c in scored.columns]].head(self.limit).to_dict("records")
            return RetrievalResult(method="analysis:community_stats", rows=rows, computed=True,
                                   notes=["Priority score weights are analytical choices (configs/gds.yaml)."])
        if kind == "temporal_comparison":
            windows = build_windows(self.temporal_cfg)
            overall = window_comparison_neo4j(self.client, windows)
            rows = overall.to_dict("records")
            notes = [CAUSALITY_NOTE]
            comm = community_window_activity_neo4j(self.client, windows, self.prop)
            if not comm.empty:
                pivot = comm.pivot_table(index="community", columns="window",
                                         values="hate_posts", fill_value=0).reset_index()
                pivot.columns = [str(c) if c == "community" else f"hate_posts_{c}" for c in pivot.columns]
                rows += pivot.head(self.limit).to_dict("records")
            return RetrievalResult(method="analysis:temporal_comparison", rows=rows, computed=True,
                                   notes=notes)
        return None
