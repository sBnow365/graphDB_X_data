"""Explicit, documented mapping from source labels to project labels.

Nothing is mapped implicitly: an unknown source label becomes ``UNKNOWN`` and a
warning is logged, so silent re-labelling cannot happen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class LabelMapper:
    mapping: dict[str, list[str]]
    project_labels: list[str]
    priority: list[str]
    unmapped_seen: set[str] = field(default_factory=set)

    @classmethod
    def from_taxonomy(cls, taxonomy: dict[str, Any], key: str) -> "LabelMapper":
        mappings = taxonomy.get("mappings", {})
        if key not in mappings:
            raise KeyError(f"No label mapping '{key}' in configs/label_taxonomy.yaml")
        raw = mappings[key] or {}
        mapping = {str(k).strip().lower(): list(v) for k, v in raw.items()}
        labels = list(taxonomy["labels"].keys())
        for src, targets in mapping.items():
            bad = [t for t in targets if t not in labels]
            if bad:
                raise ValueError(f"Mapping '{key}': {src} -> {bad} not in taxonomy {labels}")
        return cls(mapping=mapping, project_labels=labels,
                   priority=list(taxonomy.get("primary_label_priority", labels)))

    def map_labels(self, source_labels: Iterable[str | None]) -> list[str]:
        """Map source labels to project labels; unknown values become UNKNOWN."""
        out: list[str] = []
        for lab in source_labels:
            if lab is None:
                continue
            key = str(lab).strip().lower()
            targets = self.mapping.get(key)
            if targets is None:
                if key not in self.unmapped_seen:
                    logger.warning("Unmapped source label '%s' -> UNKNOWN", lab)
                    self.unmapped_seen.add(key)
                targets = ["UNKNOWN"]
            for t in targets:
                if t not in out:
                    out.append(t)
        return out or ["UNKNOWN"]

    def primary(self, labels: Iterable[str]) -> str:
        labels = list(labels)
        for p in self.priority:
            if p in labels:
                return p
        return "UNKNOWN"


def primary_label(labels: Iterable[str], priority: list[str]) -> str:
    """Pick the display label from a multi-label set using the configured priority."""
    labels = list(labels)
    for p in priority:
        if p in labels:
            return p
    return "UNKNOWN"
