"""Classifier interface. Any classifier must return :class:`Prediction` objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Prediction:
    post_id: str
    labels: list[str]
    primary_label: str
    confidence: float | None
    label_source: str            # 'human' | 'model' | 'synthetic_ground_truth' | 'unlabeled'
    model_name: str
    model_version: str
    scores: dict[str, float] = field(default_factory=dict)


class BaseClassifier(ABC):
    """Replaceable classifier: graph code only depends on this interface."""

    name: str = "base"
    version: str = "0"

    @abstractmethod
    def predict(self, post_ids: list[str], texts: list[str | None],
                source_labels: list[str | None] | None = None) -> list[Prediction]:
        """Classify a batch of posts."""
