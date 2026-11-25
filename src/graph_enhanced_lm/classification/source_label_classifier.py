"""'Classifier' that re-uses labels already present in the dataset.

Used for the synthetic data (ground truth) and for human-annotated datasets.
It records where labels came from so they are never confused with model output.
"""
from __future__ import annotations

from graph_enhanced_lm.classification.base import BaseClassifier, Prediction
from graph_enhanced_lm.classification.label_mapping import primary_label
from graph_enhanced_lm.data.schemas import split_labels


class SourceLabelClassifier(BaseClassifier):
    name = "source_labels"
    version = "1"

    def __init__(self, priority: list[str], label_source: str = "human"):
        self.priority = priority
        self.label_source = label_source

    def predict(self, post_ids, texts, source_labels=None):
        source_labels = source_labels or [None] * len(post_ids)
        preds = []
        for pid, labs in zip(post_ids, source_labels):
            labels = split_labels(labs)
            preds.append(Prediction(
                post_id=pid, labels=labels or ["UNKNOWN"],
                primary_label=primary_label(labels, self.priority) if labels else "UNKNOWN",
                confidence=None,
                label_source=self.label_source if labels else "unlabeled",
                model_name=self.name, model_version=self.version))
        return preds
