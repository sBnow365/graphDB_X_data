"""Build the configured classifier."""
from __future__ import annotations

from typing import Any

from graph_enhanced_lm.classification.base import BaseClassifier
from graph_enhanced_lm.classification.label_mapping import LabelMapper
from graph_enhanced_lm.classification.source_label_classifier import SourceLabelClassifier


def build_classifier(classifier_cfg: dict[str, Any], taxonomy: dict[str, Any],
                     dataset_source: str = "synthetic") -> BaseClassifier:
    backend = classifier_cfg.get("backend", "source_labels")
    priority = taxonomy.get("primary_label_priority", [])
    if backend == "source_labels":
        source = "synthetic_ground_truth" if dataset_source == "synthetic" else "human"
        return SourceLabelClassifier(priority, label_source=source)
    if backend == "transformer":
        from graph_enhanced_lm.classification.transformer_classifier import TransformerClassifier

        tcfg = classifier_cfg["transformer"]
        mapper = LabelMapper.from_taxonomy(taxonomy, tcfg["label_mapping_key"])
        return TransformerClassifier(tcfg, mapper)
    raise ValueError(f"Unknown classifier backend '{backend}'")
