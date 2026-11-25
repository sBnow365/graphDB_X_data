"""Hugging Face Transformer classifier wrapper (batch inference + caching).

The checkpoint's own label names (``config.id2label``) are checked against the
documented mapping in ``label_taxonomy.yaml`` BEFORE any prediction is made.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from graph_enhanced_lm.classification.base import BaseClassifier, Prediction
from graph_enhanced_lm.classification.label_mapping import LabelMapper
from graph_enhanced_lm.config import PROJECT_ROOT
from graph_enhanced_lm.utils.errors import ConfigError

logger = logging.getLogger(__name__)


class TransformerClassifier(BaseClassifier):
    def __init__(self, cfg: dict[str, Any], mapper: LabelMapper):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError("Install torch + requirements-classifier.txt to use the "
                              "transformer backend") from exc
        self.torch = torch
        self.cfg = cfg
        self.mapper = mapper
        self.name = cfg["model_name"]
        device = cfg.get("device", "auto")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        logger.info("Loading classifier %s on %s", self.name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name).to(self.device)
        self.model.eval()
        self.version = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        unmapped = [v for v in self.id2label.values() if v.strip().lower() not in mapper.mapping]
        if unmapped:
            raise ConfigError(
                f"Model labels {unmapped} have no mapping in label_taxonomy.yaml "
                f"(key '{cfg['label_mapping_key']}'). Model labels: {self.id2label}. "
                "Inspect the model card and add an explicit mapping.")
        self.cache_dir = PROJECT_ROOT / cfg.get("cache_dir", ".cache/predictions")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.name}|{self.cfg['max_length']}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def predict(self, post_ids, texts, source_labels=None):
        bs = int(self.cfg.get("batch_size", 32))
        min_conf = float(self.cfg.get("min_confidence", 0.0))
        results: list[Prediction] = []
        for i in range(0, len(texts), bs):
            ids, batch = post_ids[i:i + bs], [t or "" for t in texts[i:i + bs]]
            todo = [j for j, t in enumerate(batch) if not self._cache_path(t).exists()]
            if todo:
                enc = self.tokenizer([batch[j] for j in todo], truncation=True, padding=True,
                                     max_length=int(self.cfg["max_length"]),
                                     return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
                for j, p in zip(todo, probs):
                    scores = {self.id2label[k]: float(v) for k, v in enumerate(p)}
                    self._cache_path(batch[j]).write_text(json.dumps(scores), encoding="utf-8")
            for pid, text in zip(ids, batch):
                scores = json.loads(self._cache_path(text).read_text(encoding="utf-8"))
                top_label, conf = max(scores.items(), key=lambda kv: kv[1])
                labels = self.mapper.map_labels([top_label]) if conf >= min_conf else ["UNKNOWN"]
                results.append(Prediction(
                    post_id=pid, labels=labels, primary_label=self.mapper.primary(labels),
                    confidence=round(conf, 4), label_source="model",
                    model_name=self.name, model_version=self.version, scores=scores))
        return results
