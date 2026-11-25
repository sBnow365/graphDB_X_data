import pytest

from graph_enhanced_lm.classification.label_mapping import LabelMapper
from graph_enhanced_lm.classification.source_label_classifier import SourceLabelClassifier


def test_hatexplain_mapping(cfg):
    m = LabelMapper.from_taxonomy(cfg.label_taxonomy, "hatexplain")
    assert m.map_labels(["hatespeech"]) == ["HATE"]
    assert m.map_labels(["offensive"]) == ["OFFENSIVE"]
    assert m.map_labels(["normal"]) == ["NORMAL"]


def test_unmapped_becomes_unknown(cfg):
    m = LabelMapper.from_taxonomy(cfg.label_taxonomy, "hasoc")
    assert m.map_labels(["SOMETHING_NEW"]) == ["UNKNOWN"]
    assert "something_new" in m.unmapped_seen


def test_hasoc_fine_labels(cfg):
    m = LabelMapper.from_taxonomy(cfg.label_taxonomy, "hasoc")
    assert m.map_labels(["HATE"]) == ["HATE"]
    assert m.map_labels(["PRFN"]) == ["OFFENSIVE"]
    assert m.map_labels(["NOT"]) == ["NORMAL"]


def test_primary_priority(cfg):
    m = LabelMapper.from_taxonomy(cfg.label_taxonomy, "synthetic")
    assert m.primary(["COMMUNITY_TARGETED", "HATE"]) == "HATE"
    assert m.primary([]) == "UNKNOWN"


def test_bad_mapping_rejected(cfg):
    tax = dict(cfg.label_taxonomy)
    tax["mappings"] = {"x": {"a": ["NOT_A_LABEL"]}}
    with pytest.raises(ValueError):
        LabelMapper.from_taxonomy(tax, "x")


def test_source_label_classifier_provenance(cfg):
    clf = SourceLabelClassifier(cfg.label_taxonomy["primary_label_priority"], "synthetic_ground_truth")
    preds = clf.predict(["p1", "p2"], ["t", "t"], ["HATE;COMMUNITY_TARGETED", None])
    assert preds[0].primary_label == "HATE" and preds[0].label_source == "synthetic_ground_truth"
    assert preds[1].labels == ["UNKNOWN"] and preds[1].label_source == "unlabeled"
