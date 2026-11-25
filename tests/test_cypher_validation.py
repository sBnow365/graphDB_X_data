import yaml

from graph_enhanced_lm.config import PROJECT_ROOT
from graph_enhanced_lm.evaluation.benchmark import run_safety_suite
from graph_enhanced_lm.llm.query_validation import CypherValidator, extract_cypher, mask_literals_and_comments


def test_safety_suite_all_correct():
    items = yaml.safe_load((PROJECT_ROOT / "benchmarks/cypher_safety_suite.yaml").read_text(encoding="utf-8"))["items"]
    res = run_safety_suite(items, CypherValidator(200))
    wrong = [r for r in res["items"] if not r["correct"]]
    assert not wrong, wrong


def test_limit_appended_and_capped():
    v = CypherValidator(max_rows=50)
    assert v.validate("MATCH (u:User) RETURN u").cypher.endswith("LIMIT 50")
    assert v.validate("MATCH (u:User) RETURN u LIMIT 999").cypher.endswith("LIMIT 50")
    assert v.validate("MATCH (u:User) RETURN u LIMIT 10").cypher.endswith("LIMIT 10")
    assert v.validate("MATCH (u:User) RETURN u LIMIT $limit").ok


def test_params_are_not_keywords():
    r = CypherValidator().validate("MATCH (p:Post) WHERE p.timestamp >= datetime($start) RETURN p",
                                   {"start": "2025-01-01T00:00:00Z"})
    assert r.ok, r.errors


def test_trailing_semicolon_ok():
    assert CypherValidator().validate("MATCH (u:User) RETURN u;").ok


def test_unsafe_param_type():
    r = CypherValidator().validate("MATCH (u:User {user_id: $x}) RETURN u", {"x": {"nested": 1}})
    assert not r.ok


def test_mask_keeps_strings_out_of_keyword_check():
    clean, masked = mask_literals_and_comments("MATCH (p) WHERE p.text = 'DROP it' RETURN p // DELETE")
    assert "DELETE" not in clean and "DROP" in clean and "DROP" not in masked


def test_extract_cypher():
    assert extract_cypher("Sure!\n```cypher\nMATCH (n) RETURN n\n```\nDone") == "MATCH (n) RETURN n"
    assert extract_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"
