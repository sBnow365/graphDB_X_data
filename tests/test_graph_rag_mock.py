# """Graph RAG and Text-to-Cypher with a FAKE database client and a MOCK LLM."""
# from graph_enhanced_lm.llm.answer_generation import AnswerGenerator
# from graph_enhanced_lm.llm.backends.mock import MockLLMClient
# from graph_enhanced_lm.llm.query_validation import CypherValidator
# from graph_enhanced_lm.llm.text_to_cypher import TextToCypher
# from graph_enhanced_lm.retrieval.evidence_formatter import Evidence, format_rows
# from graph_enhanced_lm.retrieval.graph_rag import GraphRAG
# from graph_enhanced_lm.retrieval.graph_retriever import GraphRetriever


# class FakeClient:
#     def __init__(self, rows=None, qtype="r"):
#         self.rows = rows if rows is not None else [{"user_id": "u_0001", "labelled_posts": 3}]
#         self.qtype = qtype
#         self.queries = []

#     def run_read(self, query, params=None, timeout=None, max_rows=None):
#         self.queries.append((query, params))
#         return list(self.rows)

#     def explain_query_type(self, query, params=None):
#         return self.qtype


# def make_rag(cfg, client, llm):
#     retriever = GraphRetriever(client, "louvain_community", cfg.temporal,
#                                cfg.gds["community_analysis"]["priority_weights"])
#     t2c = TextToCypher(llm, CypherValidator(50), client) if llm else None
#     return GraphRAG(retriever, AnswerGenerator(llm), t2c, cfg.temporal["election_date"],
#                     cfg.temporal["timezone"], synthetic=True)


# def test_template_route_and_grounded_answer(cfg):
#     llm = MockLLMClient(default="User u_0001 posted 3 hate-labelled posts [E1].")
#     res = make_rag(cfg, FakeClient(), llm).ask(
#         "Which users posted hate-labelled content during the ten days before the election?")
#     assert res.method == "template:users_posting_label_in_range"
#     assert res.params["label"] == "HATE" and res.params["start"].startswith("2025-03-04T18:30")
#     assert res.answer.cited_ids == ["E1"] and not res.answer.unknown_citations
#     assert any("SYNTHETIC" in n for n in res.evidence.notes)


# def test_unknown_citation_flagged(cfg):
#     llm = MockLLMClient(default="Claim [E1] and invented claim [E9].")
#     res = make_rag(cfg, FakeClient(), llm).ask("Which users posted spam content?")
#     assert res.answer.unknown_citations == ["E9"]


# def test_empty_evidence_never_calls_llm(cfg):
#     llm = MockLLMClient(default="should not be used")
#     res = make_rag(cfg, FakeClient(rows=[]), llm).ask("Which users posted hate content?")
#     assert res.answer.insufficient and not res.answer.llm_called and not llm.calls


# def test_works_without_llm(cfg):
#     res = make_rag(cfg, FakeClient(), None).ask("Which users posted hate content?")
#     assert "Evidence only" in res.answer.text


# def test_text_to_cypher_fallback_executes_valid_query(cfg):
#     llm = MockLLMClient(default="```cypher\nMATCH (p:Post)-[:USES_HASHTAG]->(h:Hashtag) "
#                                 "WHERE 'SPAM' IN p.labels RETURN h.tag AS tag, count(*) AS n\n```")
#     client = FakeClient(rows=[{"tag": "deal", "n": 40}])
#     res = make_rag(cfg, client, llm).ask("What hashtags appear most in spam posts?")
#     assert res.method == "text_to_cypher"
#     assert res.cypher.endswith("LIMIT 50")
#     assert res.evidence.items[0].content.startswith("tag=deal")


# def test_text_to_cypher_rejects_destructive_then_controlled_error():
#     llm = MockLLMClient(default="```cypher\nMATCH (n) DETACH DELETE n\n```")
#     client = FakeClient()
#     r = TextToCypher(llm, CypherValidator(50), client, max_retries=1).run("delete everything")
#     assert not r.executed and r.attempts == 2 and "Forbidden" in r.errors[0]
#     assert client.queries == []                       # nothing reached the database


# def test_explain_write_type_rejected():
#     llm = MockLLMClient(default="```cypher\nMATCH (u:User) RETURN u\n```")
#     r = TextToCypher(llm, CypherValidator(50), FakeClient(qtype="rw"), max_retries=0).run("q")
#     assert not r.executed and "query type 'rw'" in r.errors[0]


# def test_thinking_blocks_removed():
#     llm = MockLLMClient(default="<think>reasoning...</think>```cypher\nMATCH (u:User) RETURN u.user_id\n```")
#     r = TextToCypher(llm, CypherValidator(10), None).run("list users")
#     assert r.valid and "think" not in r.cypher


# def test_evidence_empty_object():
#     assert Evidence().is_empty() and not format_rows([{"a": 1}]).is_empty()
"""Graph RAG and Text-to-Cypher with a FAKE database client and a MOCK LLM."""
from graph_enhanced_lm.llm.answer_generation import AnswerGenerator
from graph_enhanced_lm.llm.backends.mock import MockLLMClient
from graph_enhanced_lm.llm.query_validation import CypherValidator
from graph_enhanced_lm.llm.text_to_cypher import TextToCypher
from graph_enhanced_lm.retrieval.evidence_formatter import Evidence, format_rows
from graph_enhanced_lm.retrieval.graph_rag import GraphRAG
from graph_enhanced_lm.retrieval.graph_retriever import GraphRetriever


class FakeClient:
    def __init__(self, rows=None, qtype="r"):
        self.rows = rows if rows is not None else [{"user_id": "u_0001", "labelled_posts": 3}]
        self.qtype = qtype
        self.queries = []

    def run_read(self, query, params=None, timeout=None, max_rows=None):
        self.queries.append((query, params))
        return list(self.rows)

    def explain_query_type(self, query, params=None):
        return self.qtype


def make_rag(cfg, client, llm):
    retriever = GraphRetriever(client, "louvain_community", cfg.temporal,
                               cfg.gds["community_analysis"]["priority_weights"])
    t2c = TextToCypher(llm, CypherValidator(50), client) if llm else None
    return GraphRAG(retriever, AnswerGenerator(llm), t2c, cfg.temporal["election_date"],
                    cfg.temporal["timezone"], synthetic=True)


def test_template_route_and_grounded_answer(cfg):
    llm = MockLLMClient(default="User u_0001 posted 3 hate-labelled posts [E1].")
    res = make_rag(cfg, FakeClient(), llm).ask(
        "Which users posted hate-labelled content during the ten days before the election?")
    assert res.method == "template:users_posting_label_in_range"
    assert res.params["label"] == "HATE" and res.params["start"].startswith("2025-03-04T18:30")
    assert res.answer.cited_ids == ["E1"] and not res.answer.unknown_citations
    assert any("SYNTHETIC" in n for n in res.evidence.notes)


def test_unknown_citation_flagged(cfg):
    llm = MockLLMClient(default="Claim [E1] and invented claim [E9].")
    res = make_rag(cfg, FakeClient(), llm).ask("Which users posted spam content?")
    assert res.answer.unknown_citations == ["E9"]


def test_empty_evidence_never_calls_llm(cfg):
    llm = MockLLMClient(default="should not be used")
    res = make_rag(cfg, FakeClient(rows=[]), llm).ask("Which users posted hate content?")
    assert res.answer.insufficient and not res.answer.llm_called and not llm.calls


def test_works_without_llm(cfg):
    res = make_rag(cfg, FakeClient(), None).ask("Which users posted hate content?")
    assert "Evidence only" in res.answer.text


def test_text_to_cypher_fallback_executes_valid_query(cfg):
    llm = MockLLMClient(default="```cypher\nMATCH (p:Post)-[:USES_HASHTAG]->(h:Hashtag) "
                                "WHERE 'SPAM' IN p.labels RETURN h.tag AS tag, count(*) AS n\n```")
    client = FakeClient(rows=[{"tag": "deal", "n": 40}])
    res = make_rag(cfg, client, llm).ask("What hashtags appear most in spam posts?")
    assert res.method == "text_to_cypher"
    assert res.cypher.endswith("LIMIT 50")
    assert res.evidence.items[0].content.startswith("tag=deal")


def test_text_to_cypher_rejects_destructive_then_controlled_error():
    llm = MockLLMClient(default="```cypher\nMATCH (n) DETACH DELETE n\n```")
    client = FakeClient()
    r = TextToCypher(llm, CypherValidator(50), client, max_retries=1).run("delete everything")
    assert not r.executed and r.attempts == 2 and "Forbidden" in r.errors[0]
    assert client.queries == []                       # nothing reached the database


def test_explain_write_type_rejected():
    llm = MockLLMClient(default="```cypher\nMATCH (u:User) RETURN u\n```")
    r = TextToCypher(llm, CypherValidator(50), FakeClient(qtype="rw"), max_retries=0).run("q")
    assert not r.executed and "query type 'rw'" in r.errors[0]


def test_thinking_blocks_removed():
    llm = MockLLMClient(default="<think>reasoning...</think>```cypher\nMATCH (u:User) RETURN u.user_id\n```")
    r = TextToCypher(llm, CypherValidator(10), None).run("list users")
    assert r.valid and "think" not in r.cypher


def test_evidence_empty_object():
    assert Evidence().is_empty() and not format_rows([{"a": 1}]).is_empty()


def test_failed_text_to_cypher_reports_reason(cfg):
    llm = MockLLMClient(default="```cypher\nMATCH (n) DETACH DELETE n\n```")
    res = make_rag(cfg, FakeClient(), llm).ask("What are the 5 most used hashtags in spam posts?")
    assert res.method == "text_to_cypher_failed"
    assert res.answer.text.startswith("QUERY_FAILED") and "Forbidden" in res.answer.text
