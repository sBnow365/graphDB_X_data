"""INTEGRATION: requires a running Neo4j (docker compose up -d), NEO4J_PASSWORD and RUN_INTEGRATION=1.
WARNING: uses the configured database; run it on the development database only."""
import pytest

from graph_enhanced_lm.data.preprocessing import preprocess_canonical
from graph_enhanced_lm.graph.constraints import create_constraints_and_indexes
from graph_enhanced_lm.graph.ingestion import GraphIngestor
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client(cfg):
    c = Neo4jClient.from_config(cfg)
    c.verify()
    create_constraints_and_indexes(c, cfg.neo4j["constraints_file"], cfg.neo4j["indexes_file"])
    yield c
    c.close()


def test_ingestion_is_idempotent(client, synth_ds):
    ds = preprocess_canonical(synth_ds)
    GraphIngestor(client).ingest_dataset(ds)
    n1 = client.run_read("MATCH (p:Post {synthetic: true}) RETURN count(p) AS n")[0]["n"]
    GraphIngestor(client).ingest_dataset(ds)
    n2 = client.run_read("MATCH (p:Post {synthetic: true}) RETURN count(p) AS n")[0]["n"]
    assert n1 == n2 >= len(ds.posts)
    dup = client.run_read("MATCH (u:User) WITH u.user_id AS id, count(*) AS c WHERE c > 1 RETURN count(*) AS n")
    assert dup[0]["n"] == 0


def test_read_session_blocks_writes(client):
    with pytest.raises(Exception):
        client.run_read("CREATE (:ShouldNotExist)")


def test_explain_reports_query_types(client):
    assert client.explain_query_type("MATCH (n) RETURN count(n)") == "r"
    assert client.explain_query_type("CREATE (n:Tmp) RETURN n") in ("w", "rw")
