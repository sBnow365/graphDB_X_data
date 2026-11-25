"""Check Neo4j connectivity, GDS availability, and create constraints + indexes.

    python scripts/initialize_neo4j.py
"""
import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.gds.availability import gds_info
from graph_enhanced_lm.graph.constraints import create_constraints_and_indexes
from graph_enhanced_lm.pipeline import connect
from graph_enhanced_lm.utils.errors import GDSUnavailableError
from graph_enhanced_lm.utils.logging_config import setup_logging


def main() -> None:
    cfg = get_config()
    setup_logging(cfg.project["log_level"], cfg.path("logs"), "init_neo4j")
    with connect(cfg) as client:
        info = client.verify()
        print(f"[OK] Connected to {info.get('name')} {info.get('version')} ({info.get('edition')})")
        try:
            g = gds_info(client)
            print(f"[OK] GDS {g['version']} ({g['procedure_count']} procedures)")
            if g["missing"]:
                print(f"[WARN] Missing GDS procedures: {g['missing']}")
        except GDSUnavailableError as exc:
            print(f"[WARN] {exc}\n       Graph loading and Cypher still work; GDS steps will fail.")
        n = create_constraints_and_indexes(client, cfg.neo4j["constraints_file"], cfg.neo4j["indexes_file"])
        print(f"[OK] {n} constraint/index statements applied")


if __name__ == "__main__":
    main()
