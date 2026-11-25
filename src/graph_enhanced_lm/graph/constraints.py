"""Create constraints and indexes from the .cypher files (single source of truth)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from graph_enhanced_lm.config import PROJECT_ROOT
from graph_enhanced_lm.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def parse_cypher_file(path: str | Path) -> list[str]:
    """Split a .cypher file into statements, dropping // comments and blank lines."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*", "", text)
    return [s.strip() for s in text.split(";") if s.strip()]


def create_constraints_and_indexes(client: Neo4jClient, constraints_file: str,
                                   indexes_file: str) -> int:
    """Run every statement; ``IF NOT EXISTS`` makes this safe to repeat."""
    n = 0
    for f in (constraints_file, indexes_file):
        for stmt in parse_cypher_file(f):
            logger.info("Schema: %s", stmt.splitlines()[0][:90])
            client.run_schema(stmt)
            n += 1
    return n
