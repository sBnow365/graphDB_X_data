"""Thin wrapper around the official Neo4j Python driver.

* Credentials come from environment variables (see ``config.neo4j_settings``).
* Reads run in READ-mode managed transactions with a timeout and a row cap.
* Writes run in WRITE-mode managed transactions, batched with UNWIND.
* All queries are parameterised: user input is never concatenated into Cypher.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from graph_enhanced_lm.utils.errors import GraphConnectionError

logger = logging.getLogger(__name__)


def to_serializable(value: Any) -> Any:
    """Convert Neo4j temporal / graph types into JSON-friendly Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if hasattr(value, "iso_format"):          # neo4j.time.DateTime / Date
        return value.iso_format()
    if hasattr(value, "items"):               # Node / Relationship
        return {k: to_serializable(v) for k, v in value.items()}
    return str(value)


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 read_timeout: float = 30, write_timeout: float = 300):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise GraphConnectionError("The 'neo4j' package is not installed: pip install -r requirements.txt") from exc
        self.database = database
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as exc:  # invalid URI etc.
            raise GraphConnectionError(f"Could not create Neo4j driver for {uri}: {exc}") from exc

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def from_config(cls, cfg: Any | None = None) -> "Neo4jClient":
        from graph_enhanced_lm.config import get_config

        cfg = cfg or get_config()
        s = cfg.neo4j_settings()
        n = cfg.neo4j
        return cls(s.uri, s.user, s.password, s.database,
                   n.get("read_timeout_seconds", 30), n.get("write_timeout_seconds", 300))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def verify(self) -> dict[str, Any]:
        """Check connectivity + authentication and return server information."""
        try:
            self._driver.verify_connectivity()
            info = self.run_read("CALL dbms.components() YIELD name, versions, edition "
                                 "RETURN name, versions[0] AS version, edition")
        except Exception as exc:
            raise GraphConnectionError(
                f"Cannot connect to Neo4j: {exc}. Is the container running "
                "(docker compose up -d) and is NEO4J_PASSWORD in .env correct?") from exc
        return info[0] if info else {}

    # ------------------------------------------------------------------ reads
    def run_read(self, query: str, params: dict[str, Any] | None = None,
                 timeout: float | None = None, max_rows: int | None = None) -> list[dict[str, Any]]:
        """Run a read query in a READ transaction; returns at most ``max_rows`` rows."""
        from neo4j import READ_ACCESS, unit_of_work  # type: ignore

        def _work(tx: Any) -> list[dict[str, Any]]:
            rows = []
            for record in tx.run(query, params or {}):
                rows.append(to_serializable(record.data()))
                if max_rows is not None and len(rows) >= max_rows:
                    break
            return rows

        work = unit_of_work(timeout=timeout or self.read_timeout)(_work)
        with self._driver.session(database=self.database, default_access_mode=READ_ACCESS) as s:
            return s.execute_read(work)

    def explain_query_type(self, query: str, params: dict[str, Any] | None = None) -> str:
        """Ask Neo4j to PLAN (not run) the query and report its type.

        Returns 'r' (read), 'rw', 'w' (write) or 's' (schema). This is a strong safety
        check because it is Neo4j's own parser, not a regular expression.
        """
        from neo4j import READ_ACCESS  # type: ignore

        with self._driver.session(database=self.database, default_access_mode=READ_ACCESS) as s:
            summary = s.run("EXPLAIN " + query, params or {}).consume()
            return str(summary.query_type)

    # ------------------------------------------------------------------ writes
    def run_write(self, query: str, params: dict[str, Any] | None = None,
                  timeout: float | None = None) -> dict[str, int]:
        """Run a write query; returns the update counters (nodes created, ...)."""
        from neo4j import WRITE_ACCESS, unit_of_work  # type: ignore

        def _work(tx: Any) -> dict[str, int]:
            summary = tx.run(query, params or {}).consume()
            c = summary.counters
            return {k: getattr(c, k) for k in ("nodes_created", "nodes_deleted",
                    "relationships_created", "relationships_deleted", "properties_set")}

        work = unit_of_work(timeout=timeout or self.write_timeout)(_work)
        with self._driver.session(database=self.database, default_access_mode=WRITE_ACCESS) as s:
            return s.execute_write(work)

    def run_write_query(self, query: str, params: dict[str, Any] | None = None,
                        timeout: float | None = None) -> list[dict[str, Any]]:
        """Run a WRITE transaction that also returns rows (used for GDS procedures)."""
        from neo4j import WRITE_ACCESS, unit_of_work  # type: ignore

        def _work(tx: Any) -> list[dict[str, Any]]:
            return [to_serializable(r.data()) for r in tx.run(query, params or {})]

        work = unit_of_work(timeout=timeout or self.write_timeout)(_work)
        with self._driver.session(database=self.database, default_access_mode=WRITE_ACCESS) as s:
            return s.execute_write(work)

    def run_schema(self, query: str) -> None:
        """Run a schema statement (CREATE CONSTRAINT / INDEX) as an auto-commit query."""
        with self._driver.session(database=self.database) as s:
            s.run(query).consume()

    def write_batches(self, query: str, rows: list[dict[str, Any]], batch_size: int = 1000,
                      extra_params: dict[str, Any] | None = None) -> dict[str, int]:
        """Send ``rows`` to an ``UNWIND $rows AS row ...`` query in chunks."""
        totals: dict[str, int] = {}
        for i in range(0, len(rows), batch_size):
            params = {"rows": rows[i:i + batch_size], **(extra_params or {})}
            for k, v in self.run_write(query, params).items():
                totals[k] = totals.get(k, 0) + v
        return totals


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
