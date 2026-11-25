"""MANUAL Cypher mode (you write the query; no LLM involved).

Read queries run in a READ transaction. Write queries require --write AND typing
'yes' at the prompt (trusted administrative use only).

    python scripts/run_cypher.py --query "MATCH (u:User) RETURN u.user_id LIMIT 5"
    python scripts/run_cypher.py --file cypher/03_basic_queries.cypher --index 4
    python scripts/run_cypher.py --query "MATCH (p:Post) WHERE p.timestamp >= datetime($start) RETURN count(p)" --param start=2025-03-01T00:00:00Z
"""
import argparse
import json

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.graph.constraints import parse_cypher_file
from graph_enhanced_lm.pipeline import connect


def parse_param(text: str):
    key, _, value = text.partition("=")
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query")
    g.add_argument("--file")
    ap.add_argument("--index", type=int, default=1, help="1-based statement number in --file")
    ap.add_argument("--param", action="append", default=[], help="name=value (JSON or string)")
    ap.add_argument("--write", action="store_true", help="allow a write query (asks for confirmation)")
    ap.add_argument("--max-rows", type=int, default=50)
    args = ap.parse_args()
    query = args.query or parse_cypher_file(args.file)[args.index - 1]
    params = dict(parse_param(p) for p in args.param)
    print(f"Cypher:\n{query}\nParams: {params}\n")
    cfg = get_config()
    with connect(cfg) as client:
        qtype = client.explain_query_type(query, params)
        print(f"EXPLAIN query type: {qtype}")
        if qtype == "r":
            rows = client.run_read(query, params, max_rows=args.max_rows)
            for r in rows:
                print(json.dumps(r, default=str, ensure_ascii=False))
            print(f"\n{len(rows)} row(s) (max {args.max_rows})")
            return
        if not args.write:
            raise SystemExit("This query WRITES to the database. Re-run with --write to confirm intent.")
        if input("Type 'yes' to execute this write query: ").strip().lower() != "yes":
            raise SystemExit("Cancelled.")
        if qtype == "s":
            client.run_schema(query)
            print("Schema statement executed.")
        else:
            print(client.run_write(query, params))


if __name__ == "__main__":
    main()
