# Text-to-Cypher

## Pipeline
```
question ─► prompt (live schema + allowed $parameters + 3 example queries)
         ─► Qwen3-8B (non-thinking, temperature 0) ─► extract ```cypher block
         ─► static validator ─► Neo4j EXPLAIN (must be 'r') ─► READ transaction (timeout, row cap)
         ─► rows            (on any failure: error fed back once for a corrected query, else controlled error)
```
Code: `llm/text_to_cypher.py`, `llm/query_validation.py`, prompts in `llm/prompts.py`.

## Why parameters?
The LLM never sees raw values pasted into the query. The intent parser computes `$start`, `$end`,
`$label`, `$community_id`… and the LLM must reference them. The validator rejects unknown parameters
and non-primitive values. This prevents injection and wrong date arithmetic by the model.

## Safety layers
1. **Static validation:** comments stripped (`//`, `/* */`), string literals masked, one statement only,
   deny-list (`CREATE MERGE SET DELETE DETACH REMOVE DROP LOAD FOREACH … IN TRANSACTIONS`), no
   procedure `CALL` unless allow-listed (default: none), no `apoc./dbms./gds./db.` namespaces,
   must start with MATCH/OPTIONAL/WITH/UNWIND/RETURN/CALL{…}, LIMIT appended/capped.
2. **EXPLAIN:** Neo4j's own parser classifies the query; anything other than read-only is rejected.
3. **READ transaction** (`session.execute_read`) with timeout (`llm.yaml`) and row cap.
4. **Least privilege:** Neo4j Community has no fine-grained roles. On Enterprise/Aura, create a reader-only
   user for the QA interface **[verify RBAC availability in your edition]**.
A token check alone is **not** a security boundary; that is why layers 2–3 exist.
`benchmarks/cypher_safety_suite.yaml` contains 20 attack/benign cases; tests require 100% correct handling.

## Manual mode
`python scripts/run_cypher.py --query "..."` runs your own Cypher; write queries need `--write` and typing `yes`.

## Inspecting what happened
`python scripts/ask.py --t2c-only --json "…"` prints the generated Cypher, parameters, errors and
attempts. Every `ask` run is saved to `outputs/logs/ask_*.json` (no full post texts).

## Improving accuracy
Add verified examples to `FEW_SHOT` in `text_to_cypher.py`, tighten schema descriptions in
`graph/schema.py`, use a larger model, and measure with `evaluate_text_to_cypher.py`.
