"""Check that the configured LLM endpoint is reachable and answers.

    python scripts/check_llm.py
"""
import time

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import get_config
from graph_enhanced_lm.llm.qwen_client import build_llm_client, strip_thinking


def main() -> None:
    cfg = get_config()
    s = cfg.llm_settings()
    print(f"backend={s['backend']}  base_url={s.get('base_url')}  model={s.get('model')}")
    client = build_llm_client(s)
    if hasattr(client, "health"):
        print("models served:", client.health())
    t0 = time.perf_counter()
    r = client.chat([{"role": "user", "content": "Reply with exactly: READY"}], max_tokens=200)
    print(f"reply: {strip_thinking(r.text)!r}  ({time.perf_counter() - t0:.1f}s)")
    t0 = time.perf_counter()
    r = client.chat([{"role": "user", "content": "Write one Cypher query in a ```cypher block that counts "
                      "nodes labelled User. Output only the code block."}])
    print(f"cypher test: {strip_thinking(r.text)[:200]!r}  ({time.perf_counter() - t0:.1f}s)")
    raw = (r.raw or {}).get("choices", [{}])[0]
    print(f"finish_reason={raw.get('finish_reason')}  "
          f"thinking_in_separate_field={bool(raw.get('message', {}).get('reasoning') or raw.get('message', {}).get('thinking'))}")


if __name__ == "__main__":
    main()
