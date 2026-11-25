"""Diagnose .env problems without printing secrets.

    python scripts/check_env.py
"""
import os

import _bootstrap  # noqa: F401
from graph_enhanced_lm.config import ENV_FILE, get_config, read_env_file

KEYS = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE", "LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL"]


def main() -> None:
    print(f".env path      : {ENV_FILE}")
    print(f"exists         : {ENV_FILE.exists()}")
    others = sorted(p.name for p in ENV_FILE.parent.glob(".env*"))
    print(f".env* files    : {others}")
    pre = {k: os.environ.get(k) for k in KEYS}
    if ENV_FILE.exists():
        raw = ENV_FILE.read_bytes()
        print(f"size / start   : {len(raw)} bytes, first bytes {raw[:4]!r}")
        values = read_env_file(ENV_FILE)
        print(f"keys in file   : {sorted(values)}")
        pw = values.get("NEO4J_PASSWORD", "")
        print(f"password in file: length {len(pw)}")
    for k, v in pre.items():
        if v is not None:
            note = "EMPTY - blocks .env value" if v == "" else f"set, length {len(v)}"
            print(f"already in environment: {k} ({note})")
    get_config()
    pw = os.environ.get("NEO4J_PASSWORD", "")
    print(f"\nPython will use NEO4J_PASSWORD of length {len(pw)}"
          + ("  -> OK" if pw else "  -> STILL MISSING"))
    print(f"NEO4J_URI = {os.environ.get('NEO4J_URI')}")
    print("Docker must have been started with the same password: "
          "docker exec gelm-neo4j sh -c 'echo ${#NEO4J_AUTH}' should print 6 + that length.")


if __name__ == "__main__":
    main()
