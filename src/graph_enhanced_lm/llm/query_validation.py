"""Static safety validation for LLM-generated Cypher (read-only QA mode).

Defence in depth (this module = layer 1 of several):
  1. static checks here (comments stripped, single statement, keyword deny-list,
     procedure allow-list, parameter whitelist, LIMIT enforcement);
  2. Neo4j EXPLAIN must report query type 'r' (see TextToCypher);
  3. execution inside a READ transaction with timeout and row cap;
  4. (Enterprise only) a least-privilege reader account.
A regular-expression/token check alone is NOT a complete security boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FORBIDDEN_KEYWORDS = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "LOAD", "FOREACH",
    "ALTER", "GRANT", "REVOKE", "DENY", "RENAME", "START", "STOP", "TERMINATE",
    "TRANSACTIONS", "PERIODIC", "COMMIT", "USE", "ENABLE", "DISABLE", "SHOW",
}
DANGEROUS_NAMESPACES = ("apoc.", "dbms.", "gds.", "db.", "genai.", "tx.")
ALLOWED_START = {"MATCH", "OPTIONAL", "WITH", "UNWIND", "RETURN", "CALL"}
PARAM_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+|\$[A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)
SAFE_VALUE_TYPES = (str, int, float, bool, type(None))


@dataclass
class ValidationResult:
    ok: bool
    cypher: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def mask_literals_and_comments(query: str) -> tuple[str, str]:
    """Return (query_without_comments, masked_query).

    ``masked_query`` additionally replaces string/backtick literal contents with
    spaces so keyword checks only see real Cypher tokens.
    """
    out, masked = [], []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":                       # line comment
            while i < n and query[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":                       # block comment
            end = query.find("*/", i + 2)
            i = n if end == -1 else end + 2
            out.append(" ")
            masked.append(" ")
            continue
        if ch in ("'", '"', "`"):                          # literal
            quote, j = ch, i + 1
            while j < n:
                if query[j] == "\\" and quote != "`":
                    j += 2
                    continue
                if query[j] == quote:
                    break
                j += 1
            literal = query[i:j + 1]
            out.append(literal)
            masked.append(quote + " " * max(len(literal) - 2, 0) + (quote if j < n else ""))
            i = j + 1
            continue
        out.append(ch)
        masked.append(ch)
        i += 1
    return "".join(out), "".join(masked)


def extract_cypher(llm_text: str) -> str:
    """Pull the query out of a ```cypher fenced block (or return trimmed text)."""
    m = re.search(r"```(?:cypher|sql)?\s*(.*?)```", llm_text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else llm_text).strip()


class CypherValidator:
    def __init__(self, max_rows: int = 200, allowed_procedures: list[str] | None = None):
        self.max_rows = max_rows
        self.allowed_procedures = {p.lower() for p in (allowed_procedures or [])}

    def validate(self, query: str, params: dict[str, Any] | None = None) -> ValidationResult:
        params = params or {}
        errors: list[str] = []
        warnings: list[str] = []
        if not query or not query.strip():
            return ValidationResult(False, query, ["Empty query"])
        clean, masked = mask_literals_and_comments(query)

        # --- single statement only (a trailing semicolon is tolerated)
        stripped_masked = masked.strip()
        if stripped_masked.endswith(";"):
            cut = len(clean.rstrip()) - 1
            clean = clean.rstrip()[:cut]
            stripped_masked = stripped_masked[:-1]
        if ";" in stripped_masked:
            errors.append("Multiple statements are not allowed")

        # Parameters ($start, $limit ...) are not keywords: hide them before tokenising.
        tokens = WORD_RE.findall(PARAM_RE.sub(" ", masked))
        upper = [t.upper() for t in tokens]
        if not upper:
            return ValidationResult(False, query, ["No Cypher tokens found"])
        if upper[0] not in ALLOWED_START:
            errors.append(f"Query must start with one of {sorted(ALLOWED_START)}, got '{tokens[0]}'")

        bad = sorted({t for t in upper if t in FORBIDDEN_KEYWORDS})
        if bad:
            errors.append(f"Forbidden operation(s) in read-only mode: {bad}")

        for t in tokens:
            if t.lower().startswith(DANGEROUS_NAMESPACES) and t.lower() not in self.allowed_procedures:
                errors.append(f"Procedure/function namespace not allowed: {t}")

        # --- CALL is only allowed for subqueries "CALL {" or allow-listed procedures
        for m in re.finditer(r"\bCALL\b\s*(\{|[A-Za-z_][\w.]*)", masked, flags=re.IGNORECASE):
            target = m.group(1)
            if target != "{" and target.lower() not in self.allowed_procedures:
                errors.append(f"Procedure call not allowed: CALL {target}")

        # --- parameters must be known and of simple types
        used = set(PARAM_RE.findall(masked))
        unknown = sorted(used - set(params) - {"limit"})
        if unknown:
            errors.append(f"Unknown parameter(s): {unknown}")
        for name in used & set(params):
            if not self._safe_value(params[name]):
                errors.append(f"Parameter ${name} has an unsupported type")

        # --- enforce LIMIT
        final = clean.strip()
        lm = LIMIT_RE.search(final)
        if lm is None:
            final = f"{final}\nLIMIT {self.max_rows}"
            warnings.append(f"LIMIT {self.max_rows} appended")
        elif lm.group(1).isdigit() and int(lm.group(1)) > self.max_rows:
            final = final[:lm.start()] + f"LIMIT {self.max_rows}"
            warnings.append(f"LIMIT reduced to {self.max_rows}")
        return ValidationResult(not errors, final, errors, warnings)

    @staticmethod
    def _safe_value(v: Any) -> bool:
        if isinstance(v, SAFE_VALUE_TYPES):
            return True
        if isinstance(v, (list, tuple)):
            return all(isinstance(x, SAFE_VALUE_TYPES) for x in v)
        return False
