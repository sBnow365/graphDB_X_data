"""Logging setup. Logs go to the console and to outputs/logs/<name>.log."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None,
                  name: str = "gelm") -> logging.Logger:
    """Configure root logging once and return the project logger."""
    root = logging.getLogger()
    if not getattr(root, "_gelm_configured", False):
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        root.setLevel(level.upper())
        root._gelm_configured = True  # type: ignore[attr-defined]
        # The neo4j driver is chatty at INFO.
        logging.getLogger("neo4j").setLevel(logging.WARNING)
    return logging.getLogger(name)


def redact_text(text: str | None, max_chars: int = 40) -> str:
    """Return a short, privacy-conscious preview of a post for logs."""
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
