"""Seeds, run identifiers and environment capture for reproducible experiments."""
from __future__ import annotations

import json
import platform
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ImportError:
        pass


def new_run_id(prefix: str = "run") -> str:
    """Create a sortable, unique run id such as ``run_20250101T101500Z_ab12cd``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def capture_environment() -> dict[str, Any]:
    """Record interpreter and key package versions for the experiment log."""
    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("pandas", "numpy", "neo4j", "sklearn", "transformers", "torch"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[pkg] = None
    return env


def save_json(data: Any, path: str | Path) -> Path:
    """Write JSON with UTF-8 and indentation, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    return path


def df_to_markdown(df: Any, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
