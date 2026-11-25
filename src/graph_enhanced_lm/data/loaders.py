"""Format-aware readers for CSV, TSV, JSON, JSONL, Excel and Parquet."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from graph_enhanced_lm.utils.errors import DatasetError

logger = logging.getLogger(__name__)


def detect_format(path: str | Path) -> str:
    """Infer the file format from its extension (and first byte for .json)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".tsv", ".tab"}:
        return "tsv"
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    if ext == ".json":
        with path.open("r", encoding="utf-8") as fh:
            first = fh.read(1)
            while first and first.isspace():
                first = fh.read(1)
        return "json_array" if first == "[" else "json_object"
    if ext in {".xlsx", ".xls"}:
        return "excel"
    if ext == ".parquet":
        return "parquet"
    raise DatasetError(f"Unsupported file format: {path}")


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, keeping identifiers as strings."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    fmt = detect_format(path)
    logger.info("Reading %s as %s", path, fmt)
    if fmt == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=True, **kwargs)
    if fmt == "tsv":
        kwargs.setdefault("sep", "\t")
        return pd.read_csv(path, dtype=str, **kwargs)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if fmt == "json_array":
        return pd.read_json(path, dtype=False)
    if fmt == "json_object":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return pd.DataFrame.from_dict(data, orient="index").reset_index(names="_key")
    if fmt == "excel":
        kwargs.pop("sep", None)
        return pd.read_excel(path, dtype=str, **kwargs)
    if fmt == "parquet":
        return pd.read_parquet(path)  # needs pyarrow
    raise DatasetError(f"Unhandled format {fmt}")


def iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream items from a (possibly multi-GB) JSON array without loading it fully.

    Uses ``ijson`` when installed; falls back to ``json.load`` for small files.
    """
    path = Path(path)
    try:
        import ijson  # type: ignore
    except ImportError:
        logger.warning("ijson not installed; loading %s fully into memory", path)
        with path.open("r", encoding="utf-8") as fh:
            yield from json.load(fh)
        return
    with path.open("rb") as fh:
        yield from ijson.items(fh, "item")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV (UTF-8), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def inspect_file(path: str | Path, n: int = 5) -> dict[str, Any]:
    """Return a quick structural summary of a data file (used by inspect_dataset.py)."""
    path = Path(path)
    fmt = detect_format(path)
    info: dict[str, Any] = {"path": str(path), "format": fmt,
                            "size_mb": round(path.stat().st_size / 1e6, 3)}
    if fmt == "json_array" and info["size_mb"] > 200:
        items = []
        for i, item in enumerate(iter_json_array(path)):
            items.append(item)
            if i + 1 >= n:
                break
        df = pd.DataFrame(items)
        info["note"] = "Large JSON array: only the first records were sampled."
    else:
        df = read_table(path)
    info.update({
        "rows": len(df),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_per_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "head": df.head(n).astype(str).to_dict(orient="records"),
    })
    return info
