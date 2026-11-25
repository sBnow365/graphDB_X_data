"""Central configuration loader.

All tunable values live in ``configs/*.yaml`` and ``.env``. Code calls
:func:`get_config` instead of hard-coding constants.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from graph_enhanced_lm.utils.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

_CONFIG_FILES = [
    "project", "dataset", "label_taxonomy", "classifier", "llm",
    "neo4j", "gds", "temporal", "evaluation", "synthetic",
]


ENV_FILE = PROJECT_ROOT / ".env"


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a .env file without third-party packages.

    Handles UTF-8, UTF-8 with BOM and UTF-16 (PowerShell/Notepad "Unicode") files,
    ``export KEY=value`` lines, quoted values and comment lines.
    """
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:200]:
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _load_dotenv() -> None:
    """Copy .env values into os.environ unless a NON-EMPTY value is already set."""
    if not ENV_FILE.exists():
        return
    for key, value in read_env_file(ENV_FILE).items():
        if not os.environ.get(key):
            os.environ[key] = value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dict (empty dict for empty files)."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


@dataclass
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str


@dataclass
class AppConfig:
    """Container for every YAML section plus resolved environment settings."""

    sections: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __getattr__(self, name: str) -> dict[str, Any]:
        sections = self.__dict__.get("sections", {})
        if name in sections:
            return sections[name]
        raise AttributeError(name)

    def path(self, key: str) -> Path:
        """Resolve a path from project.yaml ``paths`` relative to the project root."""
        rel = self.sections["project"]["paths"][key]
        return PROJECT_ROOT / rel

    def neo4j_settings(self) -> Neo4jSettings:
        """Neo4j credentials come ONLY from environment variables / .env."""
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password:
            found = sorted(read_env_file(ENV_FILE)) if ENV_FILE.exists() else []
            raise ConfigError(
                f"NEO4J_PASSWORD is not set. Looked for {ENV_FILE} "
                f"(exists: {ENV_FILE.exists()}; keys found: {found}). "
                "Run: python scripts/check_env.py")
        return Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=password,
            database=os.getenv("NEO4J_DATABASE", self.sections["neo4j"].get("database", "neo4j")),
        )

    def llm_settings(self) -> dict[str, Any]:
        """LLM settings with environment-variable overrides applied."""
        cfg = dict(self.sections["llm"])
        cfg["backend"] = os.getenv("LLM_BACKEND", cfg.get("backend", "openai_compatible"))
        cfg["base_url"] = os.getenv("LLM_BASE_URL", cfg.get("base_url", ""))
        cfg["model"] = os.getenv("LLM_MODEL", cfg.get("model", ""))
        cfg["api_key"] = os.getenv("LLM_API_KEY", "") or None
        return cfg


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache all configuration sections."""
    _load_dotenv()
    sections = {name: load_yaml(CONFIG_DIR / f"{name}.yaml") for name in _CONFIG_FILES}
    return AppConfig(sections=sections)


def load_dataset_mapping(name: str) -> dict[str, Any]:
    """Load ``configs/dataset_mappings/<name>.yaml``."""
    return load_yaml(CONFIG_DIR / "dataset_mappings" / f"{name}.yaml")
