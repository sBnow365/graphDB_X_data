"""Edit top-level YAML scalars and .env entries IN PLACE, preserving comments and layout."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from graph_enhanced_lm.config import CONFIG_DIR, ENV_FILE, get_config, read_env_file


def _yaml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return f'"{text}"' if re.search(r"[:#\-\s]|^\d", text) else text


def set_yaml_scalar(config_name: str, key: str, value: Any, config_dir: Path = CONFIG_DIR) -> None:
    """Replace ``key: <value>`` at the top level of configs/<config_name>.yaml (comments kept)."""
    path = config_dir / f"{config_name}.yaml"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^({re.escape(key)}:[ \t]*)([^#\n]*?)([ \t]+#[^\n]*)?$", re.MULTILINE)
    if not pattern.search(text):
        raise KeyError(f"Top-level key '{key}' not found in {path.name}")
    text = pattern.sub(lambda m: f"{m.group(1)}{_yaml_literal(value)}{m.group(3) or ''}", text, count=1)
    path.write_text(text, encoding="utf-8")
    get_config.cache_clear()


def update_env_value(key: str, value: str, env_file: Path = ENV_FILE) -> None:
    """Set KEY=value in .env (adds the line if missing). Never used for passwords by the UI."""
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    out, done = [], False
    for line in lines:
        if re.match(rf"^\s*(export\s+)?{re.escape(key)}\s*=", line):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    env_file.write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")


def env_summary(env_file: Path = ENV_FILE) -> dict[str, str]:
    """Non-secret view of .env: passwords/keys are masked."""
    values = read_env_file(env_file) if env_file.exists() else {}
    out = {}
    for k, v in values.items():
        secret = any(s in k.upper() for s in ("PASSWORD", "KEY", "TOKEN"))
        out[k] = ("•" * 8 + f" ({len(v)} chars)") if secret and v else v
    return out
