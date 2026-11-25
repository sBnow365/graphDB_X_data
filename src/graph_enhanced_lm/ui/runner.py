"""Run project scripts as subprocesses and stream their output (used by dashboard buttons)."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterator

from graph_enhanced_lm.config import PROJECT_ROOT


def script_command(script: str, *args: str) -> list[str]:
    """``python scripts/<script> args...`` using the SAME interpreter (your .venv)."""
    return [sys.executable, str(PROJECT_ROOT / "scripts" / script), *[str(a) for a in args]]


def display_command(script: str, *args: str) -> str:
    """Human-readable command, as you would type it in the terminal."""
    quoted = [f'"{a}"' if " " in str(a) else str(a) for a in args]
    return " ".join(["python", f"scripts/{script}", *quoted])


def stream_script(script: str, *args: str, stdin_text: str | None = None) -> Iterator[str | int]:
    """Yield output lines as they arrive, then the integer return code as the last item."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(script_command(script, *args), cwd=PROJECT_ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE if stdin_text is not None else None,
                            text=True, encoding="utf-8", errors="replace")
    if stdin_text is not None and proc.stdin:
        proc.stdin.write(stdin_text)
        proc.stdin.close()
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    yield proc.wait()
