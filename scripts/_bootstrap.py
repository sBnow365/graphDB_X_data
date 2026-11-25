"""Make ``src/`` importable when scripts are run directly (python scripts/x.py)
even if ``pip install -e .`` has not been run yet."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
