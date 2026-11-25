"""Shared fixtures. Integration tests are skipped unless Neo4j credentials are set."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_enhanced_lm.config import get_config  # noqa: E402
from graph_enhanced_lm.data.synthetic import generate_synthetic_dataset  # noqa: E402


def pytest_collection_modifyitems(config, items):
    try:
        get_config()  # loads .env
    except Exception:
        pass
    if os.getenv("NEO4J_PASSWORD") and os.getenv("RUN_INTEGRATION", "0") == "1":
        return
    skip = pytest.mark.skip(reason="integration test: set NEO4J_PASSWORD and RUN_INTEGRATION=1")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cfg():
    return get_config()


@pytest.fixture(scope="session")
def small_synth_cfg(cfg):
    c = dict(cfg.synthetic)
    c.update(n_users=60, mean_posts_per_user=4)
    return c


@pytest.fixture(scope="session")
def synth_ds(small_synth_cfg):
    return generate_synthetic_dataset(small_synth_cfg)
