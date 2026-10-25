import os
import sys
import tempfile
from pathlib import Path

# Point the app at a throwaway database before anything imports app.db.
# Set TEST_DATABASE_URL to run the suite against PostgreSQL; its tables are wiped.
os.environ["DATABASE_URL"] = (os.getenv("TEST_DATABASE_URL")
                              or f"sqlite:///{Path(tempfile.mkdtemp()).as_posix()}/test.db")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from app.db import load_transactions  # noqa: E402
from scripts.generate_data import build, load  # noqa: E402


@pytest.fixture(scope="session")
def txns():
    df = build(customers=800, days=45, seed=7)
    load(df, reset_rules=True)
    return load_transactions(refresh=True)


def make_rule(**overrides) -> dict:
    rule = {
        "name": "test_rule",
        "description": "A rule used in tests",
        "severity": "high",
        "action": "review",
        "logic": "all",
        "conditions": [{"field": "amount", "op": ">", "value": 1000}],
        "rationale": "Large amounts are risky",
    }
    rule.update(overrides)
    return rule
