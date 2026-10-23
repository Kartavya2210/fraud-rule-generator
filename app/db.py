import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text,
    create_engine, select,
)
from sqlalchemy.engine import Engine

from app import config
from app.features import FEATURES

metadata = MetaData()

_FEATURE_TYPES = {"number": Float, "category": String(32), "boolean": Boolean}
_INT_FEATURES = {"account_age_days", "hour_of_day", "txn_count_1h", "txn_count_24h",
                 "distinct_merchants_1h", "distinct_countries_24h"}

transactions = Table(
    "transactions", metadata,
    Column("txn_id", String(16), primary_key=True),
    Column("ts", DateTime, nullable=False, index=True),
    Column("customer_id", String(16), nullable=False),
    Column("card_id", String(16), nullable=False, index=True),
    Column("merchant_id", String(16), nullable=False),
    *[Column(f.name, Integer if f.name in _INT_FEATURES else _FEATURE_TYPES[f.kind], nullable=False)
      for f in FEATURES.values()],
    Column("is_fraud", Boolean, nullable=False),
    Column("fraud_type", String(32)),
)

rules = Table(
    "rules", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), nullable=False),
    Column("status", String(16), nullable=False, default="draft"),  # draft | deployed | retired
    Column("rule_json", JSON, nullable=False),
    Column("source_prompt", Text),
    Column("provider", String(16)),
    Column("model", String(64)),
    Column("attempts", Integer),
    Column("created_at", DateTime, nullable=False),
    Column("deployed_at", DateTime),
)

backtests = Table(
    "backtests", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("rule_id", Integer, ForeignKey("rules.id"), nullable=False, index=True),
    Column("created_at", DateTime, nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("metrics", JSON, nullable=False),
)


def make_engine(url: str = config.DATABASE_URL) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True)


engine = make_engine()


def init_db(eng: Engine = engine) -> None:
    metadata.create_all(eng)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── historical transactions, cached as a DataFrame ──────────────────────────

_df_lock = threading.Lock()
_df_cache: pd.DataFrame | None = None


def load_transactions(eng: Engine = engine, refresh: bool = False) -> pd.DataFrame:
    global _df_cache
    with _df_lock:
        if _df_cache is None or refresh:
            with eng.connect() as conn:
                df = pd.read_sql(select(transactions), conn, parse_dates=["ts"])
            # SQLite hands booleans back as 0/1.
            for name, f in FEATURES.items():
                if f.kind == "boolean":
                    df[name] = df[name].astype(bool)
            df["is_fraud"] = df["is_fraud"].astype(bool)
            _df_cache = df
        return _df_cache
