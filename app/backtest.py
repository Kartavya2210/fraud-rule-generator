"""Back-test a rule against labelled historical transactions.

The rule is executed twice, once as a pandas mask over the cached
DataFrame and once as a SQL query inside the database. Both must flag the
identical set of transactions (proving the two compiled forms agree)
before any metric is trusted. The rule then has to clear the deployment
gates in app.config.
"""
import time

import pandas as pd
from sqlalchemy import case, func, select
from sqlalchemy.engine import Engine

from app import config
from app.compiler import to_pandas_mask, to_sql_clause, to_sql_text
from app.db import engine as default_engine, load_transactions, transactions
from app.schema import FraudRule


def _ratio(a: float, b: float) -> float:
    return round(float(a) / float(b), 4) if b else 0.0


def backtest(rule: FraudRule, eng: Engine = default_engine, df: pd.DataFrame | None = None,
             deployed: list[FraudRule] = ()) -> dict:
    if df is None:
        df = load_transactions(eng)
    if df.empty:
        raise ValueError("no historical transactions loaded; run scripts/generate_data.py first")

    # pandas execution
    t0 = time.perf_counter()
    mask = to_pandas_mask(rule, df)
    pandas_ms = (time.perf_counter() - t0) * 1000
    pandas_ids = set(df.loc[mask, "txn_id"])

    # SQL execution
    clause = to_sql_clause(rule, transactions)
    t0 = time.perf_counter()
    with eng.connect() as conn:
        sql_ids = set(conn.execute(select(transactions.c.txn_id).where(clause)).scalars())
        sql_flagged, sql_tp = conn.execute(
            select(func.count(), func.coalesce(func.sum(case((transactions.c.is_fraud, 1), else_=0)), 0))
            .where(clause)
        ).one()
    sql_ms = (time.perf_counter() - t0) * 1000

    fraud = df["is_fraud"]
    n, n_fraud = len(df), int(fraud.sum())
    flagged = int(mask.sum())
    tp = int((mask & fraud).sum())
    fp, fn = flagged - tp, n_fraud - tp
    precision, recall = _ratio(tp, flagged), _ratio(tp, n_fraud)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    days = max((df["ts"].max() - df["ts"].min()).total_seconds() / 86400, 1)
    agreement = pandas_ids == sql_ids and flagged == sql_flagged and tp == sql_tp

    by_type = {}
    for ftype, grp in df[fraud].groupby("fraud_type"):
        caught = int(mask[grp.index].sum())
        by_type[ftype] = {"total": len(grp), "caught": caught, "recall": _ratio(caught, len(grp))}

    incremental = None
    if deployed:
        covered = pd.Series(False, index=df.index)
        for r in deployed:
            covered |= to_pandas_mask(r, df)
        incremental = {
            "deployed_rules": len(deployed),
            "new_fraud_caught": int((mask & fraud & ~covered).sum()),
            "overlap_with_deployed": int((mask & covered).sum()),
        }

    gates = [
        {"gate": "engine_agreement", "passed": agreement,
         "detail": f"pandas flagged {flagged}, SQL flagged {int(sql_flagged)}, "
                   f"{len(pandas_ids ^ sql_ids)} mismatched txn ids"},
        {"gate": "min_flagged", "passed": flagged >= config.BACKTEST_MIN_FLAGGED,
         "detail": f"{flagged} flagged (need >= {config.BACKTEST_MIN_FLAGGED})"},
        {"gate": "min_precision", "passed": precision >= config.BACKTEST_MIN_PRECISION,
         "detail": f"precision {precision:.2%} (need >= {config.BACKTEST_MIN_PRECISION:.0%})"},
        {"gate": "max_flag_rate", "passed": _ratio(flagged, n) <= config.BACKTEST_MAX_FLAG_RATE,
         "detail": f"flag rate {_ratio(flagged, n):.2%} (need <= {config.BACKTEST_MAX_FLAG_RATE:.0%})"},
    ]

    sample_cols = ["txn_id", "ts", "amount", "merchant_category", "channel", "country", "is_fraud", "fraud_type"]
    sample = df.loc[mask, sample_cols].head(8).copy()
    sample["ts"] = sample["ts"].dt.strftime("%Y-%m-%d %H:%M")
    sample = sample.astype(object).where(sample.notna(), None)  # NaN is not valid JSON

    return {
        "passed": all(g["passed"] for g in gates),
        "gates": gates,
        "metrics": {
            "transactions": n, "fraud_transactions": n_fraud,
            "flagged": flagged, "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "flag_rate": _ratio(flagged, n),
            "alerts_per_day": round(flagged / days, 2),
            "fraud_amount_caught": round(float(df.loc[mask & fraud, "amount"].sum()), 2),
            "fraud_amount_total": round(float(df.loc[fraud, "amount"].sum()), 2),
        },
        "recall_by_fraud_type": by_type,
        "incremental": incremental,
        "engines": {
            "pandas_ms": round(pandas_ms, 1), "sql_ms": round(sql_ms, 1),
            "sql_dialect": eng.dialect.name, "agreement": agreement,
        },
        "sql_where": to_sql_text(rule, transactions, eng.dialect),
        "sample_flagged": sample.to_dict(orient="records"),
    }
