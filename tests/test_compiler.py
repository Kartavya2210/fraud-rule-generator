import pytest
from sqlalchemy import select

from app.compiler import to_pandas_mask, to_sql_clause
from app.db import engine, transactions
from app.prompts import EXAMPLES
from app.schema import FraudRule
from tests.conftest import make_rule

RULES = [e["output"] for e in EXAMPLES] + [
    make_rule(logic="any", conditions=[
        {"field": "amount", "op": ">=", "value": 2500},
        {"logic": "all", "conditions": [
            {"field": "merchant_category", "op": "not_in", "value": ["grocery", "fuel", "restaurant"]},
            {"field": "ip_country_mismatch", "op": "==", "value": True},
        ]},
    ]),
    make_rule(conditions=[
        {"field": "country", "op": "!=", "value": "US"},
        {"field": "is_international", "op": "!=", "value": False},
        {"field": "amount_to_avg_ratio", "op": "between", "value": [2, 50]},
    ]),
    make_rule(conditions=[{"field": "minutes_since_last_txn", "op": "<=", "value": 2}]),
]


@pytest.mark.parametrize("raw", RULES, ids=lambda r: r["name"])
def test_pandas_and_sql_flag_identical_transactions(txns, raw):
    rule = FraudRule.model_validate(raw)
    pandas_ids = set(txns.loc[to_pandas_mask(rule, txns), "txn_id"])
    with engine.connect() as conn:
        sql_ids = set(conn.execute(select(transactions.c.txn_id).where(to_sql_clause(rule, transactions))).scalars())
    assert pandas_ids == sql_ids
    assert pandas_ids, "rule should match something in the fixture data"


def test_sql_values_are_bound_parameters():
    rule = FraudRule.model_validate(make_rule(conditions=[
        {"field": "merchant_category", "op": "==", "value": "electronics"}]))
    compiled = to_sql_clause(rule, transactions).compile()
    assert "electronics" not in str(compiled)
    assert "electronics" in compiled.params.values()
