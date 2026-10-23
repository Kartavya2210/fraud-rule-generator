"""Compile a validated FraudRule into executable form.

Two independent back-ends are produced from the same rule AST:
  * a pandas boolean mask, used for fast in-memory back-testing and scoring
  * a SQLAlchemy WHERE clause, run inside PostgreSQL (or SQLite)

Both only ever touch catalog columns and bind values as parameters, so a
generated rule cannot inject SQL. The back-tester runs both and checks they
flag exactly the same transactions.
"""
import operator

import pandas as pd
from sqlalchemy import Table, and_, not_, or_
from sqlalchemy.sql.elements import ColumnElement

from app.schema import Condition, ConditionGroup, FraudRule

_CMP = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}


# ── pandas ──────────────────────────────────────────────────────────────────

def _pd_condition(c: Condition, df: pd.DataFrame) -> pd.Series:
    col = df[c.field]
    if c.op in _CMP:
        return _CMP[c.op](col, c.value)
    if c.op == "between":
        return col.between(c.value[0], c.value[1], inclusive="both")
    if c.op == "in":
        return col.isin(c.value)
    if c.op == "not_in":
        return ~col.isin(c.value)
    raise ValueError(f"unsupported op {c.op}")


def _pd_combine(masks: list[pd.Series], logic: str, index) -> pd.Series:
    out = pd.Series(logic == "all", index=index)
    for m in masks:
        out = (out & m) if logic == "all" else (out | m)
    return out


def to_pandas_mask(rule: FraudRule, df: pd.DataFrame) -> pd.Series:
    masks = []
    for node in rule.conditions:
        if isinstance(node, ConditionGroup):
            masks.append(_pd_combine([_pd_condition(c, df) for c in node.conditions], node.logic, df.index))
        else:
            masks.append(_pd_condition(node, df))
    return _pd_combine(masks, rule.logic, df.index).astype(bool)


# ── SQL ─────────────────────────────────────────────────────────────────────

def _sql_condition(c: Condition, table: Table) -> ColumnElement:
    col = table.c[c.field]
    if c.op in _CMP:
        return _CMP[c.op](col, c.value)
    if c.op == "between":
        return col.between(c.value[0], c.value[1])
    if c.op == "in":
        return col.in_(c.value)
    if c.op == "not_in":
        return not_(col.in_(c.value))
    raise ValueError(f"unsupported op {c.op}")


def _sql_combine(clauses: list[ColumnElement], logic: str) -> ColumnElement:
    return and_(*clauses) if logic == "all" else or_(*clauses)


def to_sql_clause(rule: FraudRule, table: Table) -> ColumnElement:
    clauses = []
    for node in rule.conditions:
        if isinstance(node, ConditionGroup):
            clauses.append(_sql_combine([_sql_condition(c, table) for c in node.conditions], node.logic))
        else:
            clauses.append(_sql_condition(node, table))
    return _sql_combine(clauses, rule.logic)


def to_sql_text(rule: FraudRule, table: Table, dialect) -> str:
    """Human-readable WHERE clause with values inlined. Display only."""
    clause = to_sql_clause(rule, table)
    return str(clause.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
