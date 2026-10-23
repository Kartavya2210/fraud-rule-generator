import logging
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, create_model
from sqlalchemy import func, insert, select, update

from app import config
from app.backtest import backtest
from app.compiler import to_pandas_mask, to_sql_text
from app.db import backtests, engine, init_db, load_transactions, rules, transactions, utcnow
from app.features import FEATURES
from app.generator import GenerationError, generate_rule
from app.llm import PROVIDERS
from app.schema import FraudRule, format_errors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fraud_rules")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Gen AI Fraud Rule Generator", version="1.0.0", lifespan=lifespan)
STATIC = config.ROOT / "static"


# ── request / response models ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    description: str = Field(min_length=10, max_length=1000)
    provider: Literal["ollama", "gemini", "groq"] | None = None
    model: str | None = None
    backtest: bool = True


def _feature_type(f):
    if f.kind == "number":
        return float
    if f.kind == "boolean":
        return bool
    return Literal[f.values]


TransactionIn = create_model(
    "TransactionIn", **{name: (_feature_type(f), ...) for name, f in FEATURES.items()}
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _get_rule_row(rule_id: int):
    with engine.connect() as conn:
        row = conn.execute(select(rules).where(rules.c.id == rule_id)).mappings().first()
    if row is None:
        raise HTTPException(404, f"rule {rule_id} not found")
    return row


def _latest_backtest(conn, rule_id: int):
    return conn.execute(
        select(backtests).where(backtests.c.rule_id == rule_id).order_by(backtests.c.id.desc()).limit(1)
    ).mappings().first()


def _rule_out(row, conn) -> dict:
    bt = _latest_backtest(conn, row["id"])
    return {
        "id": row["id"], "name": row["name"], "status": row["status"],
        "rule": row["rule_json"], "source_prompt": row["source_prompt"],
        "provider": row["provider"], "model": row["model"], "attempts": row["attempts"],
        "created_at": row["created_at"], "deployed_at": row["deployed_at"],
        "sql_where": to_sql_text(FraudRule.model_validate(row["rule_json"]), transactions, engine.dialect),
        "latest_backtest": None if bt is None else {
            "id": bt["id"], "passed": bt["passed"], "created_at": bt["created_at"],
            "precision": bt["metrics"]["metrics"]["precision"], "recall": bt["metrics"]["metrics"]["recall"],
            "flagged": bt["metrics"]["metrics"]["flagged"],
        },
    }


def _deployed_rules(exclude_id: int | None = None) -> list[FraudRule]:
    with engine.connect() as conn:
        rows = conn.execute(select(rules.c.id, rules.c.rule_json).where(rules.c.status == "deployed")).all()
    return [FraudRule.model_validate(r.rule_json) for r in rows if r.id != exclude_id]


def _run_backtest(rule_id: int, rule: FraudRule) -> dict:
    try:
        result = backtest(rule, engine, deployed=_deployed_rules(exclude_id=rule_id))
    except ValueError as e:
        raise HTTPException(409, str(e))
    with engine.begin() as conn:
        conn.execute(insert(backtests).values(rule_id=rule_id, created_at=utcnow(),
                                              passed=result["passed"], metrics=result))
    return result


# ── endpoints ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    with engine.connect() as conn:
        n_txn = conn.execute(select(func.count()).select_from(transactions)).scalar_one()
        n_deployed = conn.execute(select(func.count()).select_from(rules)
                                  .where(rules.c.status == "deployed")).scalar_one()
    return {"status": "ok", "database": engine.dialect.name, "transactions": n_txn,
            "deployed_rules": n_deployed, "default_provider": config.LLM_PROVIDER, "providers": PROVIDERS}


@app.get("/features")
def features():
    return [{"name": f.name, "type": f.kind, "description": f.description,
             "values": list(f.values) or None} for f in FEATURES.values()]


@app.get("/schema")
def rule_schema():
    """The JSON schema every generated rule must satisfy."""
    return FraudRule.model_json_schema()


@app.post("/rules/generate")
async def generate(req: GenerateRequest):
    try:
        gen = await run_in_threadpool(generate_rule, req.description, provider=req.provider, model=req.model)
    except GenerationError as e:
        r = e.result
        raise HTTPException(422, {"message": str(e), "attempts": r.attempts, "errors": r.errors,
                                  "last_output": r.raw_outputs[-1][:2000] if r.raw_outputs else None})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # provider unreachable, bad API key, etc.
        logger.exception("LLM call failed")
        raise HTTPException(503, f"LLM backend error: {type(e).__name__}: {e}")

    rule = gen.rule
    with engine.begin() as conn:
        rule_id = conn.execute(insert(rules).values(
            name=rule.name, status="draft", rule_json=rule.model_dump(mode="json"),
            source_prompt=req.description, provider=gen.provider, model=gen.model,
            attempts=gen.attempts, created_at=utcnow(),
        )).inserted_primary_key[0]

    result = await run_in_threadpool(_run_backtest, rule_id, rule) if req.backtest else None
    logger.info("rule %s '%s' generated in %d attempt(s), %d ms", rule_id, rule.name, gen.attempts, gen.latency_ms)
    return {
        "rule_id": rule_id,
        "rule": rule.model_dump(mode="json"),
        "sql_where": to_sql_text(rule, transactions, engine.dialect),
        "generation": {"provider": gen.provider, "model": gen.model, "attempts": gen.attempts,
                       "latency_ms": gen.latency_ms, "rejected_attempts": gen.errors},
        "backtest": result,
    }


@app.post("/rules/validate")
def validate_rule(payload: dict):
    """Validate a hand-written or edited rule without saving it."""
    try:
        rule = FraudRule.model_validate(payload)
    except ValidationError as e:
        return {"valid": False, "errors": format_errors(e)}
    return {"valid": True, "sql_where": to_sql_text(rule, transactions, engine.dialect)}


@app.get("/rules")
def list_rules(status: Literal["draft", "deployed", "retired"] | None = None):
    q = select(rules).order_by(rules.c.id.desc())
    if status:
        q = q.where(rules.c.status == status)
    with engine.connect() as conn:
        return [_rule_out(r, conn) for r in conn.execute(q).mappings()]


@app.get("/rules/{rule_id}")
def get_rule(rule_id: int):
    row = _get_rule_row(rule_id)
    with engine.connect() as conn:
        out = _rule_out(row, conn)
        bt = _latest_backtest(conn, rule_id)
    out["latest_backtest_detail"] = bt["metrics"] if bt else None
    return out


@app.post("/rules/{rule_id}/backtest")
async def backtest_rule(rule_id: int):
    row = _get_rule_row(rule_id)
    return await run_in_threadpool(_run_backtest, rule_id, FraudRule.model_validate(row["rule_json"]))


@app.post("/rules/{rule_id}/deploy")
def deploy_rule(rule_id: int):
    row = _get_rule_row(rule_id)
    if row["status"] == "deployed":
        raise HTTPException(409, "rule is already deployed")
    with engine.begin() as conn:
        bt = _latest_backtest(conn, rule_id)
        if bt is None:
            raise HTTPException(409, "rule has not been back-tested; POST /rules/{id}/backtest first")
        if not bt["passed"]:
            failed = [g["gate"] for g in bt["metrics"]["gates"] if not g["passed"]]
            raise HTTPException(409, {"message": "latest back-test failed; rule cannot be deployed",
                                      "failed_gates": failed})
        conn.execute(update(rules).where(rules.c.id == rule_id)
                     .values(status="deployed", deployed_at=utcnow()))
    return {"id": rule_id, "status": "deployed"}


@app.post("/rules/{rule_id}/retire")
def retire_rule(rule_id: int):
    _get_rule_row(rule_id)
    with engine.begin() as conn:
        conn.execute(update(rules).where(rules.c.id == rule_id).values(status="retired"))
    return {"id": rule_id, "status": "retired"}


@app.post("/score")
def score(txn: TransactionIn):
    """Run a single transaction through every deployed rule."""
    with engine.connect() as conn:
        rows = conn.execute(select(rules).where(rules.c.status == "deployed")).mappings().all()
    df = pd.DataFrame([txn.model_dump()])
    hits = []
    for r in rows:
        rule = FraudRule.model_validate(r["rule_json"])
        if bool(to_pandas_mask(rule, df).iloc[0]):
            hits.append({"rule_id": r["id"], "name": rule.name, "severity": rule.severity, "action": rule.action})
    order = {"block": 2, "review": 1, "flag": 0}
    decision = max((h["action"] for h in hits), key=order.get, default="approve")
    return {"decision": decision, "hits": hits, "rules_evaluated": len(rows)}


@app.post("/admin/reload-transactions")
def reload_transactions():
    df = load_transactions(engine, refresh=True)
    return {"transactions": len(df)}
