import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import main
from app.generator import generate_rule
from app.prompts import EXAMPLES
from tests.conftest import make_rule

GOOD = EXAMPLES[2]["output"]  # new_account_bust_out: high precision on the fixture data
NOISY = make_rule(name="any_online_purchase", conditions=[{"field": "channel", "op": "==", "value": "ecommerce"}])


@pytest.fixture
def client(txns, monkeypatch):
    def use_fake(responses):
        def fake_generate(description, **kw):
            return generate_rule(description, llm=FakeListChatModel(responses=responses))
        monkeypatch.setattr(main, "generate_rule", fake_generate)
    with TestClient(main.app) as c:
        c.use_fake = use_fake
        yield c


def test_generate_backtest_deploy_and_score(client):
    client.use_fake([json.dumps(GOOD)])
    r = client.post("/rules/generate", json={"description": "bust-out on new accounts"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backtest"]["passed"], body["backtest"]["gates"]
    assert body["backtest"]["engines"]["agreement"]
    rule_id = body["rule_id"]

    assert client.post(f"/rules/{rule_id}/deploy").json()["status"] == "deployed"

    txn = {"amount": 1800, "merchant_category": "electronics", "channel": "pos", "country": "US",
           "home_country": "US", "is_international": False, "ip_country_mismatch": False,
           "is_new_device": False, "account_age_days": 5, "hour_of_day": 14, "amount_to_avg_ratio": 4.0,
           "txn_count_1h": 2, "txn_count_24h": 3, "amount_sum_24h": 4200, "distinct_merchants_1h": 2,
           "distinct_countries_24h": 1, "minutes_since_last_txn": 30}
    scored = client.post("/score", json=txn).json()
    assert scored["decision"] == "review"
    assert [h["rule_id"] for h in scored["hits"]] == [rule_id]


def test_failing_backtest_blocks_deploy(client):
    client.use_fake([json.dumps(NOISY)])
    body = client.post("/rules/generate", json={"description": "flag every online purchase"}).json()
    assert not body["backtest"]["passed"]
    r = client.post(f"/rules/{body['rule_id']}/deploy")
    assert r.status_code == 409
    assert "min_precision" in r.json()["detail"]["failed_gates"]


def test_unrecoverable_output_returns_422(client):
    client.use_fake(['{"name": "x"}'] * 5)
    r = client.post("/rules/generate", json={"description": "something vague here"})
    assert r.status_code == 422
    assert r.json()["detail"]["attempts"] == 3


def test_validate_endpoint(client):
    assert client.post("/rules/validate", json=make_rule()).json()["valid"]
    bad = client.post("/rules/validate", json=make_rule(conditions=[{"field": "is_fraud", "op": "==", "value": True}]))
    assert not bad.json()["valid"]
