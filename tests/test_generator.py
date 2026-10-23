import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.generator import GenerationError, extract_json, generate_rule
from tests.conftest import make_rule

VALID = json.dumps(make_rule())


def test_extract_json_strips_fences_and_chatter():
    assert extract_json("Sure!\n```json\n" + VALID + "\n```")["name"] == "test_rule"


def test_retries_with_validation_feedback_until_valid():
    bad_field = json.dumps(make_rule(conditions=[{"field": "risk_score", "op": ">", "value": 1}]))
    llm = FakeListChatModel(responses=["not json at all", bad_field, VALID])
    result = generate_rule("large purchases", llm=llm, max_attempts=3)
    assert result.attempts == 3
    assert result.rule.name == "test_rule"
    assert "no JSON object" in result.errors[0][0]
    assert any("unknown field 'risk_score'" in e for e in result.errors[1])


def test_feedback_is_sent_back_to_model():
    seen = []

    class Recorder(FakeListChatModel):
        def invoke(self, messages, *a, **kw):
            seen.append(messages)
            return super().invoke(messages, *a, **kw)

    generate_rule("x" * 20, llm=Recorder(responses=["{}", VALID]), max_attempts=2)
    last_turn = seen[1][-1].content
    assert last_turn.startswith("That rule is invalid")
    assert "name: Field required" in last_turn


def test_gives_up_after_max_attempts():
    llm = FakeListChatModel(responses=["{}"] * 5)
    with pytest.raises(GenerationError) as exc:
        generate_rule("anything at all", llm=llm, max_attempts=2)
    assert exc.value.result.attempts == 2
