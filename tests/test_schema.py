import pytest
from pydantic import ValidationError

from app.schema import FraudRule, format_errors
from tests.conftest import make_rule


def errors_for(**overrides) -> list[str]:
    with pytest.raises(ValidationError) as exc:
        FraudRule.model_validate(make_rule(**overrides))
    return format_errors(exc.value)


def test_valid_rule_with_group():
    rule = FraudRule.model_validate(make_rule(conditions=[
        {"field": "channel", "op": "==", "value": "ecommerce"},
        {"logic": "any", "conditions": [
            {"field": "is_new_device", "op": "==", "value": True},
            {"field": "hour_of_day", "op": "between", "value": [0, 4]},
        ]},
    ]))
    assert rule.fields_used() == {"channel", "is_new_device", "hour_of_day"}


def test_unknown_field_rejected():
    errs = errors_for(conditions=[{"field": "customer_risk_score", "op": ">", "value": 0.9}])
    assert any("unknown field 'customer_risk_score'" in e for e in errs)


@pytest.mark.parametrize("label", ["is_fraud", "fraud_type"])
def test_label_leakage_rejected(label):
    errs = errors_for(conditions=[{"field": label, "op": "==", "value": True}])
    assert any("ground-truth label" in e for e in errs)


def test_category_value_outside_vocabulary():
    errs = errors_for(conditions=[{"field": "merchant_category", "op": "in", "value": ["electronics", "jewelry"]}])
    assert any("jewelry" in e for e in errs)


def test_numeric_op_on_category_rejected():
    errs = errors_for(conditions=[{"field": "channel", "op": ">", "value": "pos"}])
    assert any("categorical" in e for e in errs)


def test_between_bounds_and_range():
    assert any("greater than high" in e for e in errors_for(
        conditions=[{"field": "amount", "op": "between", "value": [500, 100]}]))
    assert any("outside its range" in e for e in errors_for(
        conditions=[{"field": "hour_of_day", "op": ">", "value": 25}]))


def test_boolean_strings_are_coerced():
    rule = FraudRule.model_validate(make_rule(conditions=[{"field": "is_new_device", "op": "==", "value": "true"}]))
    assert rule.conditions[0].value is True


def test_extra_keys_and_bad_name_rejected():
    assert errors_for(priority=1)
    assert errors_for(name="Bad Name!")


def test_nested_groups_not_allowed():
    errs = errors_for(conditions=[{"logic": "any", "conditions": [
        {"logic": "all", "conditions": [{"field": "amount", "op": ">", "value": 1}]}]}])
    assert errs
