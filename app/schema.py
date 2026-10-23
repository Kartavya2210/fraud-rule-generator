"""Strict JSON schema for a fraud detection rule.

A rule is a small boolean expression over catalog features:

    {"logic": "all", "conditions": [
        {"field": "amount", "op": ">", "value": 2000},
        {"logic": "any", "conditions": [
            {"field": "is_new_device", "op": "==", "value": true},
            {"field": "ip_country_mismatch", "op": "==", "value": true}]}]}

Nesting is capped at one group level, which keeps the JSON schema
non-recursive (easier for small local models to follow) and is enough to
express every pattern in the few-shot set.
"""
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError, field_validator, model_validator

from app.features import FEATURES, LABEL_COLUMNS

NUMERIC_OPS = {">", ">=", "<", "<=", "==", "!=", "between"}
CATEGORY_OPS = {"==", "!=", "in", "not_in"}
BOOLEAN_OPS = {"==", "!="}

Op = Literal[">", ">=", "<", "<=", "==", "!=", "in", "not_in", "between"]


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    op: Op
    value: Union[bool, float, str, list[Union[float, str]]]

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_bool_strings(cls, v: Any) -> Any:
        # Small models often emit "true"/"false" as strings for boolean features.
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    @model_validator(mode="after")
    def _check_against_catalog(self) -> "Condition":
        if self.field in LABEL_COLUMNS:
            raise ValueError(f"'{self.field}' is a ground-truth label and cannot be used as a rule input")
        feat = FEATURES.get(self.field)
        if feat is None:
            raise ValueError(f"unknown field '{self.field}'. Allowed fields: {', '.join(FEATURES)}")

        if feat.kind == "number":
            self._check_number(feat)
        elif feat.kind == "category":
            self._check_category(feat)
        else:
            if self.op not in BOOLEAN_OPS:
                raise ValueError(f"'{self.field}' is boolean; op must be == or !=")
            if not isinstance(self.value, bool):
                raise ValueError(f"'{self.field}' is boolean; value must be true or false")
        return self

    def _check_number(self, feat) -> None:
        if self.op not in NUMERIC_OPS:
            raise ValueError(f"'{self.field}' is numeric; op must be one of {sorted(NUMERIC_OPS)}")
        if self.op == "between":
            v = self.value
            if not (isinstance(v, list) and len(v) == 2 and all(_is_number(x) for x in v)):
                raise ValueError(f"'between' on '{self.field}' needs a [low, high] list of two numbers")
            if v[0] > v[1]:
                raise ValueError(f"'between' on '{self.field}': low ({v[0]}) is greater than high ({v[1]})")
            bounds = v
        else:
            if not _is_number(self.value):
                raise ValueError(f"'{self.field}' is numeric; value must be a number, got {self.value!r}")
            bounds = [self.value]
        for b in bounds:
            if feat.min is not None and b < feat.min or feat.max is not None and b > feat.max:
                raise ValueError(f"value {b} for '{self.field}' is outside its range [{feat.min}, {feat.max}]")

    def _check_category(self, feat) -> None:
        if self.op not in CATEGORY_OPS:
            raise ValueError(f"'{self.field}' is categorical; op must be one of {sorted(CATEGORY_OPS)}")
        values = self.value if self.op in ("in", "not_in") else [self.value]
        if self.op in ("in", "not_in") and (not isinstance(self.value, list) or not self.value):
            raise ValueError(f"'{self.op}' on '{self.field}' needs a non-empty list of values")
        bad = [v for v in values if v not in feat.values]
        if bad:
            raise ValueError(f"invalid value(s) {bad} for '{self.field}'. Allowed: {', '.join(feat.values)}")


class ConditionGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logic: Literal["all", "any"]
    conditions: list[Condition] = Field(min_length=1, max_length=8)


def _node_kind(v: Any) -> str:
    if isinstance(v, dict):
        return "group" if "conditions" in v else "condition"
    return "group" if isinstance(v, ConditionGroup) else "condition"


Node = Annotated[
    Union[Annotated[Condition, Tag("condition")], Annotated[ConditionGroup, Tag("group")]],
    Discriminator(_node_kind),
]


class FraudRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$", description="snake_case identifier")
    description: str = Field(min_length=5, max_length=500)
    severity: Literal["low", "medium", "high", "critical"]
    action: Literal["flag", "review", "block"]
    logic: Literal["all", "any"]
    conditions: list[Node] = Field(min_length=1, max_length=10)
    rationale: str = Field(min_length=5, max_length=800)

    def fields_used(self) -> set[str]:
        out = set()
        for node in self.conditions:
            for c in node.conditions if isinstance(node, ConditionGroup) else [node]:
                out.add(c.field)
        return out


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def format_errors(err: ValidationError) -> list[str]:
    """Flatten pydantic errors into short lines the model can act on."""
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"] if p not in ("condition", "group"))
        msg = e["msg"].removeprefix("Value error, ")
        lines.append(f"{loc}: {msg}" if loc else msg)
    return lines
