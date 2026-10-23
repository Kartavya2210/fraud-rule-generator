"""Natural-language description -> validated FraudRule.

Pipeline: few-shot prompt -> LLM in JSON mode -> JSON extraction ->
strict schema validation. If validation fails, the exact errors are fed
back to the model as a follow-up turn and it gets another attempt.
"""
import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app import config
from app.features import catalog_text
from app.llm import get_llm
from app.prompts import build_prompt
from app.schema import FraudRule, format_errors

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass
class GenerationResult:
    rule: FraudRule | None
    provider: str
    model: str
    attempts: int
    latency_ms: int
    errors: list[list[str]] = field(default_factory=list)  # one entry per failed attempt
    raw_outputs: list[str] = field(default_factory=list)


class GenerationError(Exception):
    def __init__(self, result: GenerationResult):
        self.result = result
        super().__init__(f"no valid rule after {result.attempts} attempt(s): {result.errors[-1]}")


def extract_json(text: str) -> dict:
    text = _FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("response contains no JSON object")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e.msg} at position {e.pos}") from None
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value must be an object")
    return obj


def parse_rule(text: str) -> FraudRule:
    """Raises ValueError with a list of human-readable problems."""
    data = extract_json(text)
    try:
        return FraudRule.model_validate(data)
    except ValidationError as e:
        raise ValueError(format_errors(e)) from None


def _content_text(msg) -> str:
    content = msg.content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


def generate_rule(description: str, *, provider: str | None = None, model: str | None = None,
                  llm: BaseChatModel | None = None, few_shot: bool = True, constrained: bool = True,
                  max_attempts: int | None = None) -> GenerationResult:
    max_attempts = max_attempts or config.MAX_GENERATION_ATTEMPTS
    if llm is None:
        schema = FraudRule.model_json_schema() if constrained else None
        llm, provider, model = get_llm(provider, model, json_schema=schema)

    messages = build_prompt(few_shot).format_messages(catalog=catalog_text(), description=description)
    result = GenerationResult(rule=None, provider=provider or "custom", model=model or "custom",
                              attempts=0, latency_ms=0)
    t0 = time.perf_counter()

    while result.attempts < max_attempts:
        result.attempts += 1
        raw = _content_text(llm.invoke(messages))
        result.raw_outputs.append(raw)
        try:
            result.rule = parse_rule(raw)
            break
        except ValueError as e:
            problems = e.args[0] if isinstance(e.args[0], list) else [str(e.args[0])]
            result.errors.append(problems)
            messages += [
                AIMessage(content=raw),
                HumanMessage(content="That rule is invalid:\n- " + "\n- ".join(problems)
                             + "\nFix these problems and return the full corrected rule as a single JSON object."),
            ]

    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    if result.rule is None:
        raise GenerationError(result)
    return result
