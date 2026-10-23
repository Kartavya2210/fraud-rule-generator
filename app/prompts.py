"""Few-shot prompt for turning a fraud-pattern description into a rule.

The examples deliberately cover only three of the patterns present in the
historical data (card testing, ATM cash-out abroad, new-account bust-out),
so evaluation on the other patterns measures generalisation rather than
copy-paste.
"""
import json

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

SYSTEM = """You are a fraud strategy analyst who writes transaction-monitoring rules.
Convert the analyst's description of a fraud pattern into exactly ONE detection rule, as JSON.

Available features. These are the ONLY fields a rule may use:
{catalog}

Rule format:
{{
  "name": snake_case identifier, 3-64 chars,
  "description": one sentence describing what the rule catches,
  "severity": "low" | "medium" | "high" | "critical",
  "action": "flag" | "review" | "block",
  "logic": "all" | "any"  (how the top-level conditions combine),
  "conditions": list of conditions or groups,
  "rationale": why these signals and thresholds catch the pattern
}}
condition = {{"field": <feature>, "op": <op>, "value": <value>}}
group     = {{"logic": "all" | "any", "conditions": [condition, ...]}}   (groups cannot be nested)

Operators:
- number features: >, >=, <, <=, ==, !=, between (value is [low, high], inclusive)
- category features: ==, != (value is one allowed value) or in, not_in (value is a list of allowed values)
- boolean features: ==, != (value is true or false)

Guidelines:
- Use only the fields and category values listed above. Never invent fields or values.
- Combine several signals with "all" rather than relying on one weak signal.
- Choose thresholds tight enough that the rule flags well under 2% of all transactions.
- Respond with the JSON object only, with no markdown and no commentary."""

EXAMPLES = [
    {
        "input": "Fraudsters testing stolen cards: a burst of tiny online payments on the same card "
                 "at several different merchants within an hour.",
        "output": {
            "name": "card_testing_burst",
            "description": "Many sub-$5 online payments on one card across several merchants within an hour.",
            "severity": "high",
            "action": "block",
            "logic": "all",
            "conditions": [
                {"field": "channel", "op": "==", "value": "ecommerce"},
                {"field": "amount", "op": "<", "value": 5},
                {"field": "txn_count_1h", "op": ">=", "value": 5},
                {"field": "distinct_merchants_1h", "op": ">=", "value": 3},
            ],
            "rationale": "Card testers validate stolen numbers with very small charges in rapid succession; "
                         "genuine customers rarely make five micro-payments at three merchants in an hour.",
        },
    },
    {
        "input": "Cash-out abroad: ATM withdrawals in a foreign country in the middle of the night.",
        "output": {
            "name": "foreign_atm_night_cashout",
            "description": "ATM cash withdrawal outside the home country between midnight and 5am.",
            "severity": "medium",
            "action": "review",
            "logic": "all",
            "conditions": [
                {"field": "channel", "op": "==", "value": "atm"},
                {"field": "is_international", "op": "==", "value": True},
                {"field": "hour_of_day", "op": "between", "value": [0, 5]},
            ],
            "rationale": "Skimmed cards are typically emptied at foreign ATMs at night when the cardholder "
                         "is unlikely to notice; travellers withdraw cash mostly in daytime.",
        },
    },
    {
        "input": "Bust-out on brand-new accounts: someone opens an account and immediately spends big on "
                 "electronics or gift cards.",
        "output": {
            "name": "new_account_bust_out",
            "description": "Account under 30 days old spending heavily on easily resold goods within a day.",
            "severity": "high",
            "action": "review",
            "logic": "all",
            "conditions": [
                {"field": "account_age_days", "op": "<", "value": 30},
                {"field": "merchant_category", "op": "in", "value": ["electronics", "gift_cards"]},
                {"field": "amount_sum_24h", "op": ">", "value": 2000},
            ],
            "rationale": "Bust-out fraudsters max out fresh accounts on goods that convert to cash quickly; "
                         "legitimate new customers ramp up spending gradually.",
        },
    },
]


def build_prompt(few_shot: bool = True) -> ChatPromptTemplate:
    messages = [("system", SYSTEM)]
    if few_shot:
        example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
        examples = [{"input": e["input"], "output": json.dumps(e["output"])} for e in EXAMPLES]
        messages.append(FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples))
    messages.append(("human", "{description}"))
    return ChatPromptTemplate.from_messages(messages)
