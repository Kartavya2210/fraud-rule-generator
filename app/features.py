"""Feature catalog: the single source of truth for what a rule may reference.

The prompt, the rule validator, both compilers (pandas + SQL) and the
synthetic data generator all read from here, so a field the model invents
or a label column it tries to peek at is rejected before it can run.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "number" | "category" | "boolean"
    description: str
    values: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None


MERCHANT_CATEGORIES = (
    "grocery", "restaurant", "fuel", "online_retail", "electronics",
    "travel", "gift_cards", "gambling", "crypto", "cash_withdrawal",
)
CHANNELS = ("pos", "ecommerce", "atm")
COUNTRIES = ("US", "GB", "IN", "DE", "AE", "SG", "BR", "NG", "RU", "CN")

FEATURES: dict[str, Feature] = {f.name: f for f in [
    Feature("amount", "number", "Transaction amount in USD", min=0),
    Feature("merchant_category", "category", "Merchant category code", MERCHANT_CATEGORIES),
    Feature("channel", "category", "pos = card present, ecommerce = online, atm = cash machine", CHANNELS),
    Feature("country", "category", "Country where the transaction happened", COUNTRIES),
    Feature("home_country", "category", "Cardholder's home country", COUNTRIES),
    Feature("is_international", "boolean", "country differs from home_country"),
    Feature("ip_country_mismatch", "boolean", "Online only: IP geolocation differs from home_country"),
    Feature("is_new_device", "boolean", "Device never seen before for this customer"),
    Feature("account_age_days", "number", "Days since the account was opened", min=0),
    Feature("hour_of_day", "number", "Local hour of the transaction, 0-23", min=0, max=23),
    Feature("amount_to_avg_ratio", "number",
            "amount divided by the customer's trailing 30-day average amount (1.0 when no history)", min=0),
    Feature("txn_count_1h", "number", "Transactions on the same card in the trailing hour, including this one", min=1),
    Feature("txn_count_24h", "number", "Transactions on the same card in the trailing 24 hours, including this one", min=1),
    Feature("amount_sum_24h", "number", "Total USD spent on the card in the trailing 24 hours, including this one", min=0),
    Feature("distinct_merchants_1h", "number", "Distinct merchants used by the card in the trailing hour", min=1),
    Feature("distinct_countries_24h", "number", "Distinct transaction countries for the card in the trailing 24 hours", min=1),
    Feature("minutes_since_last_txn", "number", "Minutes since the card's previous transaction (capped at 43200)", min=0),
]}

# Ground-truth columns used only for scoring a back-test. Never rule inputs.
LABEL_COLUMNS = ("is_fraud", "fraud_type")


def catalog_text() -> str:
    """Render the catalog for the LLM prompt."""
    lines = []
    for f in FEATURES.values():
        line = f"- {f.name} ({f.kind}): {f.description}"
        if f.values:
            line += f". Allowed values: {', '.join(f.values)}"
        lines.append(line)
    return "\n".join(lines)
