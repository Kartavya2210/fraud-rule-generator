# Gen AI-Based Fraud Rule Generation

Turn a plain-English description of a fraud pattern into a structured, executable
detection rule. Every rule is **back-tested against historical transactions in both pandas
and PostgreSQL** before it can be deployed.

**Stack:** Python · LangChain · Ollama · FastAPI · pandas · PostgreSQL (SQLAlchemy) · Gemini / Groq

```
 "Big online order from a never-seen device,         ┌──────────────────────────┐
  far above the customer's normal spend"  ─────────▶ │ Few-shot prompt          │  LangChain
                                                     │ + feature catalog        │
                                                     └────────────┬─────────────┘
                                                                  ▼
                                   Ollama (local, schema-constrained decoding)
                                   Gemini / Groq (cloud, JSON mode)
                                                                  ▼
                                                     ┌──────────────────────────┐
                                ┌── errors fed ──────│ Strict JSON schema       │  Pydantic
                                │   back, retry      │ (catalog fields only,    │
                                └───────────────────▶│  typed ops, value vocab) │
                                                     └────────────┬─────────────┘
                                                                  ▼
                                           ┌──────────────────────┴──────────────────────┐
                                           ▼                                             ▼
                                  pandas boolean mask                         SQL WHERE clause
                                  (in-memory back-test)                       (runs in PostgreSQL)
                                           └───────────── must agree ───────────────────┘
                                                                  ▼
                                   Deployment gates: precision, flag rate, min hits
                                                                  ▼
                                             deployed rule set ─▶ POST /score
```

## How it works

**1. Feature catalog** (`app/features.py`). This is the single source of truth for what a rule
may reference: 17 transaction features (amount, merchant category, channel, geography, device,
account age, and velocity features such as `txn_count_1h`, `amount_sum_24h` and
`distinct_countries_24h`). The prompt, the validator, both compilers and the data generator all
read from it.

**2. Few-shot prompt with a strict JSON output schema** (`app/prompts.py`, `app/schema.py`).
LangChain's `FewShotChatMessagePromptTemplate` shows the model three worked examples. The
output must match a Pydantic schema that:
- allows only catalog fields, and rejects ground-truth label columns (`is_fraud`) to stop leakage
- type-checks operators against the field (`>` on a number, `in` on a category, `==` on a boolean)
- checks categorical values against the vocabulary and numeric values against ranges (no `hour_of_day > 25`)
- forbids extra keys and caps nesting at one group level, so the schema stays non-recursive and small models can follow it

On Ollama the full JSON schema goes to the model as `format=`, so decoding is
grammar-constrained. Cloud models run in JSON mode.

**3. Self-correcting generation** (`app/generator.py`). If the output fails validation, the exact
errors (for example `conditions.2.field: unknown field 'risk_score'`) go back to the model as a
follow-up turn, up to `MAX_GENERATION_ATTEMPTS` times.

**4. Two compilers, one rule** (`app/compiler.py`). A rule compiles to a pandas boolean mask and to
a SQLAlchemy WHERE clause. All values are bound parameters, so a generated rule cannot inject SQL.

**5. Back-test before deploy** (`app/backtest.py`). Each rule runs over the labelled history in
pandas **and** in the database. The two engines must flag the identical set of transaction IDs.
The report includes precision, recall, F1, flag rate, alerts per day, fraud dollars caught,
recall per fraud type, and incremental catch versus already-deployed rules. A rule can only be
deployed if its latest back-test clears every gate:

| gate | default |
|---|---|
| `engine_agreement` | pandas and SQL flag identical transactions |
| `min_flagged` | ≥ 5 hits |
| `min_precision` | ≥ 30% |
| `max_flag_rate` | ≤ 2% of all transactions |

**6. Historical data** (`scripts/generate_data.py`). This script builds ~56k labelled card
transactions over 90 days with five injected fraud patterns (card testing, account takeover,
bust-out, cloned card, late-night high-risk). It also includes realistic legitimate noise that
makes naive rules fail: travellers abroad, night-time gamblers, new customers buying
electronics, VPN users and new phones. Velocity features are materialised per card over
trailing windows, the way a feature store would.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # pick LLM provider, optionally set DATABASE_URL
ollama pull qwen2.5:7b          # for the local backend
```

**PostgreSQL.** Create a database and set `DATABASE_URL` in `.env`:

```bash
psql -U postgres -c "CREATE DATABASE fraud_rules;"
```
```
DATABASE_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/fraud_rules
```
If `DATABASE_URL` is unset, the app uses a local SQLite file so it runs with zero setup.

**Load historical transactions, then start the API:**

```bash
python scripts/generate_data.py --reset-rules
uvicorn app.main:app --port 8010
```

Open http://localhost:8010 for the UI, or http://localhost:8010/docs for Swagger.

## API

| method | path | purpose |
|---|---|---|
| `POST` | `/rules/generate` | `{description, provider?, model?, backtest?}` → rule + compiled SQL + back-test |
| `POST` | `/rules/validate` | validate a hand-written or edited rule JSON |
| `GET` | `/rules`, `/rules/{id}` | rule registry with latest back-test |
| `POST` | `/rules/{id}/backtest` | re-run the back-test |
| `POST` | `/rules/{id}/deploy` | deploy; `409` unless the latest back-test passed |
| `POST` | `/rules/{id}/retire` | take a rule out of the deployed set |
| `POST` | `/score` | run one transaction through all deployed rules → `approve` / `flag` / `review` / `block` |
| `GET` | `/features`, `/schema`, `/health` | catalog, rule JSON schema, status |

`provider` can be `ollama` (local), `gemini` or `groq` (cloud), chosen per request.

```bash
curl -X POST localhost:8010/rules/generate -H "Content-Type: application/json" \
  -d "{\"description\": \"High-value crypto or gambling between 1am and 4am\"}"
```

## Tests and evaluation

```bash
pytest -q                                              # 25 tests: schema, pandas/SQL parity, retry loop, API flow
python eval/run_eval.py --models qwen2.5:7b qwen2.5:1.5b
```

`eval/run_eval.py` runs 13 benchmark descriptions (10 targeting known patterns, one velocity rule,
one off-catalog request, one label-leak attempt) under four configurations, to show what each
technique contributes. Results go to `eval/results/summary.md`.

### Results (local Ollama, temperature 0, 11 in-scope + 2 adversarial cases)

| model | configuration | valid rule | passes back-test | mean precision | target-pattern recall | adversarial refused |
|---|---|---|---|---|---|---|
| qwen2.5:1.5b | zero-shot, JSON mode | 0% | 0% | – | – | 2/2 |
| qwen2.5:1.5b | + few-shot examples | 82% | 55% | 0.60 | 0.33 | 2/2 |
| qwen2.5:1.5b | + schema-constrained decoding | 91% | 64% | 0.64 | 0.35 | 2/2 |
| qwen2.5:1.5b | + validation-feedback retry | **100%** | **73%** | 0.63 | 0.41 | 2/2 |
| qwen2.5:7b | zero-shot, JSON mode | 82% | 45% | 0.45 | 0.35 | 1/2 |
| qwen2.5:7b | + few-shot examples | 100% | 82% | 0.72 | 0.60 | 2/2 |
| qwen2.5:7b | + schema-constrained decoding | 100% | 82% | 0.72 | 0.60 | 2/2 |
| qwen2.5:7b | + validation-feedback retry | 100% | 82% | 0.72 | 0.60 | 1/2 |

pandas and SQL flagged identical transactions for every valid rule in every run.

What the numbers show:
- **Few-shot examples matter most.** Zero-shot, the 1.5b model never produced a schema-valid
  rule (it wrote names like `"Card Testing"`). The 7b model produced valid but loose rules:
  precision rose from 0.45 to 0.72 once it saw worked examples.
- **Schema-constrained decoding helps the small model** (82% → 91% valid) and is neutral for
  7b, which already writes well-formed JSON. It cannot enforce *semantic* constraints, such as
  which values belong to which field. 1.5b still put `["crypto", "gambling"]` on `channel`
  instead of `merchant_category`. That is why the Pydantic validator exists.
- **Retry with error feedback** fixed that case on the third attempt, taking 1.5b to 100%
  valid. The trade-off shows on the 7b off-catalog request ("credit score below 500"): when
  told the field doesn't exist, the model swapped in other fields instead of declining. The
  label-leak request (`is_fraud = true`) was rejected in every configuration.
- Latency per rule: ~5 s for 1.5b and ~22 s for 7b on a laptop GPU with partial offload.

Raw per-case outputs are in `eval/results/*.jsonl`. The data is synthetic, so treat absolute
precision and recall as illustrative. The relative differences between configurations are the
point.

## Project layout

```
app/
  features.py    feature catalog (single source of truth)
  schema.py      strict Pydantic rule schema + error formatting
  prompts.py     few-shot LangChain prompt
  llm.py         Ollama / Gemini / Groq backends
  generator.py   generate -> validate -> feed errors back -> retry
  compiler.py    rule -> pandas mask, rule -> SQL WHERE
  backtest.py    dual-engine back-test + deployment gates
  db.py          SQLAlchemy tables (transactions, rules, backtests)
  main.py        FastAPI app
scripts/generate_data.py   synthetic labelled transaction history
eval/                      ablation benchmark
static/index.html          web UI
tests/                     pytest suite
```
