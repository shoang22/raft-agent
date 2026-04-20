# RAFT Order Agent

An AI agent that answers natural-language questions about customer orders. Given a question like _"Show me all orders from Ohio over $500"_, it:

1. Fetches raw, deliberately messy text from a dummy orders API
2. Uses an LLM to parse the unstructured text into structured `Order` objects
3. Stores the parsed orders in an in-memory SQLite database
4. Uses the LLM to translate the original question into a SQL query and executes it
5. Returns clean JSON — or a formatted table in the web UI

A linear regression model runs in the background to impute missing order totals from historical data, improving over time as more orders are parsed.

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your key>
```

## Running

You need **two terminals**: one for the dummy orders API and one for the agent.

**Terminal 1 — start the dummy orders API:**

```bash
python dummy_customer_api.py
# Serves messy order text on http://localhost:5001
```

**Terminal 2 — run the agent:**

```bash
# Web UI (default) — opens http://localhost:7860
python main.py

# CLI — prints JSON to stdout, progress to stderr
python main.py --cli "Show me all orders from Ohio over $500"
```

The web UI streams progress updates in real time as each pipeline stage completes, then renders the final result as a table.

## Example queries

```
Show me all orders where the buyer was located in Ohio and total value was over 500
Which buyer spent the most?
Show me order 1001
List all orders under $200
```

## Output format

```json
{
  "orders": [
    { "orderId": "1003", "buyer": "Alice Smith", "state": "OH", "total": 742.10 }
  ]
}
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key for [OpenRouter](https://openrouter.ai/) |

## Persistence

Two SQLite databases are created at runtime:

- **In-memory** (`sqlite:///:memory:`) — ephemeral store for the current query; wiped between runs
- **`orders_training.db`** — persistent store that accumulates all parsed orders across runs; used to retrain the total-imputation predictor
- **`total_predictor.joblib`** — persisted linear regression model; reloaded on startup so imputation is available immediately

## Running tests

```bash
pytest           # all tests
pytest -v        # verbose
pytest tests/unit/        # unit tests only (fast, no API required)
pytest tests/integration/ # requires dummy API running on :5001
```

The test suite uses in-memory fakes (no mocking) for all external dependencies. See `tests/fakes.py`.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full diagram of the LangGraph pipeline, layer breakdown, and edge-case handling.
