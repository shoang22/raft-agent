# RAFT Order Agent

An AI agent that answers natural-language questions about customer orders. It fetches raw order data from a dummy API, parses it with an LLM, stores it in an in-memory SQLite database, and returns clean JSON — or a formatted table in the web UI.

## Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your key>
```

Start the dummy orders API in a separate terminal:

```bash
python dummy_customer_api.py
```

## Running

### Web UI (default)

```bash
python main.py
```

Opens a Gradio chat interface at `http://localhost:7860`. Type any natural-language question about orders. Progress updates stream in real time before the result table appears.

### CLI

```bash
python main.py --cli "Show me all orders from Ohio over $500"
```

Prints the result as formatted JSON to stdout.

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

## Running tests

```bash
pytest
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full diagram of the LangGraph pipeline, layer breakdown, and edge-case handling.
