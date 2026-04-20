# Architecture

## Overview

```
User Natural-Language Query
           |
           v
+---------------------------+     +---------------------------+
| entrypoints/cli.py        |     | entrypoints/gradio_app.py |
| (--cli flag)              |     | (default; streaming UI)   |
+---------------------------+     +---------------------------+
              \                         /
               v                       v
          +-------------------------------+
          |         bootstrap.py          |
          |  wires LLM, tools, UoW,       |
          |  predictor, progress reporter |
          +-------------------------------+
                         |
                         v
+---------------------------------------------------------------+
|              service_layer/agent.py (LangGraph)               |
|                                                               |
|  AgentState (TypedDict)                                       |
|  +------------------+                                        |
|  | query            |                                        |
|  | tool_name        |  fetch ──► parse ──► store ──► query   |
|  | raw_orders       |                  \                     |
|  | parsed_orders    |                   ──► format_single    |
|  | query_results    |                                        |
|  | error            |   (single-order path skips store/query)|
|  +------------------+                                        |
+---------------------------------------------------------------+
           |                              |
           v                              v
+---------------------+    +------------------------------+
| adapters/           |    | adapters/ml_model.py         |
|   orders_client.py  |    | LinearRegressionTotalPred.   |
|   repository.py     |    | retrain_async() after store  |
|   unit_of_work.py   |    +------------------------------+
+---------------------+
           |
           v
  +--------------------+          +--------------------+
  | SQLite :memory:    |          | orders_training.db |
  | (ephemeral/run)    |          | (persistent, file) |
  +--------------------+          +--------------------+
```

## Layers

```
src/raft_agent/
├── domain/
│   └── models.py          # Order, PartialOrder (Pydantic), USState (Enum)
├── adapters/
│   ├── abstractions.py    # AbstractLLM, AbstractOrdersClient, AbstractProgressReporter, ToolCall
│   ├── ml_model.py        # AbstractTotalPredictor, LinearRegressionTotalPredictor
│   ├── orders_client.py   # HTTP client → OrdersAPIClient; raises APIError
│   ├── progress.py        # CLIProgressReporter, GradioProgressReporter, NullProgressReporter
│   ├── repository.py      # AbstractOrderRepository, SqlAlchemyOrderRepository,
│   │                      # AbstractTrainingRepository, SqlAlchemyTrainingRepository
│   └── unit_of_work.py    # AbstractUnitOfWork, SqlAlchemyUnitOfWork (two engines)
├── service_layer/
│   ├── agent.py           # LangGraph graph, AgentState, AgentError, run_agent
│   ├── parsers.py         # parse_raw_orders, generate_sql_query, ParseError
│   └── prompts/
│       └── parse_order_chunk.md  # LLM extraction instructions
├── entrypoints/
│   ├── cli.py             # thin CLI; delegates to bootstrap; JSON to stdout
│   └── gradio_app.py      # Gradio chat UI; streams progress in real time
└── bootstrap.py           # composition root: build_tools, build_llm, bootstrap()
```

## Graph Nodes

| Node            | Responsibility                                                                           |
|-----------------|------------------------------------------------------------------------------------------|
| `fetch`         | LLM selects tool (`fetch_orders` or `fetch_order_by_id`) and invokes it via tool call   |
| `parse`         | LLM extracts raw text → `list[Order]` in batches; imputes missing totals via predictor  |
| `store`         | Inserts parsed orders into ephemeral SQLite; upserts training table; retrains predictor |
| `query`         | LLM generates SQL SELECT; executes against ephemeral DB; retries up to 3× on bad SQL   |
| `format_single` | Formats a single-order result (used for `fetch_order_by_id`; skips DB entirely)         |

## Routing

```
fetch
  └─► parse
          ├─ fetch_order_by_id ──► format_single ──► END
          └─ fetch_orders      ──► store ──► query ──► END

Any node sets state["error"] → END (raises AgentError to caller)
```

## Dependency Injection

`bootstrap.py` is the composition root. It instantiates all infrastructure adapters and injects them into the agent via function arguments, so tests can swap in fakes without touching production code.

```
bootstrap()
  ├── build_llm()               → _LangChainOpenAIAdapter (wraps ChatOpenRouter)
  ├── build_tools(client)       → LangChain tool objects for fetch_orders / fetch_order_by_id
  ├── OrdersAPIClient()         → HTTP adapter for dummy API
  ├── SqlAlchemyUnitOfWork()    → manages two SQLite engines (ephemeral + training)
  ├── LinearRegressionTotalPredictor()  → loads saved model from disk if present
  ├── CLIProgressReporter / GradioProgressReporter  → selected by entrypoint
  └── returns _run(query: str)  → single async callable used by both entrypoints
```

## Data Flow (bulk query)

```
User: "Orders from Ohio over $500"
  │
  ▼  [fetch node]
  LLM picks fetch_orders tool → GET /api/orders → list of raw text strings
  │
  ▼  [parse node]
  For each raw string:
    chunk_by_words() → context-window-sized chunks
    LLM invoke_structured(OrderChunk) → validated Pydantic object
    hallucination check (extracted values appear in source text)
    retry up to 3× with error feedback
  Missing totals → LinearRegressionTotalPredictor.predict(state)
  │
  ▼  [store node]
  INSERT INTO orders (ephemeral)    → used for SQL query
  UPSERT INTO training_orders       → accumulates across runs
  predictor.retrain_async()         → background thread
  │
  ▼  [query node]
  LLM generates SQL SELECT with schema context + state abbreviation mapping
  Validate: starts with SELECT, strip markdown fences, normalize Unicode quotes
  Execute → list[dict] rows
  retry up to 3× with error feedback on invalid SQL
  │
  ▼
  {"orders": [...]}
```

## Key Files

```
main.py                                      shell entry point; --cli flag or Gradio default
src/raft_agent/entrypoints/cli.py            CLI: loads .env, calls bootstrap(), prints JSON
src/raft_agent/entrypoints/gradio_app.py     Gradio chat UI; streams node-by-node progress
src/raft_agent/bootstrap.py                  wires all dependencies; returns run() callable
src/raft_agent/service_layer/agent.py        LangGraph graph, AgentState, run_agent()
src/raft_agent/service_layer/parsers.py      parse_raw_orders, generate_sql_query
src/raft_agent/adapters/abstractions.py      AbstractLLM, AbstractOrdersClient, AbstractProgressReporter
src/raft_agent/adapters/ml_model.py          LinearRegressionTotalPredictor; imputes + predicts order totals
src/raft_agent/adapters/progress.py          CLIProgressReporter, GradioProgressReporter, NullProgressReporter
src/raft_agent/adapters/orders_client.py     HTTP adapter for dummy API; raises APIError
src/raft_agent/adapters/repository.py        SQLAlchemy Core repositories (ephemeral + training)
src/raft_agent/adapters/unit_of_work.py      transaction boundary; manages two async SQLite engines
src/raft_agent/domain/models.py              Order, PartialOrder, USState
dummy_customer_api.py                        Flask API on :5001 serving deliberately messy order text
tests/fakes.py                               FakeLLM, FakeOrdersClient, FakeUnitOfWork, FakeTotalPredictor
tests/unit/test_domain.py                    domain model validation tests
tests/unit/test_ml_model.py                  LinearRegressionTotalPredictor tests
tests/unit/test_service.py                   agent graph, parser, and SQL generation tests
tests/integration/test_orders_client.py      HTTP adapter integration tests (requires API on :5001)
tests/e2e/test_bootstrap.py                  full wiring smoke tests
```

## Edge Cases Handled

| Problem                      | Solution                                                                        |
|------------------------------|---------------------------------------------------------------------------------|
| Context window overflow      | `chunk_by_words` splits input at word boundaries; 20% reserved for JSON overhead |
| LLM hallucination            | Extracted field values verified to appear in original source text               |
| Invalid SQL from LLM         | `generate_sql_query` retries up to 3×; validates output starts with SELECT      |
| Unicode/smart quotes in SQL  | Normalized to ASCII before execution                                            |
| Missing order totals         | `LinearRegressionTotalPredictor` imputes from state → total regression          |
| Predictor not yet trained    | Orders with missing totals are dropped until ≥2 training examples exist         |
| API unavailability           | `APIError` propagated as `AgentError` with message to caller                    |
| Single vs. bulk fetch        | Routing in `_post_parse_dispatch`; single orders skip DB + SQL path entirely    |
| No matching orders           | Returns `{"orders": []}` — never raises on an empty result set                  |
| LLM parse failure after 3×   | `ParseError` raised, caught in agent, propagated as `AgentError`                |
