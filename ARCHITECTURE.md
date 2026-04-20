# Architecture

## Overview

```
User NL Query
     |
     v
+---------------------------+     +---------------------------+
| entrypoints/cli.py        |     | entrypoints/gradio_app.py |
| (--cli flag)              |     | (default; streaming UI)   |
+---------------------------+     +---------------------------+
              \                         /
               v                       v
          +-------------------------------+
          |         bootstrap.py          |  <- composition root; wires LLM, tools, UoW, predictor
          +-------------------------------+
                         |
                         v
+---------------------------------------------------------------+
|              service_layer/agent.py (LangGraph)               |
|                                                               |
|  AgentState                                                   |
|  +-------------+                                             |
|  | query       |                                             |
|  | tool_name   |  fetch --> parse --> store --> query        |
|  | raw_orders  |                  \                          |
|  | parsed_     |                   --> format_single         |
|  | orders      |                                             |
|  | query_      |   (single-order path skips store/query)     |
|  | results     |                                             |
|  | error       |                                             |
|  +-------------+                                             |
+---------------------------------------------------------------+
```

## Layers

```
src/raft_agent/
├── domain/
│   └── models.py          # Order, FilterCriteria, OrdersOutput (Pydantic)
├── adapters/
│   ├── abstractions.py    # AbstractLLM, AbstractOrdersClient, AbstractProgressReporter, ToolCall
│   ├── ml_model.py        # AbstractTotalPredictor, LinearRegressionTotalPredictor
│   ├── orders_client.py   # HTTP client → OrdersAPIClient; raises APIError
│   ├── progress.py        # CLIProgressReporter, GradioProgressReporter, NullProgressReporter
│   ├── repository.py      # AbstractOrderRepository, SqlAlchemyOrderRepository
│   └── unit_of_work.py    # AbstractUnitOfWork, SqlAlchemyUnitOfWork
├── service_layer/
│   ├── agent.py           # LangGraph graph, AgentState, AgentError, run_agent
│   └── parsers.py         # parse_raw_orders, generate_sql_query, ParseError
├── entrypoints/
│   ├── cli.py             # thin CLI; delegates to bootstrap
│   └── gradio_app.py      # Gradio chat UI; streams progress updates in real time
└── bootstrap.py           # composition root; build_tools, build_llm, bootstrap
```

## Graph Nodes

| Node            | Responsibility                                                          |
|-----------------|-------------------------------------------------------------------------|
| `fetch`         | LLM selects tool (`fetch_orders` or `fetch_order_by_id`) + invokes it  |
| `parse`         | LLM parses raw text → `list[Order]` (batched); imputes missing totals  |
| `store`         | Inserts parsed orders into ephemeral SQLite; upserts training table; retrains predictor |
| `query`         | LLM generates SQL SELECT; executes against stored orders; returns rows  |
| `format_single` | Formats a single-order result (skips DB; used for `fetch_order_by_id`) |

## Routing

```
fetch
  └─► parse
          ├─ fetch_order_by_id ──► format_single ──► END
          └─ fetch_orders      ──► store ──► query ──► END

Any node sets state["error"] → END (raises AgentError to caller)
```

## Key Files

```
main.py                                    - shell entry point; --cli flag for CLI, default for Gradio
src/raft_agent/entrypoints/cli.py          - CLI: loads .env, calls bootstrap(), prints JSON
src/raft_agent/entrypoints/gradio_app.py   - Gradio chat UI; streams node-by-node progress
src/raft_agent/bootstrap.py                - wires LLM + tools + UoW + predictor; returns run() callable
src/raft_agent/service_layer/agent.py      - LangGraph graph definition and run_agent()
src/raft_agent/service_layer/parsers.py    - parse_raw_orders, generate_sql_query
src/raft_agent/adapters/abstractions.py    - AbstractLLM, AbstractOrdersClient, AbstractProgressReporter
src/raft_agent/adapters/ml_model.py        - LinearRegressionTotalPredictor; imputes + predicts order totals
src/raft_agent/adapters/progress.py        - progress reporters (CLI, Gradio, Null)
src/raft_agent/adapters/orders_client.py   - HTTP adapter; raises APIError
src/raft_agent/adapters/repository.py      - SQLAlchemy Core repository (ephemeral + training tables)
src/raft_agent/adapters/unit_of_work.py    - transaction boundary
src/raft_agent/domain/models.py            - Order, FilterCriteria, OrdersOutput
tests/fakes.py                             - FakeLLM, FakeOrdersClient, FakeUnitOfWork
tests/unit/test_domain.py                  - domain model unit tests
tests/unit/test_service.py                 - service layer + parser unit tests
tests/integration/test_orders_client.py    - HTTP adapter integration tests
```

## Edge Cases Handled

| Problem                   | Solution                                                                  |
|---------------------------|---------------------------------------------------------------------------|
| Context window overflow   | `parse_raw_orders` batches input                                          |
| LLM hallucination         | Pydantic validation on every LLM response                                  |
| Invalid SQL from LLM      | `generate_sql_query` retries up to 3×; validates output starts with SELECT |
| Missing order totals      | `LinearRegressionTotalPredictor` imputes missing values from training data |
| API unavailability        | `APIError` propagated as `AgentError` with message                        |
| Single vs. bulk fetch     | Routing in `_post_parse_dispatch`; single orders skip DB entirely         |
| No matching orders        | Returns `{"orders": []}` — never fails on empty result set                |
