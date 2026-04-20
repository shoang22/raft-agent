# Architecture

## Overview

```
User NL Query
     |
     v
+----------------------+
| entrypoints/cli.py   |
+----------------------+
     |
     v
+----------------------+
|   bootstrap.py       |  <- composition root; wires LLM, tools, UoW
+----------------------+
     |
     v
+---------------------------------------------------------------+
|              service_layer/agent.py (LangGraph)               |
|                                                               |
|  AgentState                                                   |
|  +-------------+                                             |
|  | query       |                                             |
|  | tool_name   |  decide --> execute --> parse --> store --> |
|  | tool_args   |                                    query    |
|  | raw_orders  |                                             |
|  | parsed_     |   (single-order path skips store/query)     |
|  | orders      |                                             |
|  | query_      |                                             |
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
│   ├── abstractions.py    # AbstractLLM, AbstractOrdersClient, ToolCall
│   ├── orders_client.py   # HTTP client → OrdersAPIClient
│   ├── repository.py      # AbstractOrderRepository, SqlAlchemyOrderRepository
│   └── unit_of_work.py    # AbstractUnitOfWork, SqlAlchemyUnitOfWork
├── service_layer/
│   ├── agent.py           # LangGraph graph, AgentState, AgentError, run_agent
│   └── parsers.py         # parse_raw_orders, generate_sql_query, ParseError
├── entrypoints/
│   └── cli.py             # thin CLI; delegates to bootstrap
└── bootstrap.py           # composition root; build_tools, build_llm, bootstrap
```

## Graph Nodes

| Node                  | Responsibility                                                          |
|-----------------------|-------------------------------------------------------------------------|
| `_decide_node`        | LLM selects tool (`fetch_orders` or `fetch_order_by_id`) + args         |
| `_execute_tool_node`  | Invokes selected tool; stores raw text response in state                |
| `_parse_node`         | LLM parses raw text → `list[Order]` (batched)                          |
| `_store_node`         | Inserts parsed orders into in-memory SQLite via UoW                     |
| `_query_node`         | LLM generates SQL SELECT; executes against stored orders; returns rows  |
| `_format_single_node` | Formats a single-order result (skips DB; used for `fetch_order_by_id`) |

## Routing

```
_decide_node
    └─► _execute_tool_node
            └─► _parse_node
                    ├─ fetch_order_by_id ──► _format_single_node ──► END
                    └─ fetch_orders      ──► _store_node ──► _query_node ──► END

Any node sets state["error"] → END (raises AgentError to caller)
```

## Key Files

```
main.py                              - shell entry point (delegates to cli.py)
src/raft_agent/entrypoints/cli.py      - CLI: loads .env, calls bootstrap(), prints JSON
src/raft_agent/bootstrap.py            - wires LLM + tools + UoW; returns run(query) callable
src/raft_agent/service_layer/agent.py  - LangGraph graph definition and run_agent()
src/raft_agent/service_layer/parsers.py - parse_raw_orders, generate_sql_query
src/raft_agent/adapters/abstractions.py - AbstractLLM, AbstractOrdersClient
src/raft_agent/adapters/orders_client.py - HTTP adapter; raises APIError
src/raft_agent/adapters/repository.py  - SQLAlchemy Core repository
src/raft_agent/adapters/unit_of_work.py - transaction boundary
src/raft_agent/domain/models.py        - Order, FilterCriteria, OrdersOutput
tests/fakes.py                       - FakeLLM, FakeOrdersClient, FakeUnitOfWork
tests/unit/test_domain.py            - domain model unit tests
tests/unit/test_service.py           - service layer + parser unit tests
tests/integration/test_orders_client.py - HTTP adapter integration tests
```

## Edge Cases Handled

| Problem                   | Solution                                                                  |
|---------------------------|---------------------------------------------------------------------------|
| Context window overflow   | `parse_raw_orders` batches input (configurable `batch_size`, default 1)   |
| LLM hallucination         | Pydantic validation on every LLM response; invalid items skipped          |
| Invalid SQL from LLM      | `generate_sql_query` retries up to 3×; validates output starts with SELECT |
| API unavailability        | `APIError` propagated as `AgentError` with message                        |
| Single vs. bulk fetch     | Routing in `_post_parse_dispatch`; single orders skip DB entirely         |
| No matching orders        | Returns `{"orders": []}` — never fails on empty result set                |
