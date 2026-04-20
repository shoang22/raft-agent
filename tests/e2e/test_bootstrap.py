"""E2E test — exercises the full bootstrap → run_agent → SQLite stack.

Stubs the LLM and orders API to avoid real network calls while keeping
a real in-memory SQLite database so the full persistence path is exercised.
"""
import pytest

from raft_agent.adapters.abstractions import ToolCall
from raft_agent.bootstrap import bootstrap
from raft_agent.domain.models import Order, USState
from raft_agent.service_layer.parsers import OrderChunk, OrderField
from tests.fakes import FakeLLM, FakeOrdersClient, FakeTotalPredictor

_RAW_ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
]

_PARSED_1001 = OrderChunk(
    orderId="1001", buyer="John Davis", state=USState.OH, total=742.10,
    last_field=OrderField["total"],
)
_PARSED_1002 = OrderChunk(
    orderId="1002", buyer="Sarah Liu", state=USState.TX, total=156.55,
    last_field=OrderField["total"],
)


def _make_run(query: str = "show all orders"):
    llm = FakeLLM(
        responses=["SELECT * FROM orders"],
        tool_call_responses=[ToolCall(name="fetch_orders")],
        structured_responses=[_PARSED_1001, _PARSED_1002],
    )
    client = FakeOrdersClient(_RAW_ORDERS)
    predictor = FakeTotalPredictor()
    run = bootstrap(
        client=client,
        llm=llm,
        predictor=predictor,
        training_db_url="sqlite+aiosqlite:///:memory:",
    )
    return run, llm, predictor


async def test_bootstrap_returns_all_orders():
    run, _, _ = _make_run()
    result = await run("show all orders")
    assert "orders" in result
    orders = result["orders"]
    assert len(orders) == 2
    order_ids = {o["orderId"] for o in orders}
    assert order_ids == {"1001", "1002"}


async def test_bootstrap_order_fields_are_populated():
    run, _, _ = _make_run()
    result = await run("show all orders")
    by_id = {o["orderId"]: o for o in result["orders"]}
    assert by_id["1001"]["buyer"] == "John Davis"
    assert by_id["1001"]["state"] == "OH"
    assert by_id["1001"]["total"] == pytest.approx(742.10)


async def test_bootstrap_triggers_predictor_retrain():
    run, _, predictor = _make_run()
    await run("show all orders")
    assert predictor.retrain_call_count == 1
    assert len(predictor.last_retrain_orders) == 2
