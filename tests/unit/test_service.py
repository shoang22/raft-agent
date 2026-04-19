"""Service-layer tests — use fakes, no real infrastructure."""
import pytest

from raft_llm.adapters.abstractions import ToolCall
from raft_llm.bootstrap import build_tools
from raft_llm.domain.models import FilterCriteria, Order, OrdersOutput
from raft_llm.service_layer.agent import run_agent, AgentError
from raft_llm.service_layer.parsers import (
    parse_raw_orders,
    extract_filter_criteria,
    generate_sql_query,
    ParseError,
)
from tests.fakes import FakeLLM, FakeOrdersClient, FakeErrorOrdersClient, FakeUnitOfWork

SAMPLE_RAW_ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc",
    "Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00, Items: monitor",
]

RAW_OHIO_ORDERS = [
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, Items: laptop, hdmi cable",
    "Order 1002: Buyer=Sarah Liu, Location=Austin, TX, Total=$156.55, Items: headphones",
    "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99, Items: gaming pc, mouse",
]

_TOOL_FETCH_ALL = ToolCall(name="fetch_orders")
_TOOL_FETCH_1001 = ToolCall(name="fetch_order_by_id", arguments={"order_id": "1001"})


class TestParseRawOrders:
    def test_parses_ohio_orders(self):
        ohio_output = OrdersOutput(orders=[
            Order(orderId="1001", buyer="John Davis", state="OH", total=742.10),
            Order(orderId="1003", buyer="Mike Turner", state="OH", total=1299.99),
        ])
        llm = FakeLLM(responses=[], structured_responses=[ohio_output])
        orders = parse_raw_orders(RAW_OHIO_ORDERS[:2], llm)
        assert len(orders) >= 1
        assert all(isinstance(o, Order) for o in orders)

    def test_empty_input_returns_empty_list(self):
        llm = FakeLLM(responses=[])
        assert parse_raw_orders([], llm) == []

    def test_batches_large_input(self):
        single_order = OrdersOutput(orders=[Order(orderId="1", buyer="A", state="OH", total=100.0)])
        llm = FakeLLM(responses=[], structured_responses=[single_order])
        many_orders = [f"Order {i}: Buyer=A, Location=Columbus, OH, Total=$100" for i in range(25)]
        orders = parse_raw_orders(many_orders, llm, batch_size=10)
        # ceil(25/10) = 3 batches → 3 structured LLM calls, each returning 1 order
        assert llm.structured_call_count == 3
        assert len(orders) == 3

    def test_handles_alternative_format(self):
        alt_output = OrdersOutput(orders=[Order(orderId="2001", buyer="Jane", state="TX", total=99.99)])
        llm = FakeLLM(responses=[], structured_responses=[alt_output])
        orders = parse_raw_orders(["#2001 | Jane | Texas TX | 99.99 USD"], llm)
        assert len(orders) == 1


class TestExtractFilterCriteria:
    def test_extracts_state_and_min_total(self):
        criteria_response = FilterCriteria(state="OH", min_total=500.0)
        llm = FakeLLM(responses=[], structured_responses=[criteria_response])
        criteria = extract_filter_criteria(
            "Show me all orders where the buyer was located in Ohio and total value was over 500",
            llm,
        )
        assert criteria.state == "OH"
        assert criteria.min_total == 500.0
        assert criteria.max_total is None

    def test_extracts_state_only(self):
        criteria_response = FilterCriteria(state="TX")
        llm = FakeLLM(responses=[], structured_responses=[criteria_response])
        criteria = extract_filter_criteria("orders from Texas", llm)
        assert criteria.state == "TX"
        assert criteria.min_total is None


class TestGenerateSqlQuery:
    def test_returns_valid_select(self):
        llm = FakeLLM(["SELECT * FROM orders WHERE state = 'OH'"])
        sql = generate_sql_query("orders from Ohio", llm)
        assert sql.upper().startswith("SELECT")

    def test_strips_markdown_code_block(self):
        llm = FakeLLM(["```sql\nSELECT * FROM orders\n```"])
        sql = generate_sql_query("all orders", llm)
        assert sql == "SELECT * FROM orders"

    def test_non_select_raises_parse_error(self):
        llm = FakeLLM(["DROP TABLE orders"])
        with pytest.raises(ParseError):
            generate_sql_query("drop table", llm, max_retries=1)

    def test_invalid_response_raises_parse_error_after_retries(self):
        llm = FakeLLM(["not sql at all"])
        with pytest.raises(ParseError):
            generate_sql_query("some query", llm, max_retries=2)


_ORDER_1001 = OrdersOutput(orders=[Order(orderId="1001", buyer="John Davis", state="OH", total=742.10)])
_ORDER_1002 = OrdersOutput(orders=[Order(orderId="1002", buyer="Sarah Liu", state="TX", total=156.55)])
_ORDER_1003 = OrdersOutput(orders=[Order(orderId="1003", buyer="Mike Turner", state="OH", total=1299.99)])
_ORDER_1005 = OrdersOutput(orders=[Order(orderId="1005", buyer="Chris Myers", state="OH", total=512.00)])
_ORDERS_SAMPLE = [_ORDER_1001, _ORDER_1002, _ORDER_1003, _ORDER_1005]


class TestRunAgent:
    def _run(self, query, client, llm):
        return run_agent(query, tools=build_tools(client), llm=llm, uow=FakeUnitOfWork())

    def test_ohio_over_500_returns_matching_orders(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=["SELECT * FROM orders WHERE state = 'OH' AND total >= 500"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=_ORDERS_SAMPLE,
        )

        result = self._run("Show me all orders from Ohio with total over 500", client, llm)

        assert "orders" in result
        orders = result["orders"]
        assert len(orders) == 3
        assert all(o["state"] == "OH" for o in orders)
        assert all(o["total"] >= 500.0 for o in orders)

    def test_no_matching_orders_returns_empty_list(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=["SELECT * FROM orders WHERE state = 'FL'"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=_ORDERS_SAMPLE,
        )

        result = self._run("Orders from Florida", client, llm)
        assert result["orders"] == []

    def test_api_error_raises_agent_error(self):
        client = FakeErrorOrdersClient("Connection refused")
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
        )

        with pytest.raises(AgentError, match="Connection refused"):
            self._run("any query", client, llm)

    def test_output_schema_matches_spec(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )

        result = self._run("all orders", client, llm)

        assert isinstance(result, dict)
        assert "orders" in result
        for order in result["orders"]:
            assert {"orderId", "buyer", "state", "total"} <= order.keys()

    def test_result_totals_are_floats(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )

        result = self._run("all orders", client, llm)
        for order in result["orders"]:
            assert isinstance(order["total"], float)

    def test_uow_is_committed_after_store(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )
        uow = FakeUnitOfWork()

        run_agent("all orders", tools=build_tools(client), llm=llm, uow=uow)

        assert uow.committed is True

    def test_single_order_returns_matching_order(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )

        result = self._run("show me order 1001", client, llm)

        assert len(result["orders"]) == 1
        assert result["orders"][0]["orderId"] == "1001"
        assert result["orders"][0]["buyer"] == "John Davis"

    def test_single_order_not_found_raises_agent_error(self):
        client = FakeOrdersClient([])
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[ToolCall(name="fetch_order_by_id", arguments={"order_id": "9999"})],
        )

        with pytest.raises(AgentError, match="9999"):
            self._run("show me order 9999", client, llm)

    def test_single_order_skips_uow_commit(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )
        uow = FakeUnitOfWork()

        run_agent("show me order 1001", tools=build_tools(client), llm=llm, uow=uow)

        assert uow.committed is False

    def test_single_order_api_error_raises_agent_error(self):
        client = FakeErrorOrdersClient("timeout")
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
        )

        with pytest.raises(AgentError, match="timeout"):
            self._run("show me order 1001", client, llm)
