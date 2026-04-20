"""Service-layer tests — use fakes, no real infrastructure."""
import pytest

from src.raft_agent.adapters.abstractions import ToolCall
from src.raft_agent.bootstrap import build_tools
from src.raft_agent.domain.models import FilterCriteria, Order
from src.raft_agent.service_layer.agent import run_agent, AgentError
from src.raft_agent.service_layer.parsers import (
    _PARSE_CHUNK_TEMPLATE,
    direct_extraction,
    parse_raw_orders,
    generate_sql_query,
    chunk_by_words,
    OrderChunk,
    OrderField,
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
    async def test_parses_ohio_orders(self):
        chunk1 = OrderChunk(orderId="1001", buyer="John Davis", state="OH", total=742.10, last_field=OrderField.total)
        chunk2 = OrderChunk(orderId="1002", buyer="Sarah Liu", state="TX", total=156.55, last_field=OrderField.total)
        llm = FakeLLM(responses=[], structured_responses=[chunk1, chunk2])
        orders = await parse_raw_orders(RAW_OHIO_ORDERS[:2], llm)
        assert len(orders) >= 1
        assert all(isinstance(o, Order) for o in orders)

    async def test_empty_input_returns_empty_list(self):
        llm = FakeLLM(responses=[])
        assert await parse_raw_orders([], llm) == []

    async def test_processes_multiple_orders_individually(self):
        chunk = OrderChunk(orderId="1", buyer="A", state="OH", total=100.0, last_field=OrderField.total)
        llm = FakeLLM(responses=[], structured_responses=[chunk])
        many_orders = [f"Order {i}: Buyer=A, Location=Columbus, OH, Total=$100" for i in range(5)]
        orders = await parse_raw_orders(many_orders, llm)
        # one direct_extraction call per order → 5 structured LLM calls
        assert llm.structured_call_count == 5
        assert len(orders) == 5

    async def test_handles_alternative_format(self):
        chunk = OrderChunk(orderId="2001", buyer="Jane", state="TX", total=99.99, last_field=OrderField.total)
        llm = FakeLLM(responses=[], structured_responses=[chunk])
        orders = await parse_raw_orders(["#2001 | Jane | Texas TX | 99.99 USD"], llm)
        assert len(orders) == 1


class TestGenerateSqlQuery:
    async def test_returns_valid_select(self):
        llm = FakeLLM(["SELECT * FROM orders WHERE state = 'OH'"])
        sql = await generate_sql_query("orders from Ohio", llm)
        assert sql.upper().startswith("SELECT")

    async def test_strips_markdown_code_block(self):
        llm = FakeLLM(["```sql\nSELECT * FROM orders\n```"])
        sql = await generate_sql_query("all orders", llm)
        assert sql == "SELECT * FROM orders"

    async def test_non_select_raises_parse_error(self):
        llm = FakeLLM(["DROP TABLE orders"])
        with pytest.raises(ParseError):
            await generate_sql_query("drop table", llm, max_retries=1)

    async def test_invalid_response_raises_parse_error_after_retries(self):
        llm = FakeLLM(["not sql at all"])
        with pytest.raises(ParseError):
            await generate_sql_query("some query", llm, max_retries=2)


_ORDER_1001 = OrderChunk(orderId="1001", buyer="John Davis", state="OH", total=742.10, last_field=OrderField.total)
_ORDER_1002 = OrderChunk(orderId="1002", buyer="Sarah Liu", state="TX", total=156.55, last_field=OrderField.total)
_ORDER_1003 = OrderChunk(orderId="1003", buyer="Mike Turner", state="OH", total=1299.99, last_field=OrderField.total)
_ORDER_1005 = OrderChunk(orderId="1005", buyer="Chris Myers", state="OH", total=512.00, last_field=OrderField.total)
_ORDERS_SAMPLE = [_ORDER_1001, _ORDER_1002, _ORDER_1003, _ORDER_1005]


class TestRunAgent:
    async def _run(self, query, client, llm):
        return await run_agent(query, tools=build_tools(client), llm=llm, uow=FakeUnitOfWork())

    async def test_ohio_over_500_returns_matching_orders(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=["SELECT * FROM orders WHERE state = 'OH' AND total >= 500"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=_ORDERS_SAMPLE,
        )

        result = await self._run("Show me all orders from Ohio with total over 500", client, llm)

        assert "orders" in result
        orders = result["orders"]
        assert len(orders) == 3
        assert all(o["state"] == "OH" for o in orders)
        assert all(o["total"] >= 500.0 for o in orders)

    async def test_no_matching_orders_returns_empty_list(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=["SELECT * FROM orders WHERE state = 'FL'"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=_ORDERS_SAMPLE,
        )

        result = await self._run("Orders from Florida", client, llm)
        assert result["orders"] == []

    async def test_api_error_raises_agent_error(self):
        client = FakeErrorOrdersClient("Connection refused")
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
        )

        with pytest.raises(AgentError, match="Connection refused"):
            await self._run("any query", client, llm)

    async def test_output_schema_matches_spec(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )

        result = await self._run("all orders", client, llm)

        assert isinstance(result, dict)
        assert "orders" in result
        for order in result["orders"]:
            assert {"orderId", "buyer", "state", "total"} <= order.keys()

    async def test_result_totals_are_floats(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )

        result = await self._run("all orders", client, llm)
        for order in result["orders"]:
            assert isinstance(order["total"], float)

    async def test_uow_is_committed_after_store(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )
        uow = FakeUnitOfWork()

        await run_agent("all orders", tools=build_tools(client), llm=llm, uow=uow)

        assert uow.committed is True

    async def test_single_order_returns_matching_order(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )

        result = await self._run("show me order 1001", client, llm)

        assert len(result["orders"]) == 1
        assert result["orders"][0]["orderId"] == "1001"
        assert result["orders"][0]["buyer"] == "John Davis"

    async def test_single_order_not_found_raises_agent_error(self):
        client = FakeOrdersClient([])
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[ToolCall(name="fetch_order_by_id", arguments={"order_id": "9999"})],
        )

        with pytest.raises(AgentError, match="9999"):
            await self._run("show me order 9999", client, llm)

    async def test_single_order_skips_uow_commit(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )
        uow = FakeUnitOfWork()

        await run_agent("show me order 1001", tools=build_tools(client), llm=llm, uow=uow)

        assert uow.committed is False

    async def test_single_order_api_error_raises_agent_error(self):
        client = FakeErrorOrdersClient("timeout")
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
        )

        with pytest.raises(AgentError, match="timeout"):
            await self._run("show me order 1001", client, llm)


# ---------------------------------------------------------------------------
# Linear regression predictor — imputation tests
# ---------------------------------------------------------------------------

from src.raft_agent.service_layer.parsers import _impute_totals
from tests.fakes import FakeTotalPredictor


class TestImputeTotals:
    def test_order_with_total_passes_through_unchanged(self):
        order = Order(orderId="1", buyer="Alice", state="OH", total=500.0)
        result = _impute_totals([order], predictor=None)
        assert result == [order]

    def test_order_missing_total_is_imputed_by_predictor(self):
        order = Order(orderId="2", buyer="Bob", state="TX", total=None)
        predictor = FakeTotalPredictor(predicted_value=250.0)
        result = _impute_totals([order], predictor=predictor)
        assert len(result) == 1
        assert result[0].total == 250.0
        assert result[0].orderId == "2"

    def test_order_missing_total_dropped_when_no_predictor(self):
        order = Order(orderId="3", buyer="Carol", state="CA", total=None)
        result = _impute_totals([order], predictor=None)
        assert result == []

    def test_order_missing_total_dropped_when_predictor_untrained(self):
        order = Order(orderId="4", buyer="Dan", state="NY", total=None)
        predictor = FakeTotalPredictor(predicted_value=None)
        result = _impute_totals([order], predictor=predictor)
        assert result == []

    def test_mixed_orders_only_imputes_missing(self):
        order_with = Order(orderId="5", buyer="Eve", state="FL", total=99.0)
        order_without = Order(orderId="6", buyer="Frank", state="FL", total=None)
        predictor = FakeTotalPredictor(predicted_value=50.0)
        result = _impute_totals([order_with, order_without], predictor=predictor)
        assert len(result) == 2
        assert result[0].total == 99.0
        assert result[1].total == 50.0


class TestParseRawOrdersWithPredictor:
    async def test_passes_predictor_to_impute_step(self):
        chunk = OrderChunk(orderId="10", buyer="Grace", state="OH", total=None, last_field=OrderField.buyer)
        predictor = FakeTotalPredictor(predicted_value=123.45)
        llm = FakeLLM(responses=[], structured_responses=[chunk])

        orders = await parse_raw_orders(["Order 10: Buyer=Grace, Location=Columbus, OH"], llm, predictor=predictor)

        assert len(orders) == 1
        assert orders[0].total == 123.45


class TestRetrainTriggeredAfterStore:
    async def _run(self, query, client, llm, predictor):
        uow = FakeUnitOfWork()
        await run_agent(query, tools=build_tools(client), llm=llm, uow=uow, predictor=predictor)
        return predictor

    async def test_retrain_called_after_bulk_store(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )
        predictor = FakeTotalPredictor(predicted_value=100.0)

        predictor = await self._run("all orders", client, llm, predictor)

        assert predictor.retrain_call_count == 1

    async def test_retrain_not_called_for_single_order_fetch(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )
        predictor = FakeTotalPredictor(predicted_value=100.0)

        predictor = await self._run("show me order 1001", client, llm, predictor)

        assert predictor.retrain_call_count == 0


class TestChunkByWords:
    """FakeLLM.count_tokens returns len(text.split()), so chunk_size is a word limit."""

    def test_empty_string_returns_empty_list(self):
        llm = FakeLLM(responses=[])
        assert chunk_by_words("", chunk_size=5, llm=llm) == []

    def test_text_within_limit_is_single_chunk(self):
        llm = FakeLLM(responses=[])
        result = chunk_by_words("one two three", chunk_size=10, llm=llm)
        assert result == ["one two three"]

    def test_text_splits_into_multiple_chunks(self):
        llm = FakeLLM(responses=[])
        # "one two" = 2 tokens, "three four" = 2 tokens with chunk_size=2
        result = chunk_by_words("one two three four", chunk_size=2, llm=llm)
        assert result == ["one two", "three four"]

    def test_each_chunk_does_not_exceed_chunk_size(self):
        llm = FakeLLM(responses=[])
        words = " ".join(f"word{i}" for i in range(20))
        chunks = chunk_by_words(words, chunk_size=3, llm=llm)
        for chunk in chunks:
            assert llm.count_tokens(chunk) <= 3

    def test_all_words_preserved_across_chunks(self):
        llm = FakeLLM(responses=[])
        text = "alpha beta gamma delta epsilon zeta"
        chunks = chunk_by_words(text, chunk_size=2, llm=llm)
        assert " ".join(chunks) == text

    def test_single_word_exceeding_limit_is_own_chunk(self):
        llm = FakeLLM(responses=[])
        # chunk_size=0 forces each word to its own chunk (candidate always > 0)
        result = chunk_by_words("hello world", chunk_size=0, llm=llm)
        assert result == ["hello", "world"]

    def test_exact_chunk_boundary(self):
        llm = FakeLLM(responses=[])
        result = chunk_by_words("a b c d", chunk_size=2, llm=llm)
        assert result == ["a b", "c d"]


class TestDirectExtraction:
    """FakeLLM.count_tokens returns len(text.split()), so context_window is a word limit."""

    async def test_single_chunk_populates_all_fields(self):
        chunk = OrderChunk(orderId="1005", buyer="Chris Myers", state="OH", total=512.00, last_field=OrderField.total)
        llm = FakeLLM(responses=[], structured_responses=[chunk], context_window=10_000)
        order = await direct_extraction("Order 1005: Buyer=Chris Myers, Location=Cincinnati, OH, Total=$512.00", llm)
        assert order.orderId == "1005"
        assert order.buyer == "Chris Myers"
        assert order.state.value == "OH"
        assert order.total == 512.00

    async def test_multi_chunk_concatenates_split_buyer(self):
        # context_window=745 → chunk_size=(745-607-132)//2=3, so 6-word query splits into 2 chunks of ≤3 words
        chunk1 = OrderChunk(orderId="1005", buyer="Chris", last_field=OrderField.buyer)
        chunk2 = OrderChunk(buyer="Myers", state="OH", total=512.00, last_field=OrderField.total)
        llm = FakeLLM(responses=[], structured_responses=[chunk1, chunk2], context_window=745)
        order = await direct_extraction("Order 1005 Buyer=Chris Myers Location=OH Total=512.00", llm)
        assert order.orderId == "1005"
        assert order.buyer == "Chris Myers"
        assert order.state.value == "OH"
        assert order.total == 512.00
        assert llm.structured_call_count == 2

    async def test_none_fields_in_later_chunk_do_not_overwrite(self):
        # context_window=743 → chunk_size=(743-607-132)//2=2, so 4-word query splits into 2 chunks of ≤2 words
        chunk1 = OrderChunk(orderId="1005", buyer="Chris", last_field=OrderField.buyer)
        chunk2 = OrderChunk(buyer="Myers", last_field=OrderField.buyer)
        llm = FakeLLM(responses=[], structured_responses=[chunk1, chunk2], context_window=743)
        order = await direct_extraction("Order 1005 Buyer=Chris Myers", llm)
        assert order.orderId == "1005"
        assert order.buyer == "Chris Myers"

    async def test_context_window_too_small_raises_parse_error(self):
        llm = FakeLLM(responses=[], structured_responses=[], context_window=3)
        with pytest.raises(ParseError, match="Context window"):
            await direct_extraction("Order 1005", llm)


class TestTrainingTableWrites:
    async def _run_with_uow(self, query, client, llm, predictor=None):
        uow = FakeUnitOfWork()
        await run_agent(query, tools=build_tools(client), llm=llm, uow=uow, predictor=predictor)
        return uow

    async def test_training_table_receives_orders_after_bulk_store(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )

        uow = await self._run_with_uow("all orders", client, llm)

        assert uow.training.upsert_call_count == 1
        stored = await uow.training.get_all()
        assert len(stored) == 1
        assert stored[0].orderId == "1001"

    async def test_training_table_not_written_for_single_order_fetch(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS)
        llm = FakeLLM(
            responses=[],
            tool_call_responses=[_TOOL_FETCH_1001],
            structured_responses=[_ORDER_1001],
        )

        uow = await self._run_with_uow("show me order 1001", client, llm)

        assert uow.training.upsert_call_count == 0

    async def test_retrain_uses_training_table_data(self):
        client = FakeOrdersClient(SAMPLE_RAW_ORDERS[:1])
        llm = FakeLLM(
            responses=["SELECT * FROM orders"],
            tool_call_responses=[_TOOL_FETCH_ALL],
            structured_responses=[_ORDER_1001],
        )
        predictor = FakeTotalPredictor(predicted_value=100.0)

        uow = await self._run_with_uow("all orders", client, llm, predictor=predictor)

        training_ids = {o.orderId for o in await uow.training.get_all()}
        retrained_ids = {o.orderId for o in predictor.last_retrain_orders}
        assert training_ids == retrained_ids


class TestChunkSizeFormula:
    """Verify that each invoke_structured call's prompt + schema fits within the context window."""

    async def test_prompt_plus_schema_never_exceeds_context_window(self):
        captured: list[list[dict]] = []

        class CapturingFakeLLM(FakeLLM):
            async def invoke_structured(self, messages: list, schema: type):
                captured.append(messages)
                return await super().invoke_structured(messages, schema)

        probe = FakeLLM(responses=[])
        n_prompt = probe.count_tokens(_PARSE_CHUNK_TEMPLATE)
        n_schema = probe.count_tokens(str(OrderChunk.model_json_schema()))
        chunk_room = 10
        context_window = n_prompt + n_schema + chunk_room

        # All fields None so direct_extraction's post-parse validation is a no-op
        null_chunk = OrderChunk(last_field=OrderField.total)
        llm = CapturingFakeLLM(
            responses=[],
            structured_responses=[null_chunk],
            context_window=context_window,
        )
        # Query longer than chunk_room forces multiple chunks
        query = " ".join(f"word{i}" for i in range(chunk_room * 5))
        await direct_extraction(query, llm)

        assert len(captured) > 1, "query must produce multiple chunks to exercise the formula"
        for messages in captured:
            prompt_tokens = llm.count_tokens(messages[0]["content"])
            assert prompt_tokens + n_schema <= context_window, (
                f"prompt ({prompt_tokens}) + schema ({n_schema}) exceeds context_window ({context_window})"
            )
