"""In-memory fakes for AbstractLLM, AbstractOrdersClient, and AbstractUnitOfWork.

Fakes assert on end state, not implementation details — no mock.patch needed.
"""
from typing import Any, Optional

from sqlalchemy.ext.asyncio import create_async_engine

from src.raft_agent.adapters.abstractions import AbstractLLM, AbstractOrdersClient, AbstractProgressReporter, ToolCall
from src.raft_agent.adapters.orders_client import APIError
from src.raft_agent.adapters.repository import (
    AbstractTrainingRepository,
    SqlAlchemyOrderRepository,
    _ephemeral_metadata,
)
from src.raft_agent.adapters.unit_of_work import AbstractUnitOfWork
from src.raft_agent.domain.models import Order


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM(AbstractLLM):
    """Returns pre-configured responses in sequence, cycling if exhausted.

    Pass tool_call_responses to pre-configure invoke_with_tools results.
    Pass structured_responses to pre-configure invoke_structured results.
    """

    def __init__(
        self,
        responses: list[str],
        tool_call_responses: list[ToolCall] | None = None,
        structured_responses: list[Any] | None = None,
        context_window: int = 100_000,
    ) -> None:
        super().__init__(context_window=context_window)
        self._responses = responses
        self._tool_call_responses = tool_call_responses or []
        self._structured_responses = structured_responses or []
        self.call_count = 0
        self.tool_call_count = 0
        self.structured_call_count = 0
        self.structured_messages_log: list[list] = []

    async def invoke(self, messages: list) -> _FakeResponse:
        content = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return _FakeResponse(content)

    async def invoke_with_tools(self, messages: list, tools: list) -> ToolCall:
        tool_call = self._tool_call_responses[self.tool_call_count % len(self._tool_call_responses)]
        self.tool_call_count += 1
        return tool_call

    async def invoke_structured(self, messages: list, schema: type) -> Any:
        self.structured_messages_log.append(messages)
        obj = self._structured_responses[self.structured_call_count % len(self._structured_responses)]
        self.structured_call_count += 1
        if isinstance(obj, Exception):
            return {'parsed': None, 'parsing_error': obj, 'raw': None}
        return {'parsed': obj, 'parsing_error': None, 'raw': None}

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeOrdersClient(AbstractOrdersClient):
    """Returns raw order strings extracted from the API response envelope."""

    def __init__(self, raw_orders: list[str]) -> None:
        self._orders = raw_orders

    async def fetch_orders(self, limit: Optional[int] = None) -> list[str]:
        return self._orders[:limit] if limit is not None else self._orders

    async def fetch_order_by_id(self, order_id: str) -> str:
        for order in self._orders:
            if order_id in order:
                return order
        raise APIError(f"Order {order_id} not found")


class FakeTrainingRepository(AbstractTrainingRepository):
    """In-memory training store for test assertions."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self.upsert_call_count = 0

    async def upsert_all(self, orders: list[Order]) -> None:
        self.upsert_call_count += 1
        for order in orders:
            if order.total is not None:
                self._orders[order.orderId] = order

    async def get_all(self) -> list[Order]:
        return list(self._orders.values())


class FakeUnitOfWork(AbstractUnitOfWork):
    """SQLite in-memory UoW with committed tracking for test assertions.

    Uses a real async in-memory SQLAlchemy connection for ephemeral orders (so
    generated SQL queries execute correctly) and a FakeTrainingRepository for the
    training store (so tests can inspect writes without touching the filesystem).
    """

    def __init__(self) -> None:
        self._ephemeral_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        self.training = FakeTrainingRepository()
        self.committed = False

    async def __aenter__(self):
        async with self._ephemeral_engine.begin() as conn:
            await conn.run_sync(_ephemeral_metadata.create_all)
        self._conn = await self._ephemeral_engine.connect()
        self.orders = SqlAlchemyOrderRepository(self._conn)
        return self

    async def commit(self) -> None:
        await self._conn.commit()
        self.committed = True

    async def rollback(self) -> None:
        if hasattr(self, "_conn"):
            await self._conn.rollback()

    async def close(self) -> None:
        if hasattr(self, "_conn"):
            await self._conn.close()


class FakeErrorOrdersClient(AbstractOrdersClient):
    """Always raises APIError — simulates a failing API."""

    def __init__(self, message: str) -> None:
        self._message = message

    async def fetch_orders(self, limit: Optional[int] = None) -> list:
        raise APIError(self._message)

    async def fetch_order_by_id(self, order_id: str) -> str:
        raise APIError(self._message)


class FakeProgressReporter(AbstractProgressReporter):
    """Records reported messages so tests can assert on them."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def report(self, message: str) -> None:
        self.messages.append(message)


class FakeTotalPredictor:
    """Fake predictor for testing imputation and retrain triggering."""

    def __init__(self, predicted_value: float | None = 100.0) -> None:
        self._value = predicted_value
        self.retrain_call_count = 0
        self.last_retrain_orders: list = []

    def predict(self, order) -> float | None:
        return self._value

    def retrain(self, orders: list) -> None:
        self.retrain_call_count += 1
        self.last_retrain_orders = list(orders)

    async def retrain_async(self, orders: list) -> None:
        self.retrain(orders)
