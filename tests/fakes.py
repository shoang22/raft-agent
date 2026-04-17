"""In-memory fakes for AbstractLLM, AbstractOrdersClient, and AbstractUnitOfWork.

Fakes assert on end state, not implementation details — no mock.patch needed.
"""
import json
from typing import Any, Optional

from raft_llm.adapters.abstractions import AbstractLLM, AbstractOrdersClient, ToolCall
from raft_llm.adapters.orders_client import APIError
from raft_llm.adapters.unit_of_work import AbstractUnitOfWork, SqlAlchemyUnitOfWork


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
    ) -> None:
        self._responses = responses
        self._tool_call_responses = tool_call_responses or []
        self._structured_responses = structured_responses or []
        self.call_count = 0
        self.tool_call_count = 0
        self.structured_call_count = 0

    def invoke(self, messages: list) -> _FakeResponse:
        content = self._responses[self.call_count % len(self._responses)]
        self.call_count += 1
        return _FakeResponse(content)

    def invoke_with_tools(self, messages: list, tools: list[dict]) -> ToolCall:
        tool_call = self._tool_call_responses[self.tool_call_count % len(self._tool_call_responses)]
        self.tool_call_count += 1
        return tool_call

    def invoke_structured(self, messages: list, schema: type) -> Any:
        obj = self._structured_responses[self.structured_call_count % len(self._structured_responses)]
        self.structured_call_count += 1
        return obj


class FakeOrdersClient(AbstractOrdersClient):
    """Returns fixed raw response text serialized from a list of order strings."""

    def __init__(self, raw_orders: list[str]) -> None:
        self._orders = raw_orders

    def fetch_orders(self, limit: Optional[int] = None) -> str:
        orders = self._orders[:limit] if limit is not None else self._orders
        return json.dumps({"status": "ok", "raw_orders": orders})

    def fetch_order_by_id(self, order_id: str) -> str:
        for order in self._orders:
            if order_id in order:
                return json.dumps({"status": "ok", "raw_order": order})
        raise APIError(f"Order {order_id} not found")


class FakeUnitOfWork(AbstractUnitOfWork):
    """SQLite in-memory UoW with committed tracking for test assertions."""

    def __init__(self) -> None:
        self._inner = SqlAlchemyUnitOfWork()
        self.orders = self._inner.orders
        self.committed = False

    def commit(self) -> None:
        self._inner.commit()
        self.committed = True

    def rollback(self) -> None:
        self._inner.rollback()

    def close(self) -> None:
        self._inner.close()


class FakeErrorOrdersClient(AbstractOrdersClient):
    """Always raises APIError — simulates a failing API."""

    def __init__(self, message: str) -> None:
        self._message = message

    def fetch_orders(self, limit: Optional[int] = None) -> str:
        raise APIError(self._message)

    def fetch_order_by_id(self, order_id: str) -> str:
        raise APIError(self._message)
