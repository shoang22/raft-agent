"""Composition root — assembles concrete dependencies and wires the application."""
import os
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from raft_llm.adapters.abstractions import AbstractLLM, AbstractOrdersClient, ToolCall
from raft_llm.adapters.orders_client import OrdersAPIClient
from raft_llm.adapters.unit_of_work import SqlAlchemyUnitOfWork
from raft_llm.service_layer.agent import run_agent

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-oss-120b:exacto"


def build_tools(client: AbstractOrdersClient) -> list:
    """Build LangChain tool objects that delegate to the given client."""

    @tool
    def fetch_orders() -> list[str]:
        """Fetch all customer orders. Use for broad queries that filter, aggregate, or compare across many orders."""
        return [client.fetch_orders()]

    @tool
    def fetch_order_by_id(order_id: str) -> list[str]:
        """Fetch a single customer order by its ID. Use when the query targets one specific order."""
        return [client.fetch_order_by_id(order_id)]

    return [fetch_orders, fetch_order_by_id]


class _LangChainAdapter(AbstractLLM):
    """Adapts a LangChain chat model to the AbstractLLM interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def invoke(self, messages: list) -> Any:
        return self._model.invoke(messages)

    def invoke_with_tools(self, messages: list, tools: list) -> ToolCall:
        response = self._model.bind_tools(tools).invoke(messages)
        tc = response.tool_calls[0]
        return ToolCall(name=tc["name"], arguments=tc["args"])

    def invoke_structured(self, messages: list, schema: type) -> Any:
        return self._model.with_structured_output(schema).invoke(messages)


def build_llm() -> AbstractLLM:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )
    return _LangChainAdapter(model)


def bootstrap(
    client: AbstractOrdersClient = None,
    llm: AbstractLLM = None,
):
    """Return a configured run_agent callable with all dependencies wired up.

    Pass fake/stub implementations to override defaults (useful in tests).
    A fresh SQLite in-memory database is created for each invocation.
    """
    resolved_client = client if client is not None else OrdersAPIClient()
    resolved_llm = llm if llm is not None else build_llm()

    def _run(query: str) -> dict:
        return run_agent(
            query,
            tools=build_tools(resolved_client),
            llm=resolved_llm,
            uow=SqlAlchemyUnitOfWork(),
        )

    return _run
