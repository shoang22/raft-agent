"""Composition root — assembles concrete dependencies and wires the application."""
import os
from typing import Any, Optional

import tiktoken
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from sqlalchemy.ext.asyncio import create_async_engine

from src.raft_agent.adapters.abstractions import AbstractLLM, AbstractOrdersClient, AbstractProgressReporter, ToolCall
from src.raft_agent.adapters.ml_model import AbstractTotalPredictor, LinearRegressionTotalPredictor
from src.raft_agent.adapters.orders_client import OrdersAPIClient
from src.raft_agent.adapters.unit_of_work import DEFAULT_TRAINING_DB_URL, SqlAlchemyUnitOfWork
from src.raft_agent.service_layer.agent import run_agent

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-oss-120b:exacto"


def build_tools(client: AbstractOrdersClient) -> list:
    """Build LangChain tool objects that delegate to the given client."""

    @tool
    async def fetch_orders() -> list:
        """Fetch all customer orders. Use for broad queries that filter, aggregate, or compare across many orders."""
        return await client.fetch_orders()

    @tool
    async def fetch_order_by_id(order_id: str) -> str:
        """Fetch a single customer order by its ID. Use when the query targets one specific order."""
        return await client.fetch_order_by_id(order_id)

    return [fetch_orders, fetch_order_by_id]


class _LangChainOpenAIAdapter(AbstractLLM):
    """Adapts a LangChain chat model to the AbstractLLM interface."""

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def __init__(self, model: Any, context_window: int) -> None:
        self._model: ChatOpenRouter = model
        super().__init__(context_window=context_window)

    async def invoke(self, messages: list) -> Any:
        return await self._model.ainvoke(messages)

    async def invoke_with_tools(self, messages: list, tools: list) -> ToolCall:
        response = await self._model.bind_tools(tools).ainvoke(messages)
        tc = response.tool_calls[0]
        return ToolCall(name=tc["name"], arguments=tc["args"])

    async def invoke_structured(self, messages: list, schema: type) -> Any:
        out = await self._model.with_structured_output(schema, include_raw=True).ainvoke(messages)
        return out

    def count_tokens(self, text: str) -> int:
        return len(self._ENCODING.encode(text))


def build_llm() -> AbstractLLM:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    context_window = 131_072
    model = ChatOpenRouter(
        model=MODEL_NAME,
        openrouter_api_key=api_key,
        openrouter_api_base=OPENROUTER_BASE_URL,
        temperature=0,
        openrouter_provider={"order": ["google-vertex"]}

    )
    return _LangChainOpenAIAdapter(model, context_window=context_window)


def bootstrap(
    client: AbstractOrdersClient = None,
    llm: AbstractLLM = None,
    predictor: AbstractTotalPredictor = None,
    training_db_url: str = None,
    reporter: AbstractProgressReporter = None,
):
    """Return a configured async run_agent callable with all dependencies wired up.

    A shared persistent training engine is created once and reused across calls
    so the training table accumulates data over time.  A fresh in-memory ephemeral
    engine is created per call so each agent run starts with a clean query table.
    Pass fake/stub implementations to override defaults (useful in tests).
    """
    from src.raft_agent.adapters.progress import CLIProgressReporter

    resolved_client = client if client is not None else OrdersAPIClient()
    resolved_llm = llm if llm is not None else build_llm()
    resolved_predictor = predictor if predictor is not None else LinearRegressionTotalPredictor()
    resolved_reporter = reporter if reporter is not None else CLIProgressReporter()

    db_url = training_db_url or DEFAULT_TRAINING_DB_URL
    training_engine = create_async_engine(db_url)

    async def _run(query: str, *, reporter: Optional[AbstractProgressReporter] = None) -> dict:
        return await run_agent(
            query,
            tools=build_tools(resolved_client),
            llm=resolved_llm,
            uow=SqlAlchemyUnitOfWork(training_engine=training_engine),
            predictor=resolved_predictor,
            reporter=reporter if reporter is not None else resolved_reporter,
        )

    return _run
