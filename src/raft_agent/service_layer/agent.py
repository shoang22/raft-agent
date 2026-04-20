"""LangGraph agent for querying and filtering customer orders."""
import logging
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.raft_agent.adapters.abstractions import AbstractLLM, ToolCall
from src.raft_agent.adapters.ml_model import AbstractTotalPredictor
from src.raft_agent.adapters.orders_client import APIError
from src.raft_agent.adapters.unit_of_work import AbstractUnitOfWork
from src.raft_agent.domain.models import Order
from src.raft_agent.service_layer.parsers import ParseError, generate_sql_query, parse_raw_orders

logger = logging.getLogger(__name__)


class AgentError(Exception):
    pass


class AgentState(TypedDict):
    query: str
    tool_name: Optional[str]
    raw_orders: list[str]
    parsed_orders: list[Order]
    query_results: list[dict]
    error: Optional[str]


async def _fetch_node(state: AgentState, llm: AbstractLLM, tools: list) -> dict:
    logger.info("[fetch] LLM selecting tool for query: %r", state["query"])
    tool_call: ToolCall = await llm.invoke_with_tools(
        [{"role": "user", "content": state["query"]}],
        tools,
    )
    logger.info("[fetch] LLM chose tool: %s args=%s", tool_call.name, tool_call.arguments)
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        return {"tool_name": tool_call.name, "error": f"Unknown tool: {tool_call.name!r}"}
    try:
        raw = await tool.ainvoke(tool_call.arguments)
        raw_orders: list[str] = [raw] if isinstance(raw, str) else list(raw)
        return {"tool_name": tool_call.name, "raw_orders": raw_orders}
    except APIError as e:
        return {"tool_name": tool_call.name, "error": str(e)}


async def _parse_node(
    state: AgentState,
    llm: AbstractLLM,
    predictor: Optional[AbstractTotalPredictor],
) -> dict:
    if state.get("error"):
        return {}
    logger.info("[parse] Parsing %d raw orders", len(state["raw_orders"]))
    try:
        parsed = await parse_raw_orders(state["raw_orders"], llm, predictor=predictor)
        return {"parsed_orders": parsed}
    except ParseError as e:
        return {"error": str(e)}


async def _store_node(
    state: AgentState,
    uow: AbstractUnitOfWork,
    predictor: Optional[AbstractTotalPredictor],
) -> dict:
    if state.get("error"):
        return {}
    orders_to_store = [o for o in state["parsed_orders"] if o.total is not None]
    logger.info("[store] Storing %d orders in ephemeral table", len(orders_to_store))
    try:
        await uow.orders.add_all(orders_to_store)
        await uow.commit()

        logger.info("[store] Upserting %d orders into training table", len(orders_to_store))
        await uow.training.upsert_all(orders_to_store)

        if predictor is not None:
            await _retrain_from_training(uow, predictor)
        return {}
    except Exception as e:
        return {"error": str(e)}


async def _retrain_from_training(uow: AbstractUnitOfWork, predictor: AbstractTotalPredictor) -> None:
    """Load all training orders and kick off a background retrain."""
    try:
        all_orders = await uow.training.get_all()
        await predictor.retrain_async(all_orders)
    except Exception:
        logger.exception("[store] Failed to kick off model retrain")


async def _query_node(state: AgentState, llm: AbstractLLM, uow: AbstractUnitOfWork) -> dict:
    if state.get("error"):
        return {}
    logger.info("[query] Generating SQL for: %r", state["query"])
    try:
        sql = await generate_sql_query(state["query"], llm)
        results = await uow.orders.execute_query(sql)
        return {"query_results": results}
    except (ParseError, Exception) as e:
        return {"error": str(e)}


def _format_single_node(state: AgentState) -> dict:
    orders = [
        {"orderId": o.orderId, "buyer": o.buyer, "state": o.state, "total": o.total}
        for o in state["parsed_orders"]
    ]
    return {"query_results": orders}


def _should_continue(state: AgentState) -> str:
    return "error" if state.get("error") else "continue"


def _post_parse_dispatch(state: AgentState) -> str:
    if state.get("error"):
        return "error"
    return "single" if state["tool_name"] == "fetch_order_by_id" else "bulk"


def create_graph(
    llm: AbstractLLM,
    tools: list,
    uow: AbstractUnitOfWork,
    predictor: Optional[AbstractTotalPredictor] = None,
):
    builder = StateGraph(AgentState)

    async def fetch(state):
        return await _fetch_node(state, llm, tools)

    async def parse(state):
        return await _parse_node(state, llm, predictor)

    async def store(state):
        return await _store_node(state, uow, predictor)

    async def query(state):
        return await _query_node(state, llm, uow)

    builder.add_node("fetch", fetch)
    builder.add_node("parse", parse)
    builder.add_node("store", store)
    builder.add_node("query", query)
    builder.add_node("format_single", _format_single_node)

    builder.add_edge(START, "fetch")
    builder.add_conditional_edges("fetch", _should_continue, {"continue": "parse", "error": END})
    builder.add_conditional_edges(
        "parse",
        _post_parse_dispatch,
        {"single": "format_single", "bulk": "store", "error": END},
    )
    builder.add_conditional_edges("store", _should_continue, {"continue": "query", "error": END})
    builder.add_edge("query", END)
    builder.add_edge("format_single", END)

    return builder.compile()


async def run_agent(
    query: str,
    *,
    tools: list,
    llm: AbstractLLM,
    uow: AbstractUnitOfWork,
    predictor: Optional[AbstractTotalPredictor] = None,
) -> dict:
    """Run the order-querying agent and return clean JSON-serializable output.

    Raises AgentError on API failures or unrecoverable LLM errors.
    """
    logger.info("Agent starting for query: %r", query)
    graph = create_graph(llm, tools, uow, predictor=predictor)

    initial_state: AgentState = {
        "query": query,
        "tool_name": None,
        "raw_orders": [],
        "parsed_orders": [],
        "query_results": [],
        "error": None,
    }

    async with uow:
        final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        raise AgentError(final_state["error"])

    raw_results = final_state.get("query_results", [])
    orders = [
        {
            "orderId": r.get("order_id", r.get("orderId", "")),
            "buyer": r.get("buyer", ""),
            "state": r.get("state", ""),
            "total": r.get("total", 0.0),
        }
        for r in raw_results
    ]
    logger.info("Agent complete: %d orders returned", len(orders))
    return {"orders": orders}
