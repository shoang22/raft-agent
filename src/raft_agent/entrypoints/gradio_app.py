"""Gradio frontend for the RAFT order-querying agent.

Each node in the agent graph emits progress updates that stream into the
chat window in real time via an asyncio.Queue, before the final JSON result
is appended as a formatted table.

Run:
    python -m src.raft_agent.entrypoints.gradio_app
"""
import asyncio
import logging

import gradio as gr
from dotenv import load_dotenv

from src.raft_agent.adapters.progress import GradioProgressReporter, NullProgressReporter
from src.raft_agent.bootstrap import bootstrap
from src.raft_agent.service_layer.agent import AgentError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

# Bootstrap with NullProgressReporter — each request supplies its own GradioProgressReporter.
_run = bootstrap(reporter=NullProgressReporter())


def _orders_to_markdown(orders: list[dict]) -> str:
    if not orders:
        return "_No orders matched._"
    header = "| Order ID | Buyer | State | Total |\n|---|---|---|---|\n"
    rows = "".join(
        f"| {o['orderId']} | {o['buyer']} | {o['state']} | ${o['total']:.2f} |\n"
        for o in orders
    )
    return header + rows


async def _chat(message: str, history: list):
    reporter = GradioProgressReporter()

    # Start the agent in the background; signal the reporter when it's done.
    async def _run_and_signal():
        try:
            return await _run(message, reporter=reporter)
        finally:
            reporter.done()

    task = asyncio.create_task(_run_and_signal())

    # Stream progress messages as they arrive.
    partial = ""
    async for update in reporter:
        partial += f"_{update}_\n"
        yield partial

    # Await the final result and append it.
    try:
        result = await task
        orders = result.get("orders", [])
        partial += "\n" + _orders_to_markdown(orders)
    except AgentError as e:
        partial += f"\n**Error:** {e}"

    yield partial


demo = gr.ChatInterface(
    fn=_chat,
    title="RAFT Order Agent",
    description="Ask natural-language questions about customer orders.",
    examples=[
        "Show me all orders from Ohio over $500",
        "Which buyer spent the most?",
        "Show me order 1001",
    ],
)

if __name__ == "__main__":
    demo.launch()
