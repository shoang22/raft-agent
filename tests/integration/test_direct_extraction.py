"""Integration test for direct_extraction with the real LangChain adapter.

Exercises chunked parsing: the order text is longer than chunk_size so
direct_extraction splits it across multiple LLM calls and merges the results.

Skipped unless OPENROUTER_API_KEY is set in the environment.
"""
import os

import pytest
from dotenv import load_dotenv

from src.raft_agent.domain.models import USState
from src.raft_agent.bootstrap import build_llm
from src.raft_agent.service_layer.parsers import (
    direct_extraction,
)

load_dotenv()


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY",
)

@pytest.mark.skip()
async def test_long_order_is_chunked_and_parsed():
    """Order text longer than chunk_size triggers multi-chunk parsing; merged result is correct."""
    llm = build_llm()
    context_window = llm.context_window
    long_items = ", ".join([f"item{i}" for i in range(context_window)])
    long_order = (
        "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, "
        f"Items: {long_items}"
    )
    order = await direct_extraction(long_order, llm)
    assert order.orderId == "1001"
    assert order.buyer == "John Davis"
    assert order.state == USState.OH
    assert order.total == 742.10

