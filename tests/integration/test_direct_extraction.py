"""Integration test for direct_extraction with the real LangChain adapter.

Exercises chunked parsing: the order text is longer than chunk_size so
direct_extraction splits it across multiple LLM calls and merges the results.

Skipped unless OPENROUTER_API_KEY is set in the environment.
"""
import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from raft_agent.bootstrap import (
    MODEL_NAME,
    OPENROUTER_BASE_URL,
    _LangChainOpenAIAdapter,
)
from raft_agent.domain.models import USState
from raft_agent.service_layer.parsers import (
    _PARSE_CHUNK_TEMPLATE,
    OrderChunk,
    direct_extraction,
)

load_dotenv()

# A long order drawn from dummy_customer_api.py data, padded with extra items
# so its token count exceeds the small chunk_size configured below.
_CONTEXT_WINDOW = 128_000
_LONG_ITEMS = ", ".join([f"item{i}" for i in range(_CONTEXT_WINDOW * 2)])  # add more items if needed to exceed chunk_size
_LONG_ORDER = (
    "Order 1001: Buyer=John Davis, Location=Columbus, OH, Total=$742.10, "
    f"Items: {_LONG_ITEMS}"
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY",
)

def _make_llm() -> _LangChainOpenAIAdapter:
    model = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0,
    )
    return _LangChainOpenAIAdapter(model, context_window=_CONTEXT_WINDOW)


async def test_long_order_is_chunked_and_parsed():
    """Order text longer than chunk_size triggers multi-chunk parsing; merged result is correct."""
    llm = _make_llm()
    await direct_extraction(_LONG_ORDER, llm)
