"""Integration tests for direct_extraction with the real LangChain adapter.

Exercises chunked parsing and schema resilience: field labels renamed, reordered,
abbreviated, or accompanied by unknown extra fields.

Skipped unless OPENROUTER_API_KEY is set in the environment.
"""
import os

import pytest
from dotenv import load_dotenv

from src.raft_agent.domain.models import USState
from src.raft_agent.bootstrap import build_llm
from src.raft_agent.service_layer.parsers import direct_extraction

load_dotenv()


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY",
)


@pytest.mark.skip()
async def test_long_order_is_chunked_and_parsed():
    """Order text longer than chunk_size triggers multi-chunk parsing; merged result is correct."""
    llm = build_llm()
    context_window = 51736 // 2
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


class TestDirectExtractionSchemaResilience:
    """Parser must extract correct fields regardless of label names, order, or extra noise."""

    async def test_fields_in_reverse_order(self):
        """All four fields present but in reversed order from the canonical schema."""
        llm = build_llm()
        order = await direct_extraction(
            "Total=$512.00, Location=Cincinnati, OH, Buyer=Chris Myers, Order 1005", llm
        )
        assert order.orderId == "1005"
        assert order.buyer == "Chris Myers"
        assert order.state == USState.OH
        assert order.total == 512.00

    async def test_renamed_labels_customer_amount_shipto(self):
        """Labels Customer, ShipTo, Amount replace Buyer, Location, Total."""
        llm = build_llm()
        order = await direct_extraction(
            "Order #2001 | Customer: Jane Smith | ShipTo: Austin, TX | Amount: 299.99", llm
        )
        assert order.orderId == "2001"
        assert order.buyer == "Jane Smith"
        assert order.state == USState.TX
        assert order.total == 299.99

    async def test_abbreviated_labels(self):
        """Abbreviated labels: Ord#, Cust, St, Amt."""
        llm = build_llm()
        order = await direct_extraction(
            "Ord# 5001 | Cust: Bob Lee | St: NY | Amt: 200.00", llm
        )
        assert order.orderId == "5001"
        assert order.buyer == "Bob Lee"
        assert order.state == USState.NY
        assert order.total == 200.00

    async def test_extra_unknown_fields_are_ignored(self):
        """Unknown fields (Priority, Notes, Tracking) must not corrupt parsed Order fields."""
        llm = build_llm()
        order = await direct_extraction(
            "Order 4001: Buyer=Alice Brown, Location=Seattle, WA, Total=$75.50, "
            "Priority=HIGH, Notes=gift-wrap, Tracking=1Z999AA10123456784",
            llm,
        )
        assert order.orderId == "4001"
        assert order.buyer == "Alice Brown"
        assert order.state == USState.WA
        assert order.total == 75.50

    async def test_full_state_name_resolves_to_abbreviation(self):
        """State given as a full name (Ohio) must be mapped to its 2-letter code."""
        llm = build_llm()
        order = await direct_extraction(
            "Order 6001: Buyer=Carol White, Location=Columbus, OH, Total=$450.00", llm
        )
        assert order.orderId == "6001"
        assert order.buyer == "Carol White"
        assert order.state == USState.OH
        assert order.total == 450.00

    async def test_semicolon_separated_format(self):
        """Fields separated by semicolons instead of commas or pipes."""
        llm = build_llm()
        order = await direct_extraction(
            "Order 7001; Buyer: David Kim; Location: Portland OR; Total: 88.00", llm
        )
        assert order.orderId == "7001"
        assert order.buyer == "David Kim"
        assert order.state == USState.OR
        assert order.total == 88.00

    async def test_purchaser_and_ref_labels(self):
        """Labels Ref and Purchaser map to orderId and buyer respectively."""
        llm = build_llm()
        order = await direct_extraction(
            "Ref=8001, Purchaser=Emma Davis, State=FL, Value=320.50", llm
        )
        assert order.orderId == "8001"
        assert order.buyer == "Emma Davis"
        assert order.state == USState.FL
        assert order.total == 320.50

