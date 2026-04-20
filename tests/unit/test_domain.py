"""Unit tests for domain model — no infrastructure, no fakes needed."""
import pytest

from src.raft_agent.domain.models import PartialOrder


class TestOrderModel:
    def test_valid_order(self):
        o = PartialOrder(orderId="1001", buyer="John Davis", state="OH", total=742.10)
        assert o.orderId == "1001"
        assert o.state == "OH"
        assert o.total == 742.10

    def test_total_is_rounded_to_two_decimals(self):
        o = PartialOrder(orderId="1", buyer="X", state="OH", total=742.1099999)
        assert o.total == 742.11

    def test_negative_total_raises(self):
        with pytest.raises(Exception):
            PartialOrder(orderId="1", buyer="X", state="OH", total=-5.0)

    def test_missing_total_defaults_to_none(self):
        o = PartialOrder(orderId="1", buyer="X", state="OH")
        assert o.total is None

