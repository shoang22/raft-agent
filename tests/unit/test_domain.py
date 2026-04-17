"""Unit tests for domain model — no infrastructure, no fakes needed."""
import pytest

from raft_llm.domain.models import Order, FilterCriteria


class TestOrderModel:
    def test_valid_order(self):
        o = Order(orderId="1001", buyer="John Davis", state="OH", total=742.10)
        assert o.orderId == "1001"
        assert o.state == "OH"
        assert o.total == 742.10

    def test_state_is_normalized_to_uppercase(self):
        o = Order(orderId="1", buyer="X", state="oh", total=10.0)
        assert o.state == "OH"

    def test_total_is_rounded_to_two_decimals(self):
        o = Order(orderId="1", buyer="X", state="OH", total=742.1099999)
        assert o.total == 742.11

    def test_negative_total_raises(self):
        with pytest.raises(Exception):
            Order(orderId="1", buyer="X", state="OH", total=-5.0)

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            Order(orderId="1", buyer="X", state="OH")  # missing total


class TestFilterCriteria:
    def test_all_optional_fields_default_to_none(self):
        fc = FilterCriteria()
        assert fc.state is None
        assert fc.min_total is None

    def test_state_normalized_uppercase(self):
        fc = FilterCriteria(state="oh")
        assert fc.state == "OH"

    def test_with_all_fields(self):
        fc = FilterCriteria(state="OH", min_total=500.0, max_total=2000.0)
        assert fc.state == "OH"
        assert fc.min_total == 500.0


ORDERS = [
    Order(orderId="1001", buyer="John Davis", state="OH", total=742.10),
    Order(orderId="1002", buyer="Sarah Liu", state="TX", total=156.55),
    Order(orderId="1003", buyer="Mike Turner", state="OH", total=1299.99),
    Order(orderId="1004", buyer="Rachel Kim", state="WA", total=89.50),
    Order(orderId="1005", buyer="Chris Myers", state="OH", total=512.00),
]


class TestFilterCriteriaApply:
    def test_filter_by_state(self):
        result = FilterCriteria(state="OH").apply(ORDERS)
        assert len(result) == 3
        assert all(o.state == "OH" for o in result)

    def test_filter_by_min_total(self):
        result = FilterCriteria(min_total=500.0).apply(ORDERS)
        assert all(o.total >= 500.0 for o in result)

    def test_filter_by_state_and_min_total(self):
        result = FilterCriteria(state="OH", min_total=500.0).apply(ORDERS)
        assert len(result) == 3
        assert all(o.state == "OH" and o.total >= 500.0 for o in result)

    def test_filter_by_max_total(self):
        result = FilterCriteria(max_total=200.0).apply(ORDERS)
        assert all(o.total <= 200.0 for o in result)

    def test_filter_by_buyer_name(self):
        result = FilterCriteria(buyer_name_contains="Kim").apply(ORDERS)
        assert len(result) == 1
        assert result[0].buyer == "Rachel Kim"

    def test_no_criteria_returns_all(self):
        result = FilterCriteria().apply(ORDERS)
        assert len(result) == len(ORDERS)

    def test_no_matches_returns_empty_list(self):
        result = FilterCriteria(state="FL").apply(ORDERS)
        assert result == []

    def test_case_insensitive_state_filter(self):
        result = FilterCriteria(state="oh").apply(ORDERS)
        assert len(result) == 3
