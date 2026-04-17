"""Integration tests for the HTTP orders client adapter."""
import pytest
import requests
from unittest.mock import patch, MagicMock

from raft_llm.adapters.orders_client import OrdersAPIClient, APIError


def _mock_response(text: str, status_code: int = 200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error")
    else:
        mock.raise_for_status.return_value = None
    return mock


class TestFetchAllOrders:
    def test_success_returns_raw_response_text(self):
        body = '{"status":"ok","raw_orders":["Order 1001: Buyer=John Davis"]}'
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response(body)
            result = OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()
            assert result == body

    def test_with_limit_passes_query_param(self):
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response('{"status":"ok","raw_orders":[]}')
            OrdersAPIClient(base_url="http://localhost:5001").fetch_orders(limit=1)
            assert mock_get.call_args[1]["params"]["limit"] == 1

    def test_server_error_raises_api_error(self):
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response("Internal Server Error", status_code=500)
            with pytest.raises(APIError):
                OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()

    def test_connection_error_raises_api_error(self):
        with patch(
            "raft_llm.adapters.orders_client.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            with pytest.raises(APIError, match="refused"):
                OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()


class TestFetchOrderById:
    def test_found_returns_raw_response_text(self):
        body = '{"status":"ok","raw_order":"Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99"}'
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response(body)
            result = OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1003")
            assert result == body

    def test_not_found_raises_api_error(self):
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response('{"status":"not_found"}', status_code=404)
            with pytest.raises(APIError):
                OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("9999")

    def test_server_error_raises_api_error(self):
        with patch("raft_llm.adapters.orders_client.requests.get") as mock_get:
            mock_get.return_value = _mock_response("error", status_code=500)
            with pytest.raises(APIError):
                OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1001")

    def test_connection_error_raises_api_error(self):
        with patch(
            "raft_llm.adapters.orders_client.requests.get",
            side_effect=requests.RequestException("timeout"),
        ):
            with pytest.raises(APIError, match="timeout"):
                OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1001")
