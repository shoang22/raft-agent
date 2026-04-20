"""Integration tests for the HTTP orders client adapter."""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from src.raft_agent.adapters.orders_client import OrdersAPIClient, APIError


def _make_mock_client(response: MagicMock) -> AsyncMock:
    """Return an AsyncMock that behaves as an async context manager yielding an httpx client."""
    mock_client = AsyncMock()
    mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def _ok_response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.text = json.dumps(body)
    mock.json.return_value = body
    mock.raise_for_status = MagicMock()
    return mock


def _error_response(status_code: int) -> MagicMock:
    mock = MagicMock()
    mock.text = "Error"
    mock.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} Error",
        request=httpx.Request("GET", "http://localhost:5001/api/orders"),
        response=httpx.Response(status_code),
    )
    return mock


class TestFetchAllOrders:
    async def test_success_returns_raw_order_list(self):
        raw_orders = ["Order 1001: Buyer=John Davis"]
        mock_client = _make_mock_client(_ok_response({"status": "ok", "raw_orders": raw_orders}))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            result = await OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()
        assert result == raw_orders

    async def test_with_limit_passes_query_param(self):
        mock_client = _make_mock_client(_ok_response({"status": "ok", "raw_orders": []}))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            await OrdersAPIClient(base_url="http://localhost:5001").fetch_orders(limit=1)
        mock_client.get.assert_called_once_with(
            "http://localhost:5001/api/orders", params={"limit": 1}, timeout=10
        )

    async def test_server_error_raises_api_error(self):
        mock_client = _make_mock_client(_error_response(500))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(APIError):
                await OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()

    async def test_connection_error_raises_api_error(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(APIError, match="refused"):
                await OrdersAPIClient(base_url="http://localhost:5001").fetch_orders()


class TestFetchOrderById:
    async def test_found_returns_raw_order_text(self):
        raw_order = "Order 1003: Buyer=Mike Turner, Location=Cleveland, OH, Total=$1299.99"
        mock_client = _make_mock_client(_ok_response({"status": "ok", "raw_order": raw_order}))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            result = await OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1003")
        assert result == raw_order

    async def test_not_found_raises_api_error(self):
        mock_client = _make_mock_client(_error_response(404))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(APIError):
                await OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("9999")

    async def test_server_error_raises_api_error(self):
        mock_client = _make_mock_client(_error_response(500))
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(APIError):
                await OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1001")

    async def test_connection_error_raises_api_error(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("timeout")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        with patch("src.raft_agent.adapters.orders_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(APIError, match="timeout"):
                await OrdersAPIClient(base_url="http://localhost:5001").fetch_order_by_id("1001")
