"""HTTP adapter for the customer orders API."""
import logging
from typing import Optional

import httpx

from raft_agent.adapters.abstractions import AbstractOrdersClient

logger = logging.getLogger(__name__)


class APIError(Exception):
    pass


class OrdersAPIClient(AbstractOrdersClient):
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url.rstrip("/")

    async def fetch_orders(self, limit: Optional[int] = None) -> list:
        params = {}
        if limit is not None:
            params["limit"] = limit

        url = f"{self.base_url}/api/orders"
        logger.info("Fetching orders from %s params=%s", url, params)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise APIError(str(e)) from e
        except httpx.RequestError as e:
            raise APIError(str(e)) from e

        logger.info("Received orders response (%d bytes)", len(response.text))
        return response.json()["raw_orders"]

    async def fetch_order_by_id(self, order_id: str) -> str:
        url = f"{self.base_url}/api/order/{order_id}"
        logger.info("Fetching order %s from %s", order_id, url)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise APIError(str(e)) from e
        except httpx.RequestError as e:
            raise APIError(str(e)) from e

        return response.json()["raw_order"]
