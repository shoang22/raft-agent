"""Domain model — Order value object and FilterCriteria with filtering behaviour."""
import logging
from typing import Optional
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class Order(BaseModel):
    orderId: str
    buyer: str
    state: str
    total: float

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: float) -> float:
        if v < 0:
            raise ValueError("total must be non-negative")
        return round(v, 2)


class OrdersOutput(BaseModel):
    orders: list[Order]


class FilterCriteria(BaseModel):
    state: Optional[str] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    buyer_name_contains: Optional[str] = None

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.upper().strip()
        return v

    def apply(self, orders: list[Order]) -> list[Order]:
        result = orders

        if self.state is not None:
            result = [o for o in result if o.state == self.state]

        if self.min_total is not None:
            result = [o for o in result if o.total >= self.min_total]

        if self.max_total is not None:
            result = [o for o in result if o.total <= self.max_total]

        if self.buyer_name_contains is not None:
            needle = self.buyer_name_contains.lower()
            result = [o for o in result if needle in o.buyer.lower()]

        logger.info(
            "Filter applied: %s -> %d/%d orders match",
            self.model_dump(exclude_none=True),
            len(result),
            len(orders),
        )
        return result
