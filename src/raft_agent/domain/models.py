"""Domain model — Order value object and FilterCriteria with filtering behaviour."""
import logging
from enum import Enum
from typing import Optional
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class USState(str, Enum):
    AL = "AL"
    AK = "AK"
    AZ = "AZ"
    AR = "AR"
    CA = "CA"
    CO = "CO"
    CT = "CT"
    DE = "DE"
    FL = "FL"
    GA = "GA"
    HI = "HI"
    ID = "ID"
    IL = "IL"
    IN = "IN"
    IA = "IA"
    KS = "KS"
    KY = "KY"
    LA = "LA"
    ME = "ME"
    MD = "MD"
    MA = "MA"
    MI = "MI"
    MN = "MN"
    MS = "MS"
    MO = "MO"
    MT = "MT"
    NE = "NE"
    NV = "NV"
    NH = "NH"
    NJ = "NJ"
    NM = "NM"
    NY = "NY"
    NC = "NC"
    ND = "ND"
    OH = "OH"
    OK = "OK"
    OR = "OR"
    PA = "PA"
    RI = "RI"
    SC = "SC"
    SD = "SD"
    TN = "TN"
    TX = "TX"
    UT = "UT"
    VT = "VT"
    VA = "VA"
    WA = "WA"
    WV = "WV"
    WI = "WI"
    WY = "WY"


class Order(BaseModel):
    orderId: str | None = None
    """Unique order identifier (e.g. 'ORD-1042'). Null if unknown."""
    buyer: str | None = None
    """Full name of the customer who placed the order (e.g. 'Jane Smith'). Null if unknown."""
    state: USState | None = None
    """Two-letter US state code for the shipping destination (e.g. 'CA'). Null if unknown."""
    total: float | None = None
    """Order total in USD, rounded to two decimal places (e.g. 149.99). Null if unknown."""

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v < 0:
            raise ValueError("total must be non-negative")
        return round(v, 2)


class OrderStrict(BaseModel):
    orderId: str
    buyer: str
    state: USState
    total: float | None = None

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v < 0:
            raise ValueError("total must be non-negative")
        return round(v, 2)


class OrdersOutput(BaseModel):
    orders: list[Order]


class FilterCriteria(BaseModel):
    state: Optional[USState] = None
    min_total: Optional[float] = None
    max_total: Optional[float] = None
    buyer_name_contains: Optional[str] = None

    @field_validator("state", mode="before")
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
