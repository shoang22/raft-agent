"""Domain model — Order value object and FilterCriteria with filtering behaviour."""
import logging
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

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


class PartialOrder(BaseModel):
    orderId: str | None = Field(default=None, description="Unique order identifier (e.g. '1001'). Null if unknown.")
    buyer: str | None = Field(default=None, description="Full name of the customer who placed the order (e.g. 'Jane Smith'). Null if unknown.")
    state: USState | None = Field(default=None, description="Two-letter US state code for the shipping destination (e.g. 'CA'). Null if unknown.")
    total: float | None = Field(default=None, description="Order total in USD, rounded to two decimal places (e.g. 149.99). Null if unknown.")

    @field_validator("total")
    @classmethod
    def validate_total(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v < 0:
            raise ValueError("total must be non-negative")
        return round(v, 2)


class Order(BaseModel):
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
