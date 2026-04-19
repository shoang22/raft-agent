"""Abstract interfaces (ports) for infrastructure dependencies."""
import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


class AbstractLLM(abc.ABC):
    @abc.abstractmethod
    def invoke(self, messages: list) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def invoke_with_tools(self, messages: list, tools: list) -> "ToolCall":
        """Call the LLM with tool definitions; returns the chosen ToolCall."""
        raise NotImplementedError

    @abc.abstractmethod
    def invoke_structured(self, messages: list, schema: type) -> Any:
        """Call the LLM and return a validated instance of schema."""
        raise NotImplementedError


class AbstractOrdersClient(abc.ABC):
    @abc.abstractmethod
    def fetch_orders(self, limit: Optional[int] = None) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_order_by_id(self, order_id: str) -> str:
        raise NotImplementedError
