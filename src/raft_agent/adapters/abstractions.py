"""Abstract interfaces (ports) for infrastructure dependencies."""
import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


class AbstractLLM(abc.ABC):
    def __init__(self, context_window: int) -> None:
        self.context_window = context_window

    @abc.abstractmethod
    async def invoke(self, messages: list) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    async def invoke_with_tools(self, messages: list, tools: list) -> "ToolCall":
        """Call the LLM with tool definitions; returns the chosen ToolCall."""
        raise NotImplementedError

    @abc.abstractmethod
    async def invoke_structured(self, messages: list, schema: type) -> Any:
        """Call the LLM and return a validated instance of schema."""
        raise NotImplementedError

    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in text using the model's tokenizer."""
        raise NotImplementedError


class AbstractOrdersClient(abc.ABC):
    @abc.abstractmethod
    async def fetch_orders(self, limit: Optional[int] = None) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    async def fetch_order_by_id(self, order_id: str) -> str:
        raise NotImplementedError
