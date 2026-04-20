"""Concrete progress-reporter implementations."""
import asyncio
import sys

from src.raft_agent.adapters.abstractions import AbstractProgressReporter


class CLIProgressReporter(AbstractProgressReporter):
    """Writes progress lines to stderr, keeping stdout clean for JSON output."""

    async def report(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)


class NullProgressReporter(AbstractProgressReporter):
    """Discards all progress messages."""

    async def report(self, message: str) -> None:
        pass


class GradioProgressReporter(AbstractProgressReporter):
    """Puts progress messages onto an asyncio.Queue for consumption by a Gradio generator.

    Usage in a Gradio event handler:

        reporter = GradioProgressReporter()
        task = asyncio.create_task(run_agent(query, ..., reporter=reporter))
        async for message in reporter:
            yield message
        result = await task
        yield json.dumps(result)
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def report(self, message: str) -> None:
        await self._queue.put(message)

    def done(self) -> None:
        """Signal that no more messages will be produced."""
        self._queue.put_nowait(self._SENTINEL)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self._queue.get()
        if item is self._SENTINEL:
            raise StopAsyncIteration
        return item
