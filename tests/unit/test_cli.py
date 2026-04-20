"""Unit tests for the CLI entrypoint argument parsing."""
import json
import sys
import pytest
from unittest.mock import AsyncMock, patch


async def _run_cli_with_argv(argv: list[str]) -> tuple[str, int]:
    """Run cli.main() with the given sys.argv and capture stdout."""
    output_lines: list[str] = []

    def fake_print(*args, **kwargs):
        if kwargs.get("file") is None:
            output_lines.append(" ".join(str(a) for a in args))

    captured_query: list[str] = []

    def fake_bootstrap(**_kwargs):
        async def run(query: str):
            captured_query.append(query)
            return {"orders": []}
        return run

    with (
        patch.object(sys, "argv", argv),
        patch("builtins.print", side_effect=fake_print),
        patch("src.raft_agent.bootstrap.bootstrap", side_effect=fake_bootstrap),
    ):
        from src.raft_agent.entrypoints import cli
        await cli.main()

    return captured_query[0] if captured_query else "", output_lines


async def test_cli_strips_cli_flag_from_query():
    query, _ = await _run_cli_with_argv(
        ["main.py", "--cli", "Show me all orders from Ohio"]
    )
    assert query == "Show me all orders from Ohio"
    assert "--cli" not in query


async def test_cli_query_without_flag():
    query, _ = await _run_cli_with_argv(
        ["main.py", "Show me all orders from Ohio"]
    )
    assert query == "Show me all orders from Ohio"


async def test_cli_multi_word_query_preserved():
    query, _ = await _run_cli_with_argv(
        ["main.py", "--cli", "Which", "buyer", "spent", "the", "most"]
    )
    assert query == "Which buyer spent the most"
