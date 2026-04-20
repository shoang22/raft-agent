#!/usr/bin/env python3
"""Entry point: python -m raft_agent.entrypoints.cli

Usage:
    python -m raft_agent.entrypoints.cli
    python -m raft_agent.entrypoints.cli "orders from Ohio over 500"
"""
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    from src.raft_agent.bootstrap import bootstrap
    from src.raft_agent.service_layer.agent import AgentError

    args = [a for a in sys.argv[1:] if a != "--cli"]
    if args:
        query = " ".join(args)
    else:
        print("Order Query Agent")
        print("Example: Show me all orders from Ohio with total over 500")
        print()
        query = input("Enter your query: ").strip()
        if not query:
            print("No query provided.")
            sys.exit(1)

    print(f"\nProcessing: {query!r}\n")
    try:
        run_agent = bootstrap()
        result = await run_agent(query)
        print(json.dumps(result, indent=2))
    except AgentError as e:
        logger.error("Agent failed: %s", e)
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
