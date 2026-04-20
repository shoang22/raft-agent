#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from raft_agent.entrypoints.cli import main

if __name__ == "__main__":
    asyncio.run(main())
