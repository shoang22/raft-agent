#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    if "--cli" in sys.argv:
        import asyncio
        from src.raft_agent.entrypoints.cli import main
        asyncio.run(main())
    else:
        from src.raft_agent.entrypoints.gradio_app import demo
        demo.launch()
