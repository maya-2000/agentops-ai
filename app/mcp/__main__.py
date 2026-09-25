"""Start the AgentOps MCP server on stdio: ``python -m app.mcp`` (or the ``agentops-mcp`` script).

stdout carries the MCP protocol, so every log line goes to stderr. The database is checked before
the transport starts, so a missing database fails fast with a clear message and exit code 1.
``--list-tools`` prints the catalogue and exits without opening the database.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from typing import get_args

import anyio

from app.config import MCPLogLevel, MCPTransport
from app.database.factory import get_database
from app.mcp.config import MCPServerConfig
from app.mcp.registry import MCPToolRegistry
from app.mcp.server import serve_stdio
from app.tools.registry import ToolRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.mcp", description="AgentOps MCP server (read-only analytics).")
    parser.add_argument("--transport", choices=get_args(MCPTransport), help="transport (default: MCP_TRANSPORT)")
    parser.add_argument("--log-level", choices=get_args(MCPLogLevel), help="log level (default: MCP_LOG_LEVEL)")
    parser.add_argument("--list-tools", action="store_true", help="print the enabled tools and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = MCPServerConfig.from_settings()
    except ValueError as exc:
        print(f"Invalid MCP configuration: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 2
    updates = {k: v for k, v in (("transport", args.transport), ("log_level", args.log_level)) if v}
    config = config.model_copy(update=updates)
    logging.basicConfig(stream=sys.stderr, level=config.log_level, format="%(message)s")
    if args.list_tools:
        tools = MCPToolRegistry(ToolRegistry(), config.limits, enabled=config.listed_tools)
        for tool in tools.tools():
            print(f"{tool.name}\t{tool.title}")
        return 0
    try:
        get_database(read_only=True).close()  # fail fast, before the transport starts
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(
            f"AgentOps MCP server could not start: the business database is not available ({type(exc).__name__}). "
            "Check DATABASE_URL, or build the database with: python -m data.generator.generate",
            file=sys.stderr,
        )
        return 1
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C is a normal way to stop a local server
        anyio.run(serve_stdio, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
