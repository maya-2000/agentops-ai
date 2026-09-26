"""Start the API: ``python -m app.api`` (or ``agentops-api``). ``--check-config`` only validates the settings.

Host and port come from ``API_HOST`` and ``API_PORT`` (localhost by default). Logs are JSON lines on
stderr (``LOG_FORMAT=json``). An unsafe configuration stops the process before it listens, with a
list of the settings to fix and none of their values. On SIGTERM or SIGINT, uvicorn stops
accepting connections and gives in-flight requests ``API_SHUTDOWN_GRACE_SECONDS`` to finish. The
agent service then cancels any run still going and closes the database.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from app.api.config import APIConfig
from app.config import settings_or_exit
from app.logs import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.api", description="AgentOps HTTP API")
    parser.add_argument("--check-config", action="store_true", help="validate the configuration and exit")
    args = parser.parse_args(argv)
    settings = settings_or_exit()
    configure_logging(settings.log_level, settings.log_format)
    config = APIConfig.from_settings(settings)
    problems = config.startup_problems()
    if problems:
        print("The API configuration is not safe to serve:", *(f"- {p}" for p in problems), sep="\n", file=sys.stderr)
        return 2
    if args.check_config:
        print(f"Configuration OK (environment={config.environment}, auth={config.auth_mode}).", file=sys.stderr)
        return 0
    # One worker: agent runs are serialised over a single read-only database connection. The structured
    # agentops.api request log replaces the access log (which would record client addresses).
    uvicorn.run(
        "app.api.main:app",
        host=config.host,
        port=config.port,
        workers=1,
        access_log=False,
        log_config=None,  # keep the JSON logging configured above
        timeout_graceful_shutdown=max(1, round(config.shutdown_grace_seconds)),
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
