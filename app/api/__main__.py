"""Start the API: ``python -m app.api`` (host and port from API_HOST / API_PORT; localhost by default)."""

from __future__ import annotations

import logging

import uvicorn

from app.api.config import APIConfig
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    config = APIConfig.from_settings(settings)
    logging.basicConfig(level=settings.log_level.upper(), format="%(message)s")
    # One worker: agent runs are serialised over a single read-only database connection.
    # The structured agentops.api request log replaces the access log (which would record client addresses).
    uvicorn.run("app.api.main:app", host=config.host, port=config.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
