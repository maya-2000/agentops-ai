"""Process-wide log formatting for the API (``LOG_FORMAT``, ``LOG_LEVEL``).

The application's loggers (``agentops.api``, ``agentops.agent``, ``agentops.security``) already emit
one allow-listed JSON object per event. ``JsonLogFormatter`` wraps each record into one JSON line
with ``timestamp`` (UTC), ``level`` and ``logger``, merging the event's own fields. Messages from
other libraries (uvicorn, for example) become a ``message`` field after secret and path redaction.
For an exception only its class name is kept (``exc_type``): tracebacks can carry SQL, paths or
values. That includes a traceback handed over as message text (Starlette does this for a failed
start-up), which is reduced to its exception type. ``LOG_FORMAT=text`` applies the same rules.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

from app.security.redaction import redact, redact_paths

MAX_MESSAGE_CHARS = 2000
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
TRACEBACK_OMITTED = "Traceback omitted."
_TRACEBACK_PREFIX = "Traceback (most recent call last):"
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


def safe_message(message: str) -> tuple[str, str | None]:
    """A library message fit for the log, and the exception type if the message was a traceback."""
    if message.lstrip().startswith(_TRACEBACK_PREFIX):
        match = _EXCEPTION_LINE.match(message.strip().splitlines()[-1])
        return TRACEBACK_OMITTED, match.group(1).rsplit(".", 1)[-1] if match else None
    return redact_paths(redact(message))[:MAX_MESSAGE_CHARS], None


def _exc_type(record: logging.LogRecord) -> str | None:
    return record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] is not None else None


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        message = record.getMessage()
        event = _json_object(message)
        exc_type = _exc_type(record)
        if event is not None:
            payload.update({k: v for k, v in event.items() if k not in ("level", "logger")})
        else:
            payload["message"], from_text = safe_message(message)
            exc_type = exc_type or from_text
        if exc_type:
            payload["exc_type"] = exc_type
        return json.dumps(payload, default=str)


class TextLogFormatter(logging.Formatter):
    """``LOG_FORMAT=text``: one readable line per record, under the same rules as the JSON format."""

    def __init__(self) -> None:
        super().__init__(TEXT_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        exc_type = _exc_type(record)
        if _json_object(message) is None:
            message, from_text = safe_message(message)
            exc_type = exc_type or from_text
        if exc_type:
            message = f"{message} exc_type={exc_type}"
        clean = logging.makeLogRecord({**record.__dict__, "msg": message, "args": None, "exc_info": None})
        clean.exc_text = None
        clean.stack_info = None
        return super().format(clean)


def _json_object(message: str) -> dict[str, Any] | None:
    if not message.startswith("{"):
        return None
    try:
        parsed = json.loads(message)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """One stderr handler on the root logger; uvicorn's loggers propagate to it."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter() if fmt == "json" else TextLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
