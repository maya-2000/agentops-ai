"""Redaction of secrets and internal details from anything that is logged, traced or shown.

``redact`` removes:

- well-known credential formats (Anthropic, OpenAI, AWS, GitHub, Slack and Google keys, JWTs,
  bearer tokens, private-key blocks),
- ``name=value`` / ``name: value`` pairs whose name looks secret (``api_key``, ``token``,
  ``password``, ``secret``, ``authorization``, ...),
- the literal values of secret-looking environment variables and of registered secrets (for
  example the configured API key), so that a key in an unusual format is still caught.

``redact_paths`` replaces absolute filesystem paths (error-message hygiene). ``redact_value``
applies both to strings nested in dicts and lists. Redaction is conservative: it can over-redact
a harmless string, and it cannot recognise a secret with no known pattern that is also not
registered. The main protection is that secrets are never placed in prompts, state or traces
in the first place; redaction is the second layer.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

REDACTED = "[REDACTED]"
PATH_PLACEHOLDER = "<path>"

_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{16,}"),  # OpenAI-style
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),  # AWS access key ID
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),  # Slack
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),  # Google API key
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),  # JWT
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{8,}=*"),
)
_SECRET_NAME = (
    r"(?:[A-Za-z0-9_]*(?:api[_-]?key|secret|token|passw(?:or)?d|pwd|credential|auth(?:orization)?)[A-Za-z0-9_]*)"
)
_ASSIGNMENT = re.compile(rf"(?i)\b({_SECRET_NAME})(['\"]?\s*[:=]\s*)(['\"]?)([^\s'\",;]{{4,}})\3")
_SECRET_ENV_NAME = re.compile(r"(?i)(key|secret|token|passw|pwd|credential|auth)")
_ABSOLUTE_PATH = re.compile(r"(?<![\w.])(?:/[\w.\-]+){2,}/?|\b[A-Za-z]:\\(?:[\w.\-]+\\?)+")
_MIN_SECRET_LENGTH = 8

_registered: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a secret value (e.g. a configured API key) so that it is redacted wherever it appears."""
    if value and len(value) >= _MIN_SECRET_LENGTH:
        _registered.add(value)


def _looks_like_secret_value(value: str) -> bool:
    """Opaque token-like values only: a path or an ordinary word is not treated as a secret."""
    return (
        len(value) >= 12
        and not any(ch.isspace() for ch in value)
        and not value.startswith(("/", "."))
        and any(ch.isdigit() for ch in value)
        and any(ch.isalpha() for ch in value)
    )


_env_cache: dict[int, tuple[str, ...]] = {}


def _environment_secrets() -> Iterable[str]:
    """Secret-like environment values, rescanned only when the environment changes."""
    items = tuple(os.environ.items())
    key = hash(items)
    cached = _env_cache.get(key)
    if cached is None:
        cached = tuple(
            value
            for name, value in items
            if _SECRET_ENV_NAME.search(name) and value and _looks_like_secret_value(value)
        )
        _env_cache.clear()
        _env_cache[key] = cached
    return cached


def contains_secret(text: str) -> bool:
    return redact(text) != text


def redact(text: str) -> str:
    """Replace secrets in ``text`` with ``[REDACTED]``."""
    if not text:
        return text
    result = text
    for literal in sorted({*_registered, *_environment_secrets()}, key=len, reverse=True):
        if literal in result:
            result = result.replace(literal, REDACTED)
    for pattern in _KEY_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", result)


def redact_paths(text: str) -> str:
    """Replace absolute filesystem paths (internal layout) with a placeholder."""
    return _ABSOLUTE_PATH.sub(PATH_PLACEHOLDER, text) if text else text


def redact_value(value: Any) -> Any:
    """Redact secrets in every string nested in ``value`` (dicts, lists, tuples)."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(v) for v in value)
    return value
