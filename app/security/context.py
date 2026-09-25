"""Context-window control: what the model is shown is bounded and prioritised.

Tool outputs never go to the model raw. The model sees compact evidence summaries and claim
texts, capped at ``max_context_items`` items per list, and the rendered prompt is capped at
``max_context_chars``. When a prompt is too large, the lowest-priority items are dropped first.
If it still does not fit, the call is refused (fail closed) rather than sending an oversized
prompt.

Priority, highest first:

1. evidence cited by primary (answer) claims;
2. validated claims (primary first, then observed/calculated, inferences, recommendations);
3. other evidence (key tool results), in creation order;
4. limitations (attached deterministically as caveats, so they never depend on the model).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from app.evidence.models import EvidenceGraph

_CLAIM_ORDER = {"observed_fact": 1, "calculated_result": 1, "inference": 2, "recommendation": 3}


class ContextTooLargeError(Exception):
    """The prompt cannot be reduced below the context limit without removing required content."""


class ContextUsage(BaseModel):
    items: int
    chars: int
    dropped: int
    prompt: str = Field(default="", exclude=True, repr=False)  # the rendered prompt that fits


def prioritised_evidence_ids(graph: EvidenceGraph) -> list[str]:
    """Evidence cited by primary claims, then by other claims, then the rest (creation order)."""
    primary = [e for c in graph.claims.values() if c.primary for e in c.evidence_ids]
    cited = [e for c in graph.claims.values() for e in c.evidence_ids]
    ordered = list(dict.fromkeys([*primary, *cited, *graph.evidence]))
    return [e for e in ordered if e in graph.evidence]


def prioritised_claims(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda c: (not c.get("primary"), _CLAIM_ORDER.get(str(c.get("type")), 4)))


def fit_context(
    context: dict[str, Any],
    render: Callable[[dict[str, Any]], str],
    *,
    trimmable: tuple[str, ...],
    max_items: int,
    max_chars: int,
) -> tuple[dict[str, Any], ContextUsage]:
    """Cap each trimmable list at ``max_items`` and drop trailing items until the prompt fits.

    ``trimmable`` lists keys from lowest to highest priority. Lists are assumed to be ordered
    by importance already, so their tails are dropped first.
    """
    fitted = dict(context)
    dropped = 0
    for key in trimmable:
        value = fitted.get(key)
        if isinstance(value, list) and len(value) > max_items:
            dropped += len(value) - max_items
            fitted[key] = value[:max_items]
    prompt = render(fitted)
    for key in trimmable:
        while len(prompt) > max_chars and isinstance(fitted.get(key), list) and fitted[key]:
            fitted[key] = fitted[key][:-1]
            dropped += 1
            prompt = render(fitted)
    if len(prompt) > max_chars:
        raise ContextTooLargeError(f"Prompt is {len(prompt)} characters (limit {max_chars})")
    items = sum(len(fitted[k]) for k in trimmable if isinstance(fitted.get(k), list))
    return fitted, ContextUsage(items=items, chars=len(prompt), dropped=dropped, prompt=prompt)
