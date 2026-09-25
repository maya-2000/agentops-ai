"""Phase 5 security layer: explicit, deterministic, testable and auditable controls for the agent.

The model can propose; the application decides. Modules, in the order a request meets them:

- ``limits``: ``SecurityLimits``, every limit, immutable and read from settings.
- ``input_guard`` / ``injection``: question and understanding validation; prompt-injection screen.
- ``validators``: the central value validators used by every other boundary.
- ``plan_validator`` / ``authorization``: untrusted plans; ``ToolAuthorizationPolicy`` before
  every tool call.
- ``data_policy``: approved tables, views and columns; withheld and PII fields.
- ``budget`` / ``retry`` / ``timeouts`` / ``context``: resource limits for one run.
- ``output_guard``: tool-output validation; safe response shortening.
- ``redaction`` / ``errors`` / ``events``: secret redaction, safe errors, security audit events.

Only the dependency-free core is re-exported here. ``authorization``, ``plan_validator`` and
``output_guard`` depend on the tool layer and are imported from their modules, which avoids an
import cycle with ``app.tools``.

Threat model: ``docs/security-threat-model.md``. Architecture: ``docs/security-architecture.md``.
"""

from app.security.events import SecurityEvent, Severity, security_event
from app.security.limits import SecurityLimits
from app.security.redaction import redact, register_secret

__all__ = ["SecurityEvent", "SecurityLimits", "Severity", "redact", "register_secret", "security_event"]
