"""Phase 8: the AgentOps HTTP API, a thin transport over the Phase 4/5 agent.

    client (Streamlit UI, curl) -> FastAPI routes (``routes``) -> ``AgentService`` (``service``:
        one serialised agent run per request, with a wall-clock timeout)
        -> ``AgentRunner`` (Phase 4 graph; Phase 5 secured tool execution, budgets and guardrails)
        -> ``presenter``: a typed ``AskResponse`` built from the run's existing domain objects
           (``AgentResponse``, ``Claim``, ``Evidence``, ``ValidatedRequest``) plus the
           ``visualizations`` chart specs, which only copy numbers from evidence and tool results.

The API never queries the database, never calls a tool and never calculates a business number
itself. It has no authentication: it is a local development service (bind it to localhost).

Modules:

- ``main``: ``create_app`` (the FastAPI application, its lifespan and error handlers).
- ``routes``: ``/api/v1/ask``, ``/api/v1/ask/stream``, ``/api/v1/health``,
  ``/api/v1/capabilities`` and ``/api/v1/metrics``.
- ``schemas``: request, response, error and visualization models.
- ``service``: the agent service (runner, lock, timeout, queue bound, health snapshot).
- ``presenter`` / ``visualizations``: the response and the chart specs from an agent run.
- ``errors``: the API error codes, HTTP statuses and fixed, client-safe messages.
- ``middleware`` / ``observability``: request IDs, body-size limit and structured request logs.
- ``config``: API settings from ``app.config``.

Architecture and contract: ``docs/api.md``.
"""

API_VERSION = "0.8.0"
API_PREFIX = "/api/v1"
SCHEMA_VERSION = "1.0"
