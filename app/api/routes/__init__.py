"""API routes: ``ask`` (the agent) and ``system`` (health, capabilities, metrics)."""

from app.api.routes.ask import router as ask_router
from app.api.routes.system import router as system_router

__all__ = ["ask_router", "system_router"]
