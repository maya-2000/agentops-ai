"""API routes: ``ask`` (one question), ``investigations`` (Phase 10) and ``system`` (health, capabilities, metrics)."""

from app.api.routes.ask import router as ask_router
from app.api.routes.investigations import router as investigations_router
from app.api.routes.system import router as system_router

__all__ = ["ask_router", "investigations_router", "system_router"]
