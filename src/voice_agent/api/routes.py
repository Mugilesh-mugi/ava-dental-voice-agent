"""The web endpoints.

    GET /health                          is the app working? (open: shows nothing personal)
    GET /metrics                         live numbers for monitoring tools (Prometheus format)
    GET /api/mock-consultation/bookings  what the pretend clinic booking system has received (asks for the login
                                         when DASHBOARD_PASSWORD is set, because it shows callers' names and numbers)
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..config import Settings
from ..memory.database import Database
from ..services.mock_clinic import MockConsultationTool
from .schemas import BookingEvent, HealthResponse


def build_router(settings: Settings, db: Database, clinic: MockConsultationTool, model_names: dict[str, str]) -> APIRouter:
    router = APIRouter()
    basic = HTTPBasic(auto_error=False)

    def require_login(credentials: HTTPBasicCredentials | None = Depends(basic)) -> None:
        """Ask for the dashboard login (only when DASHBOARD_PASSWORD is set)."""
        password = settings.dashboard_password_value
        if not password:
            return
        ok = (credentials is not None
              and secrets.compare_digest(credentials.username.encode(), settings.dashboard_user.encode())
              and secrets.compare_digest(credentials.password.encode(), password.encode()))
        if not ok:
            raise HTTPException(status_code=401, detail="Login required", headers={"WWW-Authenticate": "Basic"})

    @router.get("/health", response_model=HealthResponse)
    def health():
        db_ok = db.ping()
        body = HealthResponse(status="ok" if db_ok else "degraded", version=__version__, database=db_ok,
                              models=model_names,
                              login_required=bool(settings.dashboard_password_value))
        return JSONResponse(body.model_dump(), status_code=200 if db_ok else 503)

    @router.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @router.get("/api/mock-consultation/bookings", response_model=list[BookingEvent],
                dependencies=[Depends(require_login)])
    def mock_bookings():
        return clinic.received()

    return router
