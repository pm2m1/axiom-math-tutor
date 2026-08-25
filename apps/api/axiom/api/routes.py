from fastapi import APIRouter

from axiom.core.config import get_settings
from axiom.models.status import HealthResponse, StatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/v1/status", response_model=StatusResponse)
def status() -> StatusResponse:
    settings = get_settings()
    return StatusResponse(status="ok", service="axiom-api", environment=settings.environment)
