from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    status: str
    service: str
    environment: str
