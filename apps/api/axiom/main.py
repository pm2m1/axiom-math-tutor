from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from axiom.api.routes import router

# The Vite dev server runs on a different origin than the API during local
# development. This is loose on purpose for now; revisit before any real
# deployment.
DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app() -> FastAPI:
    app = FastAPI(title="Axiom API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
