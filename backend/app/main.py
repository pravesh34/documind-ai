"""FastAPI application factory and lifespan wiring."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import build_router
from app.config import Settings, get_settings
from app.container import Container
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import (
    BodyLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.rate_limit import create_limiter

logger = get_logger("main")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.environment)
    limiter = create_limiter(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = await Container.create(settings)
        app.state.container = container
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()
            app.state.container = None

    app = FastAPI(
        title="DocuMind AI",
        description=(
            "Production Graph RAG service powered by FastAPI, LangChain and LangGraph "
            "with free/local AI providers."
        ),
        version=settings.app_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.limiter = limiter
    app.state.started_at = time.time()

    register_exception_handlers(app)

    # Middleware runs in reverse order of addition (last added = outermost).
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        BodyLimitMiddleware,
        max_bytes=int(settings.max_upload_bytes * 1.6),
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts=settings.environment == "production",
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1_024)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["x-request-id", "content-disposition"],
            max_age=600,
        )

    if "*" not in settings.trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)

    app.include_router(build_router(settings, limiter), prefix=settings.api_v1_prefix)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {
            "status": "ok",
            "version": settings.app_version,
            "environment": settings.environment,
            "uptimeSeconds": round(time.time() - app.state.started_at, 1),
        }

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request):
        container: Container | None = getattr(request.app.state, "container", None)
        if container is None or not container.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "starting", "checks": {"container": False}},
            )

        healthy = await container.healthy()
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ready" if healthy else "degraded",
                "checks": {
                    "container": True,
                    "repository": healthy,
                },
                "backend": "postgres" if container.database else "memory",
            },
        )

    if settings.metrics_enabled:
        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            excluded_handlers=["/healthz", "/readyz", "/metrics"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    frontend = Path(settings.frontend_dist)
    if not frontend.is_absolute():
        frontend = (Path(__file__).resolve().parents[2] / frontend).resolve()
    if frontend.is_dir() and (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        async def root() -> dict:
            return {
                "name": "DocuMind AI",
                "version": settings.app_version,
                "docs": "/docs" if settings.enable_docs else None,
                "api": settings.api_v1_prefix,
            }

    logger.info(
        "application configured",
        environment=settings.environment,
        auth_mode=settings.resolved_auth_mode,
        backend="postgres" if settings.uses_postgres else "memory",
        docs=settings.enable_docs,
    )
    return app


app = create_app()
