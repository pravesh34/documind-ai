"""Application error types and FastAPI exception handlers."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger("errors")


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Any = None,
        expose: bool = True,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.expose = expose


def error_body(code: str, message: str, request_id: str, details: Any = None) -> dict:
    body: dict[str, Any] = {
        "error": {"code": code, "message": message, "requestId": request_id},
        "requestId": request_id,
    }
    if details is not None:
        body["error"]["details"] = details
    return body


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = request_id_of(request)
    if exc.status_code >= 500:
        logger.error(
            "application error",
            code=exc.code,
            path=request.url.path,
            request_id=request_id,
            error=str(exc),
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.code, exc.message, request_id, exc.details),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "path": ".".join(str(part) for part in error.get("loc", [])),
            "message": error.get("msg", ""),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body(
            "VALIDATION_ERROR",
            "Request validation failed",
            request_id_of(request),
            details,
        ),
    )


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code, str(exc.detail), request_id_of(request)),
    )


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    headers: dict[str, str] = {}
    window = getattr(getattr(exc, "limit", None), "limit", None)
    expiry = window.get_expiry() if hasattr(window, "get_expiry") else None
    if expiry:
        headers["retry-after"] = str(max(1, int(expiry)))

    return JSONResponse(
        status_code=429,
        headers=headers,
        content=error_body(
            "RATE_LIMITED",
            "Too many requests. Please retry later.",
            request_id_of(request),
        ),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_of(request)
    logger.exception(
        "unhandled error",
        path=request.url.path,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=500,
        content=error_body(
            "INTERNAL_ERROR",
            "Internal server error",
            request_id,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(
        RequestValidationError, validation_error_handler  # type: ignore[arg-type]
    )
    app.add_exception_handler(StarletteHTTPException, http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)
