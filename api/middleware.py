"""
Security middleware for the AMRCAIS Dashboard API.

Provides:
- Security headers (OWASP best practices)
- Rate limiting per IP
- Request size limiting
- Global exception handler (prevents stack trace leakage)
"""

import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ─── Security Headers Middleware ──────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject OWASP-recommended security headers into every response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Cache-Control"] = "no-store"
        return response


# ─── Rate Limiting Middleware ─────────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per client IP.

    Args:
        max_requests: Maximum allowed requests per window.
        window_seconds: Length of the sliding window in seconds.
        burst_limit: Maximum allowed burst requests within a short period.
        burst_window: Short window for burst detection in seconds.
    """

    def __init__(
        self,
        app: FastAPI,
        max_requests: int = 120,
        window_seconds: int = 60,
        burst_limit: int = 30,
        burst_window: int = 5,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit
        self.burst_window = burst_window
        self._request_log: Dict[str, List[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, honouring X-Forwarded-For behind a proxy."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self, ip: str, now: float) -> None:
        """Remove timestamps outside the sliding window."""
        cutoff = now - self.window_seconds
        self._request_log[ip] = [
            t for t in self._request_log[ip] if t > cutoff
        ]
        # Evict IPs with no recent activity to prevent memory growth
        if not self._request_log[ip]:
            del self._request_log[ip]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health probes
        if request.url.path in ("/api/health", "/docs", "/openapi.json"):
            return await call_next(request)

        ip = self._get_client_ip(request)
        now = time.time()
        self._cleanup(ip, now)

        timestamps = self._request_log.get(ip, [])

        # Check burst limit (short window)
        burst_cutoff = now - self.burst_window
        recent_burst = sum(1 for t in timestamps if t > burst_cutoff)
        if recent_burst >= self.burst_limit:
            logger.warning(f"Burst rate limit exceeded for {ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "detail": "Too many requests in a short period. Please slow down."},
                headers={"Retry-After": str(self.burst_window)},
            )

        # Check sliding window limit
        if len(timestamps) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "detail": "Too many requests. Please wait before retrying."},
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._request_log[ip].append(now)
        response = await call_next(request)

        # Add rate limit headers for client awareness
        remaining = self.max_requests - len(self._request_log.get(ip, []))
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        response.headers["X-RateLimit-Reset"] = str(self.window_seconds)

        return response


# ─── Request Size Limiter ─────────────────────────────────────────


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with bodies larger than max_bytes.

    By default, allow up to 1MB — the backtest request body is tiny
    but this guards against abuse.
    """

    MAX_BYTES = 1_048_576  # 1 MB

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BYTES:
            logger.warning(
                f"Request body too large: {content_length} bytes from "
                f"{request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "detail": f"Maximum body size is {self.MAX_BYTES} bytes.",
                },
            )
        return await call_next(request)


# ─── Global Exception Handler ────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers that prevent stack trace leakage."""

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred. Please try again later.",
            },
        )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        logger.warning(f"ValueError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation error", "detail": str(exc)},
        )
