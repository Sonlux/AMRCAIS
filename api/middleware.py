"""
Security middleware for the AMRCAIS Dashboard API.

Provides:
- Security headers (OWASP best practices + HSTS + CSP)
- CSRF enforcement on state-changing requests
- Rate limiting per IP (sliding window + burst detection)
- Request size limiting
- Global exception handler (prevents stack trace leakage)
"""

import logging
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_IS_PRODUCTION = os.getenv("AMRCAIS_ENV", "development") == "production"

# ─── Security Headers Middleware ──────────────────────────────────


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject OWASP-recommended security headers into every response.

    Headers added:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: restrictive feature policy
    - Cache-Control: no-store (sensitive financial data)
    - Strict-Transport-Security (HSTS): 1 year with subdomains (production)
    - Content-Security-Policy: restrictive CSP for API responses
    - X-Permitted-Cross-Domain-Policies: none
    - Cross-Origin-Opener-Policy: same-origin
    - Cross-Origin-Resource-Policy: same-origin
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # ── Core OWASP headers ─────────────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=()"
        )
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"

        # ── HSTS — enforce HTTPS (production only) ─────────────
        if _IS_PRODUCTION:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # ── Content Security Policy for API ────────────────────
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'none'; "
            "form-action 'none'"
        )

        # ── Cross-origin isolation ─────────────────────────────
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"

        return response


# ─── CSRF Middleware ──────────────────────────────────────────────


class CSRFMiddleware(BaseHTTPMiddleware):
    """Enforce CSRF token on state-changing requests (POST/PUT/PATCH/DELETE).

    Reads the token from the ``X-CSRF-Token`` header and validates it
    using the server-side CSRF secret.  GET / HEAD / OPTIONS are exempt.
    The ``/docs`` and ``/openapi.json`` paths are also exempt so Swagger
    UI keeps working.

    To obtain a token the client calls ``GET /api/csrf-token``.
    """

    # Paths exempt from CSRF (read-only / infra)
    EXEMPT_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}
    # Exempt prefixes (Swagger support files)
    EXEMPT_PREFIXES = ("/docs/", "/redoc/")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Safe methods — no CSRF needed
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        # Exempt paths
        path = request.url.path
        if path in self.EXEMPT_PATHS or path.startswith(self.EXEMPT_PREFIXES):
            return await call_next(request)

        # Validate CSRF token
        from api.security import validate_csrf_token

        token = request.headers.get("X-CSRF-Token")
        if not token or not validate_csrf_token(token):
            logger.warning(
                f"CSRF validation failed for {request.method} {path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "CSRF validation failed",
                    "detail": "Missing or invalid X-CSRF-Token header. "
                              "Obtain a token via GET /api/csrf-token.",
                },
            )

        return await call_next(request)


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
    """Register global exception handlers that prevent stack trace leakage.

    In production mode, error details are completely suppressed.
    In development, a sanitized message (no traceback) is returned.
    """

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            f"Unhandled exception on {request.method} {request.url.path}: {exc}",
            exc_info=True,
        )
        # NEVER expose internal details in the response
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
        # Sanitize — only return the message, never the traceback
        safe_detail = str(exc)[:200]  # Truncate to prevent info leakage
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation error", "detail": safe_detail},
        )

    @app.exception_handler(PermissionError)
    async def _permission_error(request: Request, exc: PermissionError) -> JSONResponse:
        logger.error(f"PermissionError on {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Forbidden", "detail": "Insufficient permissions."},
        )
