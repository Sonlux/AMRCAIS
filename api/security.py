"""
Security module for the AMRCAIS Dashboard API.

Provides:
- API key authentication for protected endpoints
- CSRF token generation and validation
- Secure cookie configuration
- Input sanitization helpers
- File upload validation

All security measures follow OWASP best practices.
"""

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)


# ─── API Key Authentication ──────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Load API key from environment; if not set the system operates in
# "open" mode (suitable for local dev) but logs a warning on startup.
_CONFIGURED_API_KEY: Optional[str] = os.getenv("AMRCAIS_API_KEY")

if not _CONFIGURED_API_KEY:
    logger.warning(
        "AMRCAIS_API_KEY not set — API runs without authentication. "
        "Set this env var in production."
    )


def _constant_time_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


async def require_api_key(
    api_key: Optional[str] = Depends(_API_KEY_HEADER),
) -> str:
    """Dependency that enforces API key authentication.

    In development (no key configured), all requests are allowed.
    In production, the X-API-Key header must match AMRCAIS_API_KEY.

    Returns:
        The validated API key string.

    Raises:
        HTTPException 401: Missing or invalid key.
    """
    # Open mode — no key configured
    if not _CONFIGURED_API_KEY:
        return "open-mode"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not _constant_time_compare(api_key, _CONFIGURED_API_KEY):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def optional_api_key(
    api_key: Optional[str] = Depends(_API_KEY_HEADER),
) -> Optional[str]:
    """Dependency that checks API key if provided but does NOT reject
    unauthenticated requests.  Useful for read-only GET endpoints that
    should work without a key in dev but log authenticated access."""
    if not _CONFIGURED_API_KEY or not api_key:
        return None
    if _constant_time_compare(api_key, _CONFIGURED_API_KEY):
        return api_key
    return None


# ─── CSRF Protection ────────────────────────────────────────────

# CSRF secret — shared server-side key for token generation
_CSRF_SECRET = os.getenv("AMRCAIS_CSRF_SECRET", secrets.token_hex(32))
_CSRF_TOKEN_LIFETIME = timedelta(hours=8)


def generate_csrf_token() -> str:
    """Generate a time-limited CSRF token.

    The token encodes a timestamp + random nonce signed with the server
    secret so it cannot be forged.

    Returns:
        Hex-encoded CSRF token.
    """
    timestamp = int(datetime.now(timezone.utc).timestamp())
    nonce = secrets.token_hex(16)
    payload = f"{timestamp}:{nonce}"
    signature = hmac.new(
        _CSRF_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{signature}"


def validate_csrf_token(token: str) -> bool:
    """Validate a CSRF token's signature and freshness.

    Args:
        token: The token string from the X-CSRF-Token header.

    Returns:
        True if valid and not expired.
    """
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False

        timestamp_str, nonce, signature = parts
        payload = f"{timestamp_str}:{nonce}"
        expected = hmac.new(
            _CSRF_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return False

        # Check expiration
        token_time = datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
        if datetime.now(timezone.utc) - token_time > _CSRF_TOKEN_LIFETIME:
            return False

        return True
    except (ValueError, TypeError):
        return False


async def verify_csrf(request: Request) -> None:
    """FastAPI dependency: reject state-changing requests without a valid
    CSRF token.  Only enforced on POST / PUT / PATCH / DELETE.

    The token is read from the ``X-CSRF-Token`` header.

    Raises:
        HTTPException 403: Missing or invalid CSRF token.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    token = request.headers.get("X-CSRF-Token")
    if not token or not validate_csrf_token(token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing or invalid CSRF token",
        )


# ─── Secure Cookie Helpers ───────────────────────────────────────

COOKIE_DEFAULTS = {
    "httponly": True,
    "secure": os.getenv("AMRCAIS_ENV", "development") == "production",
    "samesite": "strict",
    "max_age": 3600,  # 1 hour
}


# ─── Input Sanitization ─────────────────────────────────────────

_SAFE_STRING = re.compile(r"^[\w\s\-.,;:!?'\"()\[\]{}/@#$%^&*+=<>|\\~`]+$")
_PATH_TRAVERSAL = re.compile(r"\.\.[/\\]")


def sanitize_string(value: str, max_length: int = 500) -> str:
    """Sanitize a generic string input.

    Strips leading/trailing whitespace, limits length, and removes
    null bytes.

    Args:
        value: Raw input.
        max_length: Maximum allowed length.

    Returns:
        Sanitized string.
    """
    if not isinstance(value, str):
        return ""
    value = value.strip().replace("\x00", "")
    return value[:max_length]


def is_safe_path(path: str) -> bool:
    """Check a path for directory-traversal patterns.

    Returns:
        True if the path does NOT contain traversal sequences.
    """
    return not bool(_PATH_TRAVERSAL.search(path))


def sanitize_query_param(value: str, allowed_pattern: re.Pattern | None = None) -> str:
    """Sanitize a URL query parameter.

    Args:
        value: Raw query value.
        allowed_pattern: Optional regex the value must match.

    Returns:
        Sanitized value.

    Raises:
        ValueError: If the value fails the allowed_pattern check.
    """
    value = sanitize_string(value, max_length=200)
    if allowed_pattern and not allowed_pattern.fullmatch(value):
        raise ValueError(f"Invalid query parameter value: {value!r}")
    return value


# ─── File Upload Validation ──────────────────────────────────────

ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".json", ".yaml", ".yml"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def validate_upload(
    filename: str,
    content_length: int,
    content_type: str | None = None,
) -> None:
    """Validate an uploaded file before processing.

    Args:
        filename: Original filename.
        content_length: Size in bytes.
        content_type: MIME type (optional).

    Raises:
        HTTPException: If validation fails.
    """
    # Check extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' not allowed. "
                   f"Accepted: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    # Check size
    if content_length > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({content_length} bytes). "
                   f"Max: {MAX_UPLOAD_SIZE_BYTES} bytes.",
        )

    # Check for path traversal in filename
    if not is_safe_path(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename — path traversal detected.",
        )

    # Block double extensions (e.g., shell.csv.exe)
    name_parts = filename.rsplit(".", maxsplit=2)
    if len(name_parts) > 2:
        second_ext = f".{name_parts[-2].lower()}"
        dangerous = {".exe", ".bat", ".cmd", ".sh", ".ps1", ".py", ".js"}
        if second_ext in dangerous:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Suspicious double extension detected.",
            )
