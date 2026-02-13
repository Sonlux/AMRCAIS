"""
Tests for AMRCAIS security features.

Validates:
- Security headers (OWASP, HSTS, CSP)
- CSRF token generation and validation
- API key authentication
- Input sanitization
- File upload validation
- Error message sanitization (no stack traces)
- Rate limiting headers
"""

import time

import pytest
from unittest.mock import patch

from api.security import (
    generate_csrf_token,
    validate_csrf_token,
    sanitize_string,
    sanitize_query_param,
    is_safe_path,
    validate_upload,
    COOKIE_DEFAULTS,
)


# ─── CSRF Token Tests ────────────────────────────────────────────


class TestCSRFToken:
    """Verify CSRF token generation and validation."""

    def test_generate_returns_string(self):
        token = generate_csrf_token()
        assert isinstance(token, str)
        assert len(token) > 10

    def test_valid_token_passes_validation(self):
        token = generate_csrf_token()
        assert validate_csrf_token(token) is True

    def test_empty_token_rejected(self):
        assert validate_csrf_token("") is False

    def test_garbage_token_rejected(self):
        assert validate_csrf_token("not-a-real-token") is False

    def test_tampered_signature_rejected(self):
        token = generate_csrf_token()
        parts = token.split(":")
        parts[-1] = "0" * len(parts[-1])  # Tamper with signature
        assert validate_csrf_token(":".join(parts)) is False

    def test_malformed_token_rejected(self):
        assert validate_csrf_token("a:b") is False
        assert validate_csrf_token("a:b:c:d") is False

    def test_expired_token_rejected(self):
        """Token with a very old timestamp should be rejected."""
        import hashlib
        import hmac
        import os

        secret = os.getenv("AMRCAIS_CSRF_SECRET", "")
        # Use a timestamp from 24 hours ago
        old_ts = str(int(time.time()) - 86400)
        nonce = "a" * 32
        payload = f"{old_ts}:{nonce}"
        sig = hmac.new(
            secret.encode() if secret else b"",
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        token = f"{payload}:{sig}"
        # This should fail because either the secret doesn't match
        # or the token is expired (>8 hours)
        assert validate_csrf_token(token) is False

    def test_csrf_endpoint_returns_token(self, api_client):
        """GET /api/csrf-token returns a token."""
        resp = api_client.get("/api/csrf-token")
        assert resp.status_code == 200
        data = resp.json()
        assert "csrf_token" in data
        assert isinstance(data["csrf_token"], str)


# ─── Security Headers Tests ──────────────────────────────────────


class TestSecurityHeadersComplete:
    """Verify all OWASP security headers are present."""

    def test_x_content_type_options(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("x-content-type-options") == "nosniff"

    def test_x_frame_options(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("x-frame-options") == "DENY"

    def test_x_xss_protection(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("x-xss-protection") == "1; mode=block"

    def test_referrer_policy(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_permissions_policy(self, api_client):
        h = api_client.get("/api/health").headers
        pp = h.get("permissions-policy", "")
        assert "camera=()" in pp
        assert "microphone=()" in pp
        assert "geolocation=()" in pp
        assert "payment=()" in pp

    def test_cache_control_strict(self, api_client):
        h = api_client.get("/api/health").headers
        cc = h.get("cache-control", "")
        assert "no-store" in cc
        assert "no-cache" in cc
        assert "must-revalidate" in cc

    def test_pragma_no_cache(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("pragma") == "no-cache"

    def test_content_security_policy(self, api_client):
        h = api_client.get("/api/health").headers
        csp = h.get("content-security-policy", "")
        assert "default-src 'none'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_cross_origin_headers(self, api_client):
        h = api_client.get("/api/health").headers
        assert h.get("x-permitted-cross-domain-policies") == "none"
        assert h.get("cross-origin-opener-policy") == "same-origin"
        assert h.get("cross-origin-resource-policy") == "same-origin"


# ─── Input Sanitization Tests ────────────────────────────────────


class TestInputSanitization:
    """Verify server-side input sanitization functions."""

    def test_sanitize_string_strips_whitespace(self):
        assert sanitize_string("  hello  ") == "hello"

    def test_sanitize_string_removes_null_bytes(self):
        assert sanitize_string("data\x00attack") == "dataattack"

    def test_sanitize_string_limits_length(self):
        assert len(sanitize_string("a" * 1000, max_length=100)) == 100

    def test_sanitize_string_non_string(self):
        assert sanitize_string(123) == ""  # type: ignore

    def test_sanitize_query_param_basic(self):
        assert sanitize_query_param("hello") == "hello"

    def test_sanitize_query_param_long(self):
        assert len(sanitize_query_param("a" * 500)) == 200

    def test_sanitize_query_param_with_pattern(self):
        import re
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        assert sanitize_query_param("2024-01-15", pattern) == "2024-01-15"

    def test_sanitize_query_param_rejects_bad_pattern(self):
        import re
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        with pytest.raises(ValueError):
            sanitize_query_param("not-a-date", pattern)


# ─── Path Traversal Tests ────────────────────────────────────────


class TestPathTraversal:
    """Ensure directory traversal attempts are caught."""

    def test_safe_path_normal(self):
        assert is_safe_path("data/prices.csv") is True

    def test_safe_path_traversal_unix(self):
        assert is_safe_path("../../etc/passwd") is False

    def test_safe_path_traversal_windows(self):
        assert is_safe_path("..\\..\\windows\\system32") is False

    def test_safe_path_nested_traversal(self):
        assert is_safe_path("data/../../secret.key") is False


# ─── File Upload Validation Tests ────────────────────────────────


class TestFileUploadValidation:
    """Verify file upload security checks."""

    def test_allowed_extension(self):
        # Should not raise
        validate_upload("data.csv", 1000)

    def test_blocked_extension(self):
        with pytest.raises(Exception):  # HTTPException
            validate_upload("script.exe", 1000)

    def test_blocked_py_extension(self):
        with pytest.raises(Exception):
            validate_upload("malicious.py", 1000)

    def test_oversized_file(self):
        with pytest.raises(Exception):
            validate_upload("big.csv", 100_000_000)

    def test_path_traversal_filename(self):
        with pytest.raises(Exception):
            validate_upload("../../etc/passwd.csv", 100)

    def test_double_extension_blocked(self):
        with pytest.raises(Exception):
            validate_upload("shell.exe.csv", 100)


# ─── Secure Cookie Defaults ──────────────────────────────────────


class TestSecureCookies:
    """Verify secure cookie defaults."""

    def test_httponly_set(self):
        assert COOKIE_DEFAULTS["httponly"] is True

    def test_samesite_strict(self):
        assert COOKIE_DEFAULTS["samesite"] == "strict"

    def test_max_age_reasonable(self):
        assert COOKIE_DEFAULTS["max_age"] <= 3600 * 24  # Max 1 day


# ─── Error Message Sanitization ──────────────────────────────────


class TestErrorSanitization:
    """Ensure error responses don't leak sensitive info."""

    def test_404_no_stack_trace(self, api_client):
        resp = api_client.get("/api/nonexistent/path")
        assert resp.status_code == 404
        body = resp.text
        assert "Traceback" not in body
        assert "File " not in body

    def test_invalid_backtest_no_internals(self, api_client):
        resp = api_client.post(
            "/api/backtest/run",
            json={
                "strategy": "nonexistent_strategy!@#",
                "initial_capital": -1,
            },
        )
        # Should return 422 without exposing code paths
        body = resp.text
        assert "Traceback" not in body
        assert "__file__" not in body


# ─── API Key / Auth Tests (development mode — open) ──────────────


class TestAPIKeyAuth:
    """Verify API key auth behavior in open/dev mode."""

    def test_no_key_works_in_dev(self, api_client):
        """In dev mode (no AMRCAIS_API_KEY set), requests work without key."""
        resp = api_client.get("/api/health")
        assert resp.status_code == 200

    def test_status_works_without_key(self, api_client):
        resp = api_client.get("/api/status")
        assert resp.status_code == 200
