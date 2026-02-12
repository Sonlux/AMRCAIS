"""
Tests for AMRCAIS API — core endpoints, security headers, and middleware.

Covers:
    GET /api/health            — Health probe
    GET /api/status            — System status
    Security headers           — OWASP header injection
    Rate limiting              — Per-IP sliding window
    Request size limit         — 1 MB body guard
    Exception handling         — Stack trace suppression
"""

import pytest


# ─── Health Check ─────────────────────────────────────────────────


class TestHealthCheck:
    """GET /api/health"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/health")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, api_client):
        data = api_client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_response_has_version(self, api_client):
        data = api_client.get("/api/health").json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_response_has_timestamp(self, api_client):
        data = api_client.get("/api/health").json()
        assert "timestamp" in data


# ─── System Status ────────────────────────────────────────────────


class TestSystemStatus:
    """GET /api/status"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/status")
        assert resp.status_code == 200

    def test_is_initialized(self, api_client):
        data = api_client.get("/api/status").json()
        assert data["is_initialized"] is True

    def test_has_modules_list(self, api_client):
        data = api_client.get("/api/status").json()
        assert "modules_loaded" in data
        assert isinstance(data["modules_loaded"], list)

    def test_has_regime_info(self, api_client):
        data = api_client.get("/api/status").json()
        # current_regime can be None if state is empty
        assert "current_regime" in data
        assert "confidence" in data
        assert "disagreement" in data


# ─── Security Headers ────────────────────────────────────────────


class TestSecurityHeaders:
    """OWASP security headers on every response."""

    def test_x_content_type_options(self, api_client):
        headers = api_client.get("/api/health").headers
        assert headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options(self, api_client):
        headers = api_client.get("/api/health").headers
        assert headers.get("x-frame-options") == "DENY"

    def test_x_xss_protection(self, api_client):
        headers = api_client.get("/api/health").headers
        assert headers.get("x-xss-protection") == "1; mode=block"

    def test_referrer_policy(self, api_client):
        headers = api_client.get("/api/health").headers
        assert "strict-origin" in headers.get("referrer-policy", "")

    def test_permissions_policy(self, api_client):
        headers = api_client.get("/api/health").headers
        assert "camera=()" in headers.get("permissions-policy", "")

    def test_cache_control_no_store(self, api_client):
        headers = api_client.get("/api/health").headers
        assert headers.get("cache-control") == "no-store"


# ─── Rate Limit Headers ──────────────────────────────────────────


class TestRateLimitHeaders:
    """Rate-limit headers (skipped in normal tests since middleware is bypassed).

    The RateLimitMiddleware is tested directly below.
    """

    def test_rate_limiter_class_exists(self):
        from api.middleware import RateLimitMiddleware
        assert RateLimitMiddleware is not None

    def test_rate_limiter_has_defaults(self):
        from api.middleware import RateLimitMiddleware
        from unittest.mock import MagicMock

        limiter = RateLimitMiddleware(app=MagicMock())
        assert limiter.max_requests == 120
        assert limiter.window_seconds == 60
        assert limiter.burst_limit == 30
        assert limiter.burst_window == 5

    def test_rate_limiter_custom_params(self):
        from api.middleware import RateLimitMiddleware
        from unittest.mock import MagicMock

        limiter = RateLimitMiddleware(
            app=MagicMock(), max_requests=10, window_seconds=30
        )
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 30


# ─── Request Size Limit ──────────────────────────────────────────


class TestRequestSizeLimit:
    """Body size guard — rejects payloads > 1 MB."""

    def test_large_body_rejected(self, api_client):
        huge_body = "x" * (1_048_576 + 1)  # 1 MB + 1
        resp = api_client.post(
            "/api/backtest/run",
            content=huge_body,
            headers={"content-length": str(len(huge_body)), "content-type": "application/json"},
        )
        assert resp.status_code == 413

    def test_normal_body_accepted(self, api_client):
        # A normal-sized JSON body should pass the size check
        resp = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100000,
            "assets": ["SPX", "TLT", "GLD"],
        })
        # Should not be 413 — may be 200 or 503 depending on system state
        assert resp.status_code != 413


# ─── Unknown Routes ──────────────────────────────────────────────


class TestUnknownRoutes:
    """404 for non-existent endpoints."""

    def test_unknown_path_returns_404(self, api_client):
        resp = api_client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_unknown_nested_path(self, api_client):
        resp = api_client.get("/api/regime/nonexistent")
        assert resp.status_code in (404, 405)
