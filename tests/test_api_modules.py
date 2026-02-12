"""
Tests for AMRCAIS API — analytical module endpoints.

Covers:
    GET /api/modules/summary            — All 5 module signals
    GET /api/modules/{name}/analyze     — Full analysis for one module
    GET /api/modules/{name}/history     — Signal history for a module
"""

import pytest

VALID_MODULES = ["macro", "yield_curve", "options", "factors", "correlations"]
VALID_SIGNALS = {"bullish", "bearish", "neutral", "cautious"}


# ─── Module Summary ──────────────────────────────────────────────


class TestModuleSummary:
    """GET /api/modules/summary"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/modules/summary")
        assert resp.status_code == 200

    def test_has_5_signals(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        assert len(data["signals"]) == 5

    def test_signal_entry_schema(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        for sig in data["signals"]:
            assert "module" in sig
            assert "signal" in sig
            assert sig["signal"] in VALID_SIGNALS
            assert "strength" in sig
            assert 0 <= sig["strength"] <= 1
            assert "confidence" in sig

    def test_has_regime_info(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        assert "current_regime" in data
        assert 1 <= data["current_regime"] <= 4
        assert "regime_name" in data

    def test_has_timestamp(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        assert "timestamp" in data

    def test_all_modules_present(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        module_names = {s["module"] for s in data["signals"]}
        assert module_names == set(VALID_MODULES)


# ─── Module Analysis ─────────────────────────────────────────────


class TestModuleAnalysis:
    """GET /api/modules/{name}/analyze"""

    @pytest.mark.parametrize("module_name", VALID_MODULES)
    def test_valid_module_returns_200(self, api_client, module_name):
        resp = api_client.get(f"/api/modules/{module_name}/analyze")
        assert resp.status_code == 200

    def test_response_has_signal(self, api_client):
        data = api_client.get("/api/modules/macro/analyze").json()
        assert "signal" in data
        assert data["signal"]["signal"] in VALID_SIGNALS

    def test_response_has_raw_metrics(self, api_client):
        data = api_client.get("/api/modules/macro/analyze").json()
        assert "raw_metrics" in data
        assert isinstance(data["raw_metrics"], dict)

    def test_response_has_regime_parameters(self, api_client):
        data = api_client.get("/api/modules/macro/analyze").json()
        assert "regime_parameters" in data
        assert isinstance(data["regime_parameters"], dict)

    def test_invalid_module_returns_404(self, api_client):
        resp = api_client.get("/api/modules/invalid_module/analyze")
        assert resp.status_code == 404

    def test_invalid_module_error_detail(self, api_client):
        data = api_client.get("/api/modules/nonexistent/analyze").json()
        assert "detail" in data
        assert "Unknown module" in data["detail"]


# ─── Module History ───────────────────────────────────────────────


class TestModuleHistory:
    """GET /api/modules/{name}/history"""

    @pytest.mark.parametrize("module_name", VALID_MODULES)
    def test_valid_module_returns_200(self, api_client, module_name):
        resp = api_client.get(f"/api/modules/{module_name}/history")
        assert resp.status_code == 200

    def test_response_has_module_name(self, api_client):
        data = api_client.get("/api/modules/macro/history").json()
        assert data["module"] == "macro"

    def test_response_has_history_list(self, api_client):
        data = api_client.get("/api/modules/macro/history").json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_invalid_module_returns_404(self, api_client):
        resp = api_client.get("/api/modules/bogus/history")
        assert resp.status_code == 404
