"""
Tests for AMRCAIS API — analytical module endpoints.

Covers:
    GET /api/modules/summary            — All 5 module signals
    GET /api/modules/{name}/analyze     — Full analysis for one module
    GET /api/modules/{name}/history     — Signal history for a module
    GET /api/modules/yield_curve/curve  — Yield curve snapshot
    GET /api/modules/options/surface    — Implied volatility surface grid
"""

import pytest

VALID_MODULES = ["macro", "yield_curve", "options", "factors", "correlations",
                 "contagion", "surprise_decay"]
VALID_SIGNALS = {"bullish", "bearish", "neutral", "cautious"}


# ─── Module Summary ──────────────────────────────────────────────


class TestModuleSummary:
    """GET /api/modules/summary"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/modules/summary")
        assert resp.status_code == 200

    def test_has_7_signals(self, api_client):
        data = api_client.get("/api/modules/summary").json()
        assert len(data["signals"]) == 7

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


# ─── Yield Curve Data ─────────────────────────────────────────────


class TestYieldCurveData:
    """GET /api/modules/yield_curve/curve"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/modules/yield_curve/curve")
        assert resp.status_code == 200

    def test_has_tenors_and_yields(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert "tenors" in data
        assert "yields" in data
        assert len(data["tenors"]) == len(data["yields"])
        assert len(data["tenors"]) >= 2

    def test_tenors_are_sorted(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert data["tenors"] == sorted(data["tenors"])

    def test_has_curve_shape(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert "curve_shape" in data
        assert data["curve_shape"] in {"normal", "flat", "inverted", "humped", "twisted"}

    def test_has_slope_and_curvature(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert "slope_2_10" in data
        assert "curvature" in data

    def test_has_regime_info(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert "regime" in data
        assert 1 <= data["regime"] <= 4
        assert "regime_name" in data

    def test_has_timestamp(self, api_client):
        data = api_client.get("/api/modules/yield_curve/curve").json()
        assert "timestamp" in data


# ─── Vol Surface ──────────────────────────────────────────────────


class TestVolSurface:
    """GET /api/modules/options/surface"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/modules/options/surface")
        assert resp.status_code == 200

    def test_has_surface_grid(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert "moneyness" in data
        assert "expiry_days" in data
        assert "iv_grid" in data

    def test_grid_dimensions_match(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        n_expiry = len(data["expiry_days"])
        n_moneyness = len(data["moneyness"])
        assert len(data["iv_grid"]) == n_expiry
        for row in data["iv_grid"]:
            assert len(row) == n_moneyness

    def test_moneyness_is_sorted(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert data["moneyness"] == sorted(data["moneyness"])

    def test_expiry_days_are_sorted(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert data["expiry_days"] == sorted(data["expiry_days"])

    def test_iv_values_are_positive(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        for row in data["iv_grid"]:
            for iv in row:
                assert iv > 0

    def test_has_atm_vol(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert "atm_vol" in data
        assert data["atm_vol"] > 0

    def test_has_regime_info(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert "regime" in data
        assert 1 <= data["regime"] <= 4
        assert "regime_name" in data

    def test_moneyness_includes_atm(self, api_client):
        data = api_client.get("/api/modules/options/surface").json()
        assert 1.0 in data["moneyness"]
