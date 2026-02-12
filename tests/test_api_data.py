"""
Tests for AMRCAIS API — data endpoints.

Covers:
    GET /api/data/assets                — Available asset list
    GET /api/data/prices/{asset}        — OHLCV price data
    GET /api/data/correlations          — Cross-asset correlation matrix
"""

import pytest

ALL_ASSETS = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"]


# ─── Asset List ───────────────────────────────────────────────────


class TestAssetList:
    """GET /api/data/assets"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/data/assets")
        assert resp.status_code == 200

    def test_returns_6_assets(self, api_client):
        assets = api_client.get("/api/data/assets").json()
        assert len(assets) == 6

    def test_expected_assets(self, api_client):
        assets = api_client.get("/api/data/assets").json()
        assert set(assets) == set(ALL_ASSETS)


# ─── Asset Prices ────────────────────────────────────────────────


class TestAssetPrices:
    """GET /api/data/prices/{asset}"""

    @pytest.mark.parametrize("asset", ALL_ASSETS)
    def test_valid_asset_returns_200(self, api_client, asset):
        resp = api_client.get(f"/api/data/prices/{asset}")
        assert resp.status_code == 200

    def test_response_has_asset_name(self, api_client):
        data = api_client.get("/api/data/prices/SPX").json()
        assert data["asset"] == "SPX"

    def test_response_has_prices_list(self, api_client):
        data = api_client.get("/api/data/prices/SPX").json()
        assert "prices" in data
        assert isinstance(data["prices"], list)

    def test_response_has_total_points(self, api_client):
        data = api_client.get("/api/data/prices/SPX").json()
        assert "total_points" in data
        assert data["total_points"] == len(data["prices"])

    def test_empty_when_no_market_data(self, api_client):
        """Mock system has market_data=None → should return empty prices."""
        data = api_client.get("/api/data/prices/SPX").json()
        assert data["total_points"] == 0
        assert data["prices"] == []

    def test_case_insensitive(self, api_client):
        resp = api_client.get("/api/data/prices/spx")
        assert resp.status_code == 200
        assert resp.json()["asset"] == "SPX"

    def test_invalid_asset_returns_404(self, api_client):
        resp = api_client.get("/api/data/prices/INVALID")
        assert resp.status_code == 404

    def test_invalid_asset_error_detail(self, api_client):
        data = api_client.get("/api/data/prices/NOPE").json()
        assert "detail" in data
        assert "Unknown asset" in data["detail"]


# ─── Correlation Matrix ──────────────────────────────────────────


class TestCorrelationMatrix:
    """GET /api/data/correlations"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/data/correlations")
        assert resp.status_code == 200

    def test_response_has_assets(self, api_client):
        data = api_client.get("/api/data/correlations").json()
        assert "assets" in data
        assert isinstance(data["assets"], list)

    def test_response_has_matrix(self, api_client):
        data = api_client.get("/api/data/correlations").json()
        assert "matrix" in data
        assert isinstance(data["matrix"], list)

    def test_default_window_60(self, api_client):
        data = api_client.get("/api/data/correlations").json()
        assert data["window"] == 60

    def test_custom_window(self, api_client):
        resp = api_client.get("/api/data/correlations", params={"window": 120})
        assert resp.status_code == 200
        assert resp.json()["window"] == 120

    def test_window_too_small_rejected(self, api_client):
        resp = api_client.get("/api/data/correlations", params={"window": 5})
        assert resp.status_code == 422

    def test_window_too_large_rejected(self, api_client):
        resp = api_client.get("/api/data/correlations", params={"window": 300})
        assert resp.status_code == 422

    def test_matrix_square(self, api_client):
        data = api_client.get("/api/data/correlations").json()
        n = len(data["assets"])
        matrix = data["matrix"]
        assert len(matrix) == n
        for row in matrix:
            assert len(row) == n
