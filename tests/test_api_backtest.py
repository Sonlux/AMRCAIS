"""
Tests for AMRCAIS API — backtest endpoints and input validation.

Covers:
    POST /api/backtest/run              — Execute a new backtest
    GET  /api/backtest/results          — List saved backtest results
    GET  /api/backtest/results/{id}     — Get single backtest result
    Input validation                    — Strategy, assets, dates, capital
"""

import pytest


# ─── Run Backtest ────────────────────────────────────────────────


class TestRunBacktest:
    """POST /api/backtest/run"""

    def test_returns_200(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        })
        assert resp.status_code == 200

    def test_response_has_id(self, api_client):
        data = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        }).json()
        assert "id" in data
        assert data["id"].startswith("bt_")

    def test_response_has_equity_curve(self, api_client):
        data = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        }).json()
        assert "equity_curve" in data
        assert isinstance(data["equity_curve"], list)
        assert len(data["equity_curve"]) > 0

    def test_response_has_metrics(self, api_client):
        data = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        }).json()
        for key in ["total_return", "sharpe_ratio", "max_drawdown", "benchmark_return"]:
            assert key in data

    def test_response_has_regime_returns(self, api_client):
        data = api_client.post("/api/backtest/run", json={
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        }).json()
        assert "regime_returns" in data
        assert len(data["regime_returns"]) == 4

    def test_default_values_work(self, api_client):
        """BacktestRequest has defaults — an empty body should work."""
        resp = api_client.post("/api/backtest/run", json={})
        assert resp.status_code == 200


# ─── Input Validation ────────────────────────────────────────────


class TestBacktestValidation:
    """Input validation on POST /api/backtest/run"""

    def test_invalid_strategy_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "strategy": "invalid_strat",
        })
        assert resp.status_code == 422
        assert "Invalid strategy" in resp.json()["detail"]

    def test_invalid_asset_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "assets": ["SPX", "FAKE_ASSET"],
        })
        assert resp.status_code == 422
        assert "Invalid assets" in resp.json()["detail"]

    def test_negative_capital_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "initial_capital": -1000,
        })
        assert resp.status_code == 422

    def test_zero_capital_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "initial_capital": 0,
        })
        assert resp.status_code == 422

    def test_excessive_capital_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "initial_capital": 2_000_000_000,
        })
        assert resp.status_code == 422
        assert "initial_capital" in resp.json()["detail"]

    def test_invalid_date_format_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "start_date": "01/01/2020",
            "end_date": "2024-12-31",
        })
        assert resp.status_code == 422

    def test_start_after_end_422(self, api_client):
        resp = api_client.post("/api/backtest/run", json={
            "start_date": "2025-01-01",
            "end_date": "2020-01-01",
        })
        assert resp.status_code == 422
        assert "before" in resp.json()["detail"]

    def test_valid_strategies_accepted(self, api_client):
        for strategy in ["regime_following", "momentum", "mean_reversion"]:
            resp = api_client.post("/api/backtest/run", json={
                "strategy": strategy,
            })
            assert resp.status_code == 200, f"Strategy '{strategy}' should be valid"


# ─── List / Get Results ──────────────────────────────────────────


class TestBacktestResults:
    """GET /api/backtest/results and GET /api/backtest/results/{id}"""

    def test_list_results_200(self, api_client):
        resp = api_client.get("/api/backtest/results")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_result_not_found_404(self, api_client):
        resp = api_client.get("/api/backtest/results/bt_nonexistent")
        assert resp.status_code == 404

    def test_run_then_get_result(self, api_client):
        """Run a backtest and then fetch it by ID."""
        run_resp = api_client.post("/api/backtest/run", json={
            "start_date": "2021-01-01",
            "end_date": "2023-12-31",
            "strategy": "regime_following",
            "initial_capital": 50_000,
            "assets": ["SPX", "TLT", "GLD"],
        })
        assert run_resp.status_code == 200
        bt_id = run_resp.json()["id"]

        # Fetch by ID
        get_resp = api_client.get(f"/api/backtest/results/{bt_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == bt_id

    def test_run_appears_in_list(self, api_client):
        """Results from POST /run should show up in GET /results."""
        # Run a backtest
        api_client.post("/api/backtest/run", json={
            "start_date": "2022-01-01",
            "end_date": "2024-01-01",
            "strategy": "regime_following",
            "initial_capital": 100_000,
            "assets": ["SPX", "TLT", "GLD"],
        })

        results = api_client.get("/api/backtest/results").json()
        assert len(results) >= 1
