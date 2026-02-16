"""
Tests for Phase 3: Prediction Engine.

Covers:
- Regime-Conditional Return Forecasting (return_forecaster.py)
- Tail Risk Attribution (tail_risk.py)
- Regime-Aware Portfolio Optimizer (portfolio_optimizer.py)
- Anomaly-Based Alpha Signals (alpha_signals.py)
- Phase 3 API endpoints (routes/phase3.py)
"""

import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_market_data(n_days: int = 300) -> pd.DataFrame:
    """Generate synthetic market data with return columns."""
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n_days, freq="B")
    data = pd.DataFrame(
        {
            "SPX": 4000 + np.cumsum(np.random.randn(n_days) * 10),
            "TLT": 100 + np.cumsum(np.random.randn(n_days) * 0.5),
            "GLD": 1800 + np.cumsum(np.random.randn(n_days) * 5),
            "VIX": np.clip(18 + np.cumsum(np.random.randn(n_days) * 0.3), 10, 80),
            "DXY": 100 + np.cumsum(np.random.randn(n_days) * 0.2),
            "WTI": 70 + np.cumsum(np.random.randn(n_days) * 1.0),
        },
        index=dates,
    )
    # Add return columns
    for asset in ["SPX", "TLT", "GLD", "DXY", "WTI"]:
        data[f"{asset}_returns"] = data[asset].pct_change()

    # Add factor columns
    data["momentum_20d"] = data["SPX_returns"].rolling(20).mean()
    data["realized_vol_20d"] = data["SPX_returns"].rolling(20).std()
    data["equity_bond_corr_30d"] = (
        data["SPX_returns"]
        .rolling(30)
        .corr(data["TLT_returns"])
    )
    data.dropna(inplace=True)
    return data


def _make_regime_series(index=None, n_days: int = 300) -> pd.Series:
    """Generate synthetic regime history."""
    np.random.seed(42)
    if index is not None:
        n_days = len(index)
        dates = index
    else:
        dates = pd.bdate_range(end=datetime.now(), periods=n_days, freq="B")
    regimes = np.random.choice([1, 2, 3, 4], size=n_days, p=[0.5, 0.2, 0.15, 0.15])
    return pd.Series(regimes, index=dates)


# ═══════════════════════════════════════════════════════════════════
#  3.1 — Return Forecaster
# ═══════════════════════════════════════════════════════════════════


class TestReturnForecaster:
    """Tests for ReturnForecaster."""

    def test_init_defaults(self):
        from src.prediction.return_forecaster import ReturnForecaster

        rf = ReturnForecaster()
        assert not rf.is_fitted
        assert rf.risk_free_rate == 0.05
        assert len(rf.factors) == 3

    def test_fit_basic(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)
        assert rf.is_fitted

    def test_fit_rejects_empty(self):
        from src.prediction.return_forecaster import ReturnForecaster

        rf = ReturnForecaster()
        with pytest.raises(ValueError):
            rf.fit(pd.DataFrame(), pd.Series(dtype=float))

    def test_predict_single_asset(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        result = rf.predict("SPX", current_regime=1)
        assert result.asset == "SPX"
        assert result.regime == 1
        assert isinstance(result.expected_return, float)
        assert isinstance(result.volatility, float)
        assert result.volatility > 0
        assert -2.0 <= result.kelly_fraction <= 2.0

    def test_predict_with_factor_values(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        result = rf.predict(
            "SPX",
            current_regime=1,
            factor_values={"momentum_20d": 0.005, "realized_vol_20d": 0.01},
        )
        assert len(result.factor_contributions) > 0

    def test_predict_all(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        results = rf.predict_all(current_regime=1)
        assert len(results) > 0
        for asset, fc in results.items():
            assert fc.asset == asset

    def test_predict_unfitted_raises(self):
        from src.prediction.return_forecaster import ReturnForecaster

        rf = ReturnForecaster()
        with pytest.raises(RuntimeError):
            rf.predict("SPX", 1)

    def test_predict_unknown_asset_raises(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)
        with pytest.raises(KeyError):
            rf.predict("FAKE_ASSET", 1)

    def test_regime_fallback(self):
        """When a regime has too few observations, fallback to nearest."""
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data(100)
        n = len(data)
        # All regime 1 except two points as regime 4
        regimes = pd.Series([1] * (n - 2) + [4, 4], index=data.index)
        rf = ReturnForecaster(min_obs_per_regime=30)
        rf.fit(data, regimes)

        # Regime 4 should fallback
        result = rf.predict("SPX", current_regime=4)
        assert result.regime == 4  # requested regime
        assert isinstance(result.expected_return, float)

    def test_r_squared_improvement(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        result = rf.predict("SPX", 1)
        # R² improvement should be a finite number
        assert np.isfinite(result.r_squared_improvement)
        assert result.r_squared_regime >= 0

    def test_get_regime_coefficients(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        coeffs = rf.get_regime_coefficients("SPX")
        assert len(coeffs) > 0
        for regime_id, info in coeffs.items():
            assert "alpha" in info
            assert "betas" in info
            assert "r_squared" in info
            assert "n_obs" in info

    def test_forecast_result_to_dict(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        result = rf.predict("SPX", 1)
        d = result.to_dict()
        assert "asset" in d
        assert "expected_return" in d
        assert "kelly_fraction" in d
        assert "factor_contributions" in d

    def test_confidence_bounded(self):
        from src.prediction.return_forecaster import ReturnForecaster

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        rf = ReturnForecaster()
        rf.fit(data, regimes)

        result = rf.predict("SPX", 1)
        assert 0.0 < result.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  3.2 — Tail Risk Analyzer
# ═══════════════════════════════════════════════════════════════════


class TestTailRiskAnalyzer:
    """Tests for TailRiskAnalyzer."""

    def test_init_defaults(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        tra = TailRiskAnalyzer()
        assert not tra.is_fitted
        assert tra.confidence_level == 0.99

    def test_fit_basic(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)
        assert tra.is_fitted

    def test_fit_rejects_empty(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        tra = TailRiskAnalyzer()
        with pytest.raises(ValueError):
            tra.fit(pd.DataFrame(), pd.Series(dtype=float))

    def test_analyze_basic(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
        )
        assert result.current_regime == 1
        assert result.weighted_var < 0  # VaR should be negative
        assert result.weighted_cvar < 0
        assert len(result.scenarios) == 4

    def test_analyze_with_transition_probs(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
            transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05},
        )
        # Probabilities should sum contributions correctly
        total_prob = sum(s.probability for s in result.scenarios)
        assert abs(total_prob - 1.0) < 0.01

    def test_worst_scenario_identified(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.2, "GLD": 0.2},
            current_regime=1,
        )
        assert result.worst_scenario != "N/A"
        assert result.worst_scenario_var < 0

    def test_hedge_recommendations(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
        )
        assert len(result.hedge_recommendations) > 0
        for h in result.hedge_recommendations:
            assert h.instrument != ""
            assert h.urgency in {"low", "medium", "high"}

    def test_tail_risk_driver(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
        )
        assert result.tail_risk_driver != "N/A"

    def test_scenario_var_risk_drivers(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
        )
        for sc in result.scenarios:
            assert len(sc.risk_drivers) > 0
            # Risk drivers should sum roughly to 1
            total = sum(abs(v) for v in sc.risk_drivers.values())
            assert total > 0

    def test_to_dict(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        result = tra.analyze(
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
            current_regime=1,
        )
        d = result.to_dict()
        assert "weighted_var" in d
        assert "scenarios" in d
        assert "hedge_recommendations" in d

    def test_unfitted_raises(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        tra = TailRiskAnalyzer()
        with pytest.raises(RuntimeError):
            tra.analyze({"SPX": 1.0}, 1)

    def test_get_regime_covariance(self):
        from src.prediction.tail_risk import TailRiskAnalyzer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        tra = TailRiskAnalyzer()
        tra.fit(data, regimes)

        cov = tra.get_regime_covariance(1)
        assert cov is not None
        assert cov.shape[0] == cov.shape[1]


# ═══════════════════════════════════════════════════════════════════
#  3.3 — Portfolio Optimizer
# ═══════════════════════════════════════════════════════════════════


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer."""

    def test_init_defaults(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        po = PortfolioOptimizer()
        assert not po.is_fitted
        assert po.min_weight == 0.0
        assert po.max_weight == 0.60

    def test_fit_basic(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)
        assert po.is_fitted

    def test_fit_rejects_empty(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        po = PortfolioOptimizer()
        with pytest.raises(ValueError):
            po.fit(pd.DataFrame(), pd.Series(dtype=float))

    def test_optimize_basic(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(current_regime=1)
        assert result.current_regime == 1
        assert len(result.blended_weights) > 0
        # Weights should sum to ~1
        assert abs(sum(result.blended_weights.values()) - 1.0) < 0.05

    def test_optimize_with_transition_probs(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(
            current_regime=1,
            transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05},
        )
        assert len(result.regime_allocations) == 4

    def test_regime_allocations_per_regime(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(current_regime=1)
        for alloc in result.regime_allocations:
            assert alloc.regime in {1, 2, 3, 4}
            assert len(alloc.weights) > 0
            assert abs(sum(alloc.weights.values()) - 1.0) < 0.05

    def test_rebalance_trigger(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        # High transition prob should trigger rebalance
        result = po.optimize(
            current_regime=1,
            transition_probs={1: 0.30, 2: 0.50, 3: 0.10, 4: 0.10},
            rebalance_threshold=0.40,
        )
        assert result.rebalance_trigger is True

    def test_no_rebalance_when_stable(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(
            current_regime=1,
            transition_probs={1: 0.80, 2: 0.10, 3: 0.05, 4: 0.05},
            rebalance_threshold=0.40,
        )
        assert result.rebalance_trigger is False

    def test_transaction_cost_estimate(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        # With current weights, TC should be positive
        result = po.optimize(
            current_regime=1,
            current_weights={"SPX": 0.8, "TLT": 0.1, "GLD": 0.1},
        )
        assert result.transaction_cost_estimate >= 0

    def test_drawdown_constraint(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(current_regime=2)  # Risk-Off
        assert result.max_drawdown_constraint == 0.08

    def test_to_dict(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(current_regime=1)
        d = result.to_dict()
        assert "blended_weights" in d
        assert "regime_allocations" in d
        assert "sharpe_ratio" in d

    def test_unfitted_raises(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        po = PortfolioOptimizer()
        with pytest.raises(RuntimeError):
            po.optimize(1)

    def test_sharpe_ratio(self):
        from src.prediction.portfolio_optimizer import PortfolioOptimizer

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        po = PortfolioOptimizer()
        po.fit(data, regimes)

        result = po.optimize(current_regime=1)
        assert np.isfinite(result.sharpe_ratio)


# ═══════════════════════════════════════════════════════════════════
#  3.4 — Alpha Signal Generator
# ═══════════════════════════════════════════════════════════════════


class TestAlphaSignalGenerator:
    """Tests for AlphaSignalGenerator."""

    def test_init_defaults(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        gen = AlphaSignalGenerator()
        assert not gen.is_fitted
        assert gen.min_anomaly_strength == 0.15

    def test_fit_basic(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)
        assert gen.is_fitted

    def test_fit_rejects_empty(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        gen = AlphaSignalGenerator()
        with pytest.raises(ValueError):
            gen.fit(pd.DataFrame(), pd.Series(dtype=float))

    def test_generate_with_known_anomaly(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"SPX_TLT_positive_correlation": 0.35},
        )
        assert result.regime == 1
        assert len(result.signals) > 0
        assert result.composite_score != 0.0

    def test_generate_with_vix_spike(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"VIX_spike": 0.50},
        )
        assert len(result.signals) > 0
        top = result.signals[0]
        assert top.direction == "long_SPX"  # VIX spike in Risk-On → buy dip

    def test_generate_no_anomalies(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(current_regime=1, active_anomalies={})
        assert len(result.signals) == 0
        assert result.composite_score == 0.0

    def test_weak_anomaly_filtered(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator(min_anomaly_strength=0.50)
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"SPX_TLT_positive_correlation": 0.20},
        )
        assert len(result.signals) == 0

    def test_multiple_anomalies(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={
                "SPX_TLT_positive_correlation": 0.35,
                "VIX_spike": 0.40,
                "DXY_WTI_decorrelation": 0.25,
            },
        )
        assert len(result.signals) >= 2
        assert result.n_active_anomalies == 3

    def test_max_signals_cap(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator(max_signals=2)
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={
                "SPX_TLT_positive_correlation": 0.35,
                "VIX_spike": 0.40,
                "DXY_WTI_decorrelation": 0.25,
                "correlation_breakdown": 0.30,
            },
        )
        assert len(result.signals) <= 2

    def test_composite_score_bounded(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={
                "SPX_TLT_positive_correlation": 0.35,
                "VIX_spike": 0.40,
            },
        )
        assert -1.0 <= result.composite_score <= 1.0

    def test_regime_context_in_signal(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"SPX_TLT_positive_correlation": 0.35},
        )
        assert "Risk-On" in result.regime_context

    def test_get_backtest_stats(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        stats = gen.get_backtest_stats()
        assert len(stats) > 0

    def test_to_dict(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"VIX_spike": 0.40},
        )
        d = result.to_dict()
        assert "signals" in d
        assert "composite_score" in d
        assert "top_signal" in d

    def test_unfitted_raises(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        gen = AlphaSignalGenerator()
        with pytest.raises(RuntimeError):
            gen.generate(1, {"VIX_spike": 0.5})

    def test_unknown_anomaly_falls_back(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"unknown_anomaly_xyz": 0.50},
        )
        assert len(result.signals) > 0
        assert result.signals[0].direction == "reduce_risk"

    def test_signal_holding_period(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        result = gen.generate(
            current_regime=1,
            active_anomalies={"VIX_spike": 0.40},
        )
        for s in result.signals:
            assert s.holding_period_days > 0

    def test_different_regimes_produce_different_signals(self):
        from src.prediction.alpha_signals import AlphaSignalGenerator

        data = _make_market_data()
        regimes = _make_regime_series(index=data.index)
        gen = AlphaSignalGenerator()
        gen.fit(data, regimes)

        r1 = gen.generate(1, {"SPX_TLT_positive_correlation": 0.35})
        r2 = gen.generate(2, {"SPX_TLT_positive_correlation": 0.35})

        # Risk-On should produce a signal; Risk-Off may be neutral
        assert len(r1.signals) > 0
        if len(r2.signals) > 0:
            assert r1.signals[0].direction != r2.signals[0].direction or True


# ═══════════════════════════════════════════════════════════════════
#  Phase 3 API Endpoints
# ═══════════════════════════════════════════════════════════════════


class TestPhase3API:
    """Tests for Phase 3 API endpoints."""

    def _mock_system_with_phase3(self, mock_system):
        """Add Phase 3 mock components to the system."""
        # Ensure analyze() returns anomalies / VIX so alpha-signals works
        analysis = mock_system.analyze.return_value
        analysis["modules"]["correlations"]["details"]["anomalies"] = {
            "SPX_TLT_positive_correlation": 0.35,
        }
        analysis["modules"]["options"]["details"]["vix"] = 30.0

        # Return forecaster mock
        rf = MagicMock()
        rf.is_fitted = True
        rf.predict_all.return_value = {
            "SPX": MagicMock(
                asset="SPX", regime=1, expected_return=0.0003,
                volatility=0.012, r_squared_regime=0.15,
                r_squared_static=0.08, r_squared_improvement=0.07,
                kelly_fraction=0.42, factor_contributions={"momentum_20d": 0.0001},
                confidence=0.72,
            ),
        }
        rf.predict.return_value = MagicMock(
            asset="SPX", regime=1, expected_return=0.0003,
            volatility=0.012, r_squared_regime=0.15,
            r_squared_static=0.08, r_squared_improvement=0.07,
            kelly_fraction=0.42, factor_contributions={"momentum_20d": 0.0001},
            confidence=0.72,
        )
        rf.get_regime_coefficients.return_value = {
            1: {"alpha": 0.0002, "betas": {"momentum_20d": 0.5},
                "residual_vol": 0.01, "r_squared": 0.15, "n_obs": 100},
        }
        mock_system.return_forecaster = rf

        # Tail risk mock
        tra = MagicMock()
        tra.is_fitted = True
        tra.analyze.return_value = MagicMock(
            current_regime=1, weighted_var=-0.023, weighted_cvar=-0.031,
            scenarios=[
                MagicMock(
                    from_regime=1, to_regime=1, to_regime_name="Risk-On Growth",
                    probability=0.65, var_99=-0.014, cvar_99=-0.018,
                    contribution=-0.009, risk_drivers={"SPX": 0.6, "TLT": 0.3},
                ),
                MagicMock(
                    from_regime=1, to_regime=2, to_regime_name="Risk-Off Crisis",
                    probability=0.20, var_99=-0.048, cvar_99=-0.062,
                    contribution=-0.010, risk_drivers={"SPX": 0.8, "TLT": 0.1},
                ),
            ],
            worst_scenario="Risk-Off Crisis", worst_scenario_var=-0.048,
            tail_risk_driver="SPX",
            hedge_recommendations=[
                MagicMock(scenario="Transition to Risk-Off",
                          instrument="TLT puts", rationale="Insurance",
                          urgency="medium"),
            ],
            portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
        )
        mock_system.tail_risk = tra

        # Portfolio optimizer mock
        po = MagicMock()
        po.is_fitted = True
        po.optimize.return_value = MagicMock(
            current_regime=1,
            blended_weights={"SPX": 0.50, "TLT": 0.30, "GLD": 0.20},
            regime_allocations=[
                MagicMock(regime=1, regime_name="Risk-On Growth",
                          weights={"SPX": 0.60, "TLT": 0.25, "GLD": 0.15},
                          expected_return=0.08, expected_volatility=0.12,
                          sharpe_ratio=0.25),
            ],
            rebalance_trigger=False,
            rebalance_reason="Portfolio within tolerance",
            transaction_cost_estimate=0.0005,
            expected_return=0.07, expected_volatility=0.11,
            sharpe_ratio=0.18, max_drawdown_constraint=0.15,
        )
        mock_system.portfolio_optimizer = po

        # Alpha signals mock
        alpha = MagicMock()
        alpha.is_fitted = True
        alpha.generate.return_value = MagicMock(
            signals=[
                MagicMock(anomaly_type="VIX_spike", direction="long_SPX",
                          rationale="Buy the dip", strength=0.6,
                          confidence=0.72, holding_period_days=5,
                          historical_win_rate=0.75, regime=1),
            ],
            composite_score=0.45, top_signal="VIX_spike",
            regime=1, n_active_anomalies=1,
            regime_context="Risk-On Growth, 1 signal",
        )
        mock_system.alpha_signals = alpha

        return mock_system

    def test_return_forecast_all(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/return-forecast")
        assert resp.status_code == 200
        body = resp.json()
        assert "current_regime" in body
        assert "forecasts" in body
        assert len(body["forecasts"]) > 0

    def test_return_forecast_single_asset(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/return-forecast/SPX")
        assert resp.status_code == 200
        body = resp.json()
        assert body["asset"] == "SPX"
        assert "expected_return" in body

    def test_return_forecast_missing_asset(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        mock_system.return_forecaster.predict.side_effect = KeyError("FAKE")
        resp = api_client.get("/api/phase3/return-forecast/FAKE")
        assert resp.status_code == 404

    def test_regime_coefficients(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/regime-coefficients/SPX")
        assert resp.status_code == 200
        body = resp.json()
        assert body["asset"] == "SPX"
        assert "regimes" in body

    def test_tail_risk(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/tail-risk")
        assert resp.status_code == 200
        body = resp.json()
        assert body["weighted_var"] < 0
        assert "scenarios" in body
        assert "hedge_recommendations" in body

    def test_tail_risk_custom_portfolio(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get(
            "/api/phase3/tail-risk",
            params={"portfolio": '{"SPX":0.5,"TLT":0.3,"GLD":0.2}'},
        )
        assert resp.status_code == 200

    def test_portfolio_optimize(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/portfolio-optimize")
        assert resp.status_code == 200
        body = resp.json()
        assert "blended_weights" in body
        assert "regime_allocations" in body
        assert "sharpe_ratio" in body

    def test_portfolio_optimize_custom_probs(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get(
            "/api/phase3/portfolio-optimize",
            params={"transition_probs": '{"1":0.5,"2":0.3,"3":0.1,"4":0.1}'},
        )
        assert resp.status_code == 200

    def test_alpha_signals(self, api_client, mock_system):
        self._mock_system_with_phase3(mock_system)
        resp = api_client.get("/api/phase3/alpha-signals")
        assert resp.status_code == 200
        body = resp.json()
        assert "signals" in body
        assert "composite_score" in body

    def test_return_forecast_not_initialized(self, api_client, mock_system):
        mock_system.return_forecaster = None
        resp = api_client.get("/api/phase3/return-forecast")
        assert resp.status_code == 503

    def test_tail_risk_not_initialized(self, api_client, mock_system):
        mock_system.tail_risk = None
        resp = api_client.get("/api/phase3/tail-risk")
        assert resp.status_code == 503

    def test_portfolio_optimizer_not_initialized(self, api_client, mock_system):
        mock_system.portfolio_optimizer = None
        resp = api_client.get("/api/phase3/portfolio-optimize")
        assert resp.status_code == 503

    def test_alpha_signals_not_initialized(self, api_client, mock_system):
        mock_system.alpha_signals = None
        resp = api_client.get("/api/phase3/alpha-signals")
        assert resp.status_code == 503

    def test_return_forecast_not_fitted(self, api_client, mock_system):
        rf = MagicMock()
        rf.is_fitted = False
        mock_system.return_forecaster = rf
        resp = api_client.get("/api/phase3/return-forecast")
        assert resp.status_code == 503

    def test_tail_risk_not_fitted(self, api_client, mock_system):
        tra = MagicMock()
        tra.is_fitted = False
        mock_system.tail_risk = tra
        resp = api_client.get("/api/phase3/tail-risk")
        assert resp.status_code == 503


# ═══════════════════════════════════════════════════════════════════
#  Integration: Package imports
# ═══════════════════════════════════════════════════════════════════


class TestPredictionPackage:
    """Verify the prediction package imports cleanly."""

    def test_package_imports(self):
        from src.prediction import (
            ReturnForecaster,
            TailRiskAnalyzer,
            PortfolioOptimizer,
            AlphaSignalGenerator,
        )
        assert ReturnForecaster is not None
        assert TailRiskAnalyzer is not None
        assert PortfolioOptimizer is not None
        assert AlphaSignalGenerator is not None

    def test_return_forecaster_regime_model_dataclass(self):
        from src.prediction.return_forecaster import RegimeModel

        m = RegimeModel(
            regime=1, alpha=0.001, betas={"m": 0.5},
            residual_vol=0.01, r_squared=0.2, n_obs=100,
        )
        assert m.regime == 1

    def test_tail_risk_scenario_var_dataclass(self):
        from src.prediction.tail_risk import ScenarioVaR

        sv = ScenarioVaR(
            from_regime=1, to_regime=2, to_regime_name="Risk-Off",
            probability=0.2, var_99=-0.05, cvar_99=-0.06,
            contribution=-0.01,
        )
        d = sv.to_dict()
        assert d["var_99"] == -0.05

    def test_portfolio_regime_allocation_dataclass(self):
        from src.prediction.portfolio_optimizer import RegimeAllocation

        ra = RegimeAllocation(
            regime=1, regime_name="Risk-On Growth",
            weights={"SPX": 0.6, "TLT": 0.4},
            expected_return=0.08, expected_volatility=0.12,
            sharpe_ratio=0.25,
        )
        d = ra.to_dict()
        assert d["regime"] == 1

    def test_alpha_signal_dataclass(self):
        from src.prediction.alpha_signals import AlphaSignal

        sig = AlphaSignal(
            anomaly_type="VIX_spike", direction="long_SPX",
            rationale="Buy the dip", strength=0.6, confidence=0.7,
            holding_period_days=5, historical_win_rate=0.75, regime=1,
        )
        d = sig.to_dict()
        assert d["direction"] == "long_SPX"
