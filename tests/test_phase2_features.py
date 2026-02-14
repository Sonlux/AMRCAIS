"""
Tests for Phase 2: Intelligence Expansion features.

Covers:
- Regime Transition Probability Model (transition_model.py)
- Macro Surprise Decay Model (macro_surprise_decay.py)
- Cross-Asset Contagion Network (contagion_network.py)
- Multi-Timeframe Regime Detection (multi_timeframe.py)
- Natural Language Narrative Generator (narrative_generator.py)
- Phase 2 API endpoints (routes/phase2.py)
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
    """Generate synthetic market data for testing."""
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
    return data


def _make_regime_series(n_days: int = 300) -> pd.Series:
    """Generate synthetic regime history."""
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.now(), periods=n_days, freq="B")
    # Regime 1 dominant
    regimes = np.random.choice([1, 2, 3, 4], size=n_days, p=[0.5, 0.2, 0.15, 0.15])
    return pd.Series(regimes, index=dates)


def _make_analysis_dict() -> dict:
    """Build a full analysis dict that matches main.py analyze() output."""
    return {
        "regime": {
            "id": 1,
            "name": "Risk-On Growth",
            "confidence": 0.85,
            "disagreement": 0.15,
            "individual_predictions": {"hmm": 1, "ml": 1, "correlation": 2, "volatility": 1},
            "probabilities": {"1": 0.6, "2": 0.2, "3": 0.1, "4": 0.1},
            "transition_warning": False,
        },
        "modules": {
            "macro": {
                "signal": {"signal": "bullish", "strength": 0.7, "confidence": 0.8,
                           "explanation": "GDP growth strong", "regime_context": "Risk-on"},
                "details": {},
            },
            "yield_curve": {
                "signal": {"signal": "neutral", "strength": 0.3, "confidence": 0.6,
                           "explanation": "Curve flat", "regime_context": "Watching"},
                "details": {},
            },
            "options": {
                "signal": {"signal": "bearish", "strength": 0.5, "confidence": 0.7,
                           "explanation": "VIX elevated", "regime_context": "Hedging"},
                "details": {},
            },
            "factors": {
                "signal": {"signal": "bullish", "strength": 0.6, "confidence": 0.75,
                           "explanation": "Quality rotating in", "regime_context": "Growth"},
                "details": {},
            },
            "correlations": {
                "signal": {"signal": "cautious", "strength": 0.4, "confidence": 0.55,
                           "explanation": "Correlations rising", "regime_context": "Uncertainty"},
                "details": {},
            },
        },
        "summary": {"overall_signal": "bullish", "signal_strength": 0.5},
    }


# ═══════════════════════════════════════════════════════════════════
#  Regime Transition Model
# ═══════════════════════════════════════════════════════════════════


class TestRegimeTransitionModel:
    """Tests for RegimeTransitionModel."""

    def test_initialization(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(horizon_days=30)
        assert model.horizon_days == 30
        assert not model.is_fitted

    def test_fit_requires_min_history(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(min_history=100)
        data = _make_market_data(50)
        regimes = _make_regime_series(50)

        with pytest.raises(ValueError, match="Need >="):
            model.fit(data, regimes)

    def test_fit_and_predict(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(horizon_days=30, min_history=60)
        data = _make_market_data(300)
        regimes = _make_regime_series(300)

        model.fit(data, regimes)
        assert model.is_fitted

        forecast = model.predict(current_regime=1, market_data=data)
        assert forecast.current_regime == 1
        assert forecast.horizon_days == 30
        assert sum(forecast.blended_probs.values()) == pytest.approx(1.0, abs=0.01)
        assert 0 <= forecast.transition_risk <= 1.0
        assert 1 <= forecast.most_likely_next <= 4
        assert 0 <= forecast.confidence <= 1.0

    def test_hmm_probs_sum_to_one(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(horizon_days=10, min_history=60)
        data = _make_market_data(200)
        regimes = _make_regime_series(200)

        model.fit(data, regimes)
        forecast = model.predict(current_regime=2, market_data=data)

        assert sum(forecast.hmm_probs.values()) == pytest.approx(1.0, abs=0.01)
        assert sum(forecast.indicator_probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_predict_before_fit_raises(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel()
        data = _make_market_data(100)

        with pytest.raises(ValueError, match="fitted"):
            model.predict(current_regime=1, market_data=data)

    def test_forecast_to_dict(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(min_history=60)
        data = _make_market_data(200)
        regimes = _make_regime_series(200)

        model.fit(data, regimes)
        forecast = model.predict(current_regime=1, market_data=data)
        d = forecast.to_dict()

        assert "current_regime" in d
        assert "blended_probs" in d
        assert "transition_risk" in d
        assert "most_likely_next_name" in d

    def test_custom_horizon(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(min_history=60)
        data = _make_market_data(200)
        regimes = _make_regime_series(200)
        model.fit(data, regimes)

        f10 = model.predict(current_regime=1, market_data=data, horizon_days=10)
        f60 = model.predict(current_regime=1, market_data=data, horizon_days=60)

        assert f10.horizon_days == 10
        assert f60.horizon_days == 60

    def test_all_regimes_have_probs(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(min_history=60)
        data = _make_market_data(200)
        regimes = _make_regime_series(200)
        model.fit(data, regimes)

        for from_regime in [1, 2, 3, 4]:
            forecast = model.predict(current_regime=from_regime, market_data=data)
            assert len(forecast.blended_probs) == 4
            for r in [1, 2, 3, 4]:
                assert r in forecast.blended_probs

    def test_transition_matrix_valid(self):
        from src.regime_detection.transition_model import RegimeTransitionModel

        model = RegimeTransitionModel(min_history=60)
        data = _make_market_data(200)
        regimes = _make_regime_series(200)
        model.fit(data, regimes)

        # Internal transition matrix should be row-stochastic
        tm = model._transition_matrix
        assert tm.shape == (4, 4)
        for i in range(4):
            assert tm[i, :].sum() == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
#  Macro Surprise Decay
# ═══════════════════════════════════════════════════════════════════


class TestSurpriseDecayModel:
    """Tests for SurpriseDecayModel."""

    def test_initialization(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        assert model.name == "SurpriseDecayModel"
        assert len(model._active_surprises) == 0

    def test_add_surprise_with_value(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)

        ds = model.add_surprise("NFP", surprise=2.5)
        assert ds.indicator == "NFP"
        assert ds.surprise == 2.5
        assert ds.regime_at_release == 1
        assert len(model._active_surprises) == 1

    def test_add_surprise_with_actual_consensus(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)

        ds = model.add_surprise("CPI", actual=3.5, consensus=3.0)
        assert ds.surprise != 0.0  # Should be computed
        assert ds.indicator == "CPI"

    def test_add_surprise_missing_both_raises(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        with pytest.raises(ValueError, match="surprise"):
            model.add_surprise("NFP")

    def test_decay_reduces_impact_over_time(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)

        release = datetime.now() - timedelta(days=0)
        model.add_surprise("NFP", surprise=1.0, release_date=release)

        impact_now = model.get_current_impact(current_date=release)
        impact_later = model.get_current_impact(
            current_date=release + timedelta(days=10)
        )

        assert impact_later < impact_now

    def test_cumulative_index(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)

        model.add_surprise("NFP", surprise=2.0)
        model.add_surprise("CPI", surprise=-1.0)

        idx = model.get_cumulative_index()
        assert "index" in idx
        assert "direction" in idx
        assert isinstance(idx["direction"], str)
        assert len(idx["direction"]) > 0

    def test_decay_curves(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)
        model.add_surprise("NFP", surprise=1.5)

        curves = model.get_decay_curves(forward_days=30)
        assert isinstance(curves, dict)
        assert len(curves) > 0

        for indicator, points in curves.items():
            assert len(points) > 0
            # Impact should decrease over days
            impacts = [p["impact"] for p in points]
            assert impacts[0] >= impacts[-1]

    def test_prune_stale_surprises(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel(stale_threshold=0.05)
        model.update_regime(1, 0.85)

        # Add a very old surprise
        old_date = datetime.now() - timedelta(days=365)
        model.add_surprise("NFP", surprise=1.0, release_date=old_date)

        pruned = model._prune_stale(datetime.now())
        assert pruned >= 1
        assert len(model._active_surprises) == 0

    def test_regime_changes_half_life(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()

        hl_risk_on = model._get_half_life("NFP", regime=1)
        hl_crisis = model._get_half_life("NFP", regime=2)

        # Crisis should have longer half-life for NFP
        assert hl_crisis > hl_risk_on

    def test_analyze_returns_valid_dict(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        model.update_regime(1, 0.85)
        model.add_surprise("NFP", surprise=2.0)

        data = _make_market_data(60)
        result = model.analyze(data)

        assert "signal" in result
        assert "cumulative_index" in result
        signal = result["signal"]
        # signal may be a ModuleSignal object or dict
        if hasattr(signal, "signal"):
            assert signal.signal in ("bullish", "bearish", "neutral", "cautious")
        else:
            assert signal["signal"] in ("bullish", "bearish", "neutral", "cautious")

    def test_get_regime_parameters(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        params = model.get_regime_parameters(regime=2)
        assert isinstance(params, dict)
        assert "regime" in params
        assert "half_lives" in params
        assert "weights" in params

    def test_decaying_surprise_to_dict(self):
        from src.modules.macro_surprise_decay import DecayingSurprise

        ds = DecayingSurprise(
            indicator="CPI",
            surprise=1.5,
            release_date=datetime.now(),
            half_life_days=5.0,
            initial_weight=1.2,
            regime_at_release=3,
        )
        d = ds.to_dict()
        assert d["indicator"] == "CPI"
        assert d["surprise"] == 1.5

    def test_half_lives_property(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        hl = model.get_half_lives()
        assert isinstance(hl, dict)
        assert "NFP" in hl
        assert isinstance(hl["NFP"], dict)

    @property
    def active_surprises(self):
        from src.modules.macro_surprise_decay import SurpriseDecayModel

        model = SurpriseDecayModel()
        assert hasattr(model, "_active_surprises")


# ═══════════════════════════════════════════════════════════════════
#  Contagion Network
# ═══════════════════════════════════════════════════════════════════


class TestContagionNetwork:
    """Tests for ContagionNetwork module."""

    def test_initialization(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        assert cn.name == "ContagionNetwork"

    def test_analyze_returns_valid_dict(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        cn.update_regime(1, 0.85)

        data = _make_market_data(200)
        result = cn.analyze(data)

        assert "signal" in result
        assert "granger_network" in result
        assert "contagion_flags" in result

    def test_granger_results_structure(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        cn.update_regime(1, 0.85)

        data = _make_market_data(200)
        result = cn.analyze(data)

        granger = result.get("granger_network", [])
        assert isinstance(granger, list)

        if granger:
            g = granger[0]
            assert "cause" in g
            assert "effect" in g
            assert "f_stat" in g
            assert "p_value" in g

    def test_spillover_results(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        cn.update_regime(1, 0.85)

        data = _make_market_data(200)
        result = cn.analyze(data)

        spillover = result.get("spillover", {})
        assert isinstance(spillover, dict)
        # spillover may be empty if computation fails, but should be a dict
        if spillover:
            assert "total_spillover_index" in spillover

    def test_network_graph(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        cn.update_regime(1, 0.85)

        data = _make_market_data(200)
        result = cn.analyze(data)

        network = result.get("network_graph", {})
        assert isinstance(network, dict)
        assert "nodes" in network
        assert "edges" in network

    def test_granger_test_pair(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        data = _make_market_data(200)

        returns = data.pct_change().dropna()
        result = cn._granger_test_pair(
            returns["SPX"].values,
            returns["TLT"].values,
            max_lag=3,
        )

        assert "cause" in result or "f_stat" in result

    def test_regime_parameters(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        params = cn.get_regime_parameters(regime=2)
        assert isinstance(params, dict)
        assert "expected_spillover" in params
        assert "regime" in params

    def test_contagion_flags(self):
        from src.modules.contagion_network import ContagionNetwork

        cn = ContagionNetwork()
        cn.update_regime(1, 0.85)

        data = _make_market_data(200)
        result = cn.analyze(data)
        flags = result.get("contagion_flags", {})
        assert isinstance(flags, dict)

    def test_granger_result_to_dict(self):
        from src.modules.contagion_network import GrangerResult

        gr = GrangerResult(
            cause="SPX", effect="TLT",
            f_statistic=3.5, p_value=0.02, lag=2, significant=True,
        )
        d = gr.to_dict()
        assert d["cause"] == "SPX"
        assert d["significant"] is True

    def test_spillover_result_to_dict(self):
        from src.modules.contagion_network import SpilloverResult
        import numpy as np

        sr = SpilloverResult(
            total_spillover_index=45.0,
            directional_to={"SPX": 30.0},
            directional_from={"SPX": 20.0},
            net_spillover={"SPX": 10.0},
            pairwise=np.array([[0, 1], [1, 0]], dtype=float),
            assets=["SPX", "TLT"],
        )
        d = sr.to_dict()
        assert d["total_spillover_index"] == 45.0

    def test_f_survival_fallback(self):
        """Test _f_survival static method with scipy and without."""
        from src.modules.contagion_network import ContagionNetwork

        p = ContagionNetwork._f_survival(5.0, 2, 100)
        assert 0 <= p <= 1.0

    def test_network_density_method(self):
        from src.modules.contagion_network import ContagionNetwork, GrangerResult

        cn = ContagionNetwork()
        # Empty network
        d = cn._network_density([], 0)
        assert d == 0.0

        # Small network with GrangerResult objects
        links = [GrangerResult(
            cause="SPX", effect="TLT",
            f_statistic=5.0, p_value=0.01, lag=1, significant=True,
        )]
        d = cn._network_density(links, 3)
        assert 0 < d <= 1.0


# ═══════════════════════════════════════════════════════════════════
#  Multi-Timeframe Regime Detection
# ═══════════════════════════════════════════════════════════════════


class TestMultiTimeframeDetector:
    """Tests for MultiTimeframeDetector."""

    def test_initialization(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        assert not mtf.is_fitted

    def test_fit_with_sufficient_data(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        data = _make_market_data(300)

        # fit should not raise with sufficient data
        mtf.fit(data)
        assert mtf.is_fitted

    def test_predict_returns_result(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        data = _make_market_data(300)
        mtf.fit(data)

        result = mtf.predict(data)
        assert result is not None
        assert hasattr(result, "daily")
        assert hasattr(result, "weekly")
        assert hasattr(result, "monthly")

    def test_result_to_dict(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        data = _make_market_data(300)
        mtf.fit(data)

        result = mtf.predict(data)
        d = result.to_dict()

        assert "daily" in d
        assert "weekly" in d
        assert "monthly" in d
        assert "conflict_detected" in d
        assert "trade_signal" in d

    def test_timeframe_result_to_dict(self):
        from src.regime_detection.multi_timeframe import TimeframeResult

        tfr = TimeframeResult(
            timeframe="daily",
            regime=1,
            regime_name="Risk-On Growth",
            confidence=0.85,
            disagreement=0.15,
            transition_warning=False,
            duration=10,
        )
        d = tfr.to_dict()
        assert d["timeframe"] == "daily"
        assert d["regime"] == 1
        assert d["duration"] == 10

    def test_resample_to_weekly(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        data = _make_market_data(100)
        weekly = MultiTimeframeDetector._resample_to_weekly(data)

        assert len(weekly) < len(data)
        assert len(weekly) >= 15  # ~100 trading days = ~20 weeks

    def test_resample_to_monthly(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        data = _make_market_data(300)
        monthly = MultiTimeframeDetector._resample_to_monthly(data)

        assert len(monthly) < len(data)
        assert len(monthly) >= 10  # ~300 trading days = ~14 months

    def test_agreement_score_range(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        data = _make_market_data(300)
        mtf.fit(data)

        result = mtf.predict(data)
        assert 0 <= result.agreement_score <= 1.0

    def test_trade_signal_non_empty(self):
        from src.regime_detection.multi_timeframe import MultiTimeframeDetector

        mtf = MultiTimeframeDetector()
        data = _make_market_data(300)
        mtf.fit(data)

        result = mtf.predict(data)
        assert isinstance(result.trade_signal, str)
        assert len(result.trade_signal) > 0

    def test_interpret_timeframe_stack(self):
        from src.regime_detection.multi_timeframe import (
            MultiTimeframeDetector, TimeframeResult,
        )

        mtf = MultiTimeframeDetector()
        daily = TimeframeResult(
            timeframe="daily", regime=1, regime_name="Risk-On Growth",
            confidence=0.8, disagreement=0.2, transition_warning=False,
            duration=10,
        )
        weekly = TimeframeResult(
            timeframe="weekly", regime=1, regime_name="Risk-On Growth",
            confidence=0.8, disagreement=0.2, transition_warning=False,
            duration=5,
        )
        monthly = TimeframeResult(
            timeframe="monthly", regime=1, regime_name="Risk-On Growth",
            confidence=0.8, disagreement=0.2, transition_warning=False,
            duration=3,
        )
        signal = mtf._interpret_timeframe_stack(daily, weekly, monthly)
        assert "risk_on" in signal.lower() or "bullish" in signal.lower()


# ═══════════════════════════════════════════════════════════════════
#  Narrative Generator
# ═══════════════════════════════════════════════════════════════════


class TestNarrativeGenerator:
    """Tests for NarrativeGenerator."""

    def test_initialization(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        assert gen is not None

    def test_generate_produces_briefing(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        assert briefing.headline != ""
        assert briefing.regime_section != ""
        assert briefing.signal_section != ""
        assert briefing.risk_section != ""
        assert briefing.full_text != ""

    def test_headline_contains_regime_name(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        assert "Risk-On Growth" in briefing.headline

    def test_signal_section_lists_modules(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        # Should mention at least one module signal
        assert "bullish" in briefing.signal_section.lower() or "bearish" in briefing.signal_section.lower()

    def test_positioning_section_generated(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator(include_positioning=True)
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        assert briefing.positioning_section != ""

    def test_positioning_disabled(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator(include_positioning=False)
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        assert briefing.positioning_section == ""

    def test_briefing_to_dict(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)
        d = briefing.to_dict()

        assert "headline" in d
        assert "regime_section" in d
        assert "full_text" in d
        assert "data_sources" in d
        assert "timestamp" in d

    def test_data_sources_populated(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        briefing = gen.generate(analysis)

        assert len(briefing.data_sources) > 0

    def test_all_four_regimes(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()

        for regime_id in [1, 2, 3, 4]:
            analysis = _make_analysis_dict()
            analysis["regime"]["id"] = regime_id
            names = {1: "Risk-On Growth", 2: "Risk-Off Crisis",
                     3: "Stagflation", 4: "Disinflationary Boom"}
            analysis["regime"]["name"] = names[regime_id]

            briefing = gen.generate(analysis)
            assert names[regime_id] in briefing.headline

    def test_risk_section_with_transition_warning(self):
        from src.narrative.narrative_generator import NarrativeGenerator

        gen = NarrativeGenerator()
        analysis = _make_analysis_dict()
        analysis["regime"]["transition_warning"] = True
        analysis["regime"]["disagreement"] = 0.65

        briefing = gen.generate(analysis)
        assert briefing.risk_section != ""

    def test_narrative_briefing_dataclass(self):
        from src.narrative.narrative_generator import NarrativeBriefing

        nb = NarrativeBriefing(
            headline="Test headline",
            regime_section="regime stuff",
            signal_section="signals",
            risk_section="risks",
            positioning_section="positions",
            full_text="all of it",
            data_sources={"headline": "regime data"},
        )
        assert nb.headline == "Test headline"
        assert nb.timestamp is not None


# ═══════════════════════════════════════════════════════════════════
#  Phase 2 API Schemas
# ═══════════════════════════════════════════════════════════════════


class TestPhase2Schemas:
    """Test Phase 2 Pydantic schemas."""

    def test_transition_forecast_response(self):
        from api.schemas import TransitionForecastResponse

        r = TransitionForecastResponse(
            current_regime=1,
            horizon_days=30,
            blended_probs={"1": 0.6, "2": 0.2, "3": 0.1, "4": 0.1},
            transition_risk=0.4,
            most_likely_next=2,
            confidence=0.75,
        )
        assert r.current_regime == 1
        assert r.transition_risk == 0.4

    def test_multi_timeframe_response(self):
        from api.schemas import MultiTimeframeResponse, TimeframeRegimeResponse

        daily = TimeframeRegimeResponse(
            timeframe="daily", regime=1, regime_name="Risk-On Growth",
            confidence=0.85, disagreement=0.15,
        )
        weekly = TimeframeRegimeResponse(
            timeframe="weekly", regime=1, regime_name="Risk-On Growth",
            confidence=0.80, disagreement=0.20,
        )
        monthly = TimeframeRegimeResponse(
            timeframe="monthly", regime=2, regime_name="Risk-Off Crisis",
            confidence=0.70, disagreement=0.30,
        )

        r = MultiTimeframeResponse(
            daily=daily, weekly=weekly, monthly=monthly,
            conflict_detected=True,
            agreement_score=0.67,
        )
        assert r.conflict_detected is True

    def test_contagion_analysis_response(self):
        from api.schemas import (
            ContagionAnalysisResponse, GrangerLinkResponse,
            SpilloverResponse, NetworkGraphResponse, ModuleSignalResponse,
        )

        signal = ModuleSignalResponse(module="contagion", signal="cautious", strength=0.5)
        spillover = SpilloverResponse(total_spillover_index=45.0)
        network = NetworkGraphResponse(nodes=[], edges=[])

        r = ContagionAnalysisResponse(
            signal=signal, spillover=spillover, network_graph=network,
        )
        assert r.n_significant_links == 0

    def test_surprise_index_response(self):
        from api.schemas import SurpriseIndexResponse

        r = SurpriseIndexResponse(
            index=1.5, direction="positive", active_surprises=3,
        )
        assert r.index == 1.5
        assert r.direction == "positive"

    def test_narrative_response(self):
        from api.schemas import NarrativeResponse

        r = NarrativeResponse(
            headline="Test", regime_section="regime",
            signal_section="signals", risk_section="risks",
            full_text="everything",
        )
        assert r.headline == "Test"

    def test_decay_curve_point(self):
        from api.schemas import DecayCurvePoint

        p = DecayCurvePoint(day=5, impact=0.3, is_stale=False)
        assert p.day == 5


# ═══════════════════════════════════════════════════════════════════
#  Phase 2 API Endpoints
# ═══════════════════════════════════════════════════════════════════


def _create_phase2_mock_system():
    """Create a mock system with Phase 2 components."""
    from tests.conftest import _create_mock_system

    system = _create_mock_system()

    # Add Phase 2 modules
    for name in ["contagion", "surprise_decay"]:
        mod = MagicMock()
        mod.get_regime_parameters.return_value = {"regime_name": "Risk-On Growth"}
        system.modules[name] = mod

    # Surprise decay specific mocks
    system.modules["surprise_decay"]._active_surprises = []
    system.modules["surprise_decay"].active_surprises = []
    system.modules["surprise_decay"].get_decay_curves.return_value = {
        "NFP": [{"day": 0, "impact": 1.0, "is_stale": False},
                {"day": 10, "impact": 0.5, "is_stale": False}],
    }

    # Add Phase 2 analysis results to mock
    system.analyze.return_value["modules"]["contagion"] = {
        "signal": {"signal": "cautious", "strength": 0.5, "confidence": 0.6,
                   "explanation": "High connectedness", "regime_context": "Risk-on"},
        "details": {
            "granger_network": [
                {"cause": "SPX", "effect": "TLT", "f_stat": 4.2,
                 "p_value": 0.01, "lag": 2, "significant": True},
            ],
            "spillover": {
                "total_spillover_index": 42.5,
                "directional_to": {"SPX": 30.0, "TLT": 25.0},
                "directional_from": {"SPX": 20.0, "TLT": 22.0},
                "net_spillover": {"SPX": 10.0, "TLT": 3.0},
                "pairwise": [[0, 10], [10, 0]],
                "assets": ["SPX", "TLT"],
            },
            "network_graph": {
                "nodes": [{"id": "SPX", "role": "transmitter"},
                          {"id": "TLT", "role": "receiver"}],
                "edges": [{"source": "SPX", "target": "TLT", "weight": 4.2}],
            },
            "contagion_flags": {"high_systemic_connectedness": False},
        },
    }
    system.analyze.return_value["modules"]["surprise_decay"] = {
        "signal": {"signal": "bullish", "strength": 0.6, "confidence": 0.7,
                   "explanation": "Surprise tailwind", "regime_context": "Growth"},
        "details": {
            "cumulative_index": 1.5,
            "direction": "positive",
            "components": {"NFP": 1.0, "CPI": 0.5},
            "active_surprises": 2,
            "total_historical": 5,
        },
    }

    # Transition model mock
    transition_model = MagicMock()
    transition_model.is_fitted = True
    transition_model.predict.return_value = MagicMock(
        current_regime=1,
        horizon_days=30,
        hmm_probs={1: 0.6, 2: 0.2, 3: 0.1, 4: 0.1},
        indicator_probs={1: 0.5, 2: 0.25, 3: 0.15, 4: 0.1},
        blended_probs={1: 0.55, 2: 0.22, 3: 0.13, 4: 0.1},
        leading_indicators={"vix_term_structure_slope": -0.02},
        transition_risk=0.45,
        most_likely_next=2,
        confidence=0.72,
    )
    system.transition_model = transition_model

    # Multi-timeframe mock
    multi_timeframe = MagicMock()
    multi_timeframe.is_fitted = True
    system.multi_timeframe = multi_timeframe

    # Narrative mock
    narrative_generator = MagicMock()
    system.narrative_generator = narrative_generator

    # Add Phase 2 analysis results
    system.analyze.return_value["transition_forecast"] = {
        "current_regime": 1,
        "horizon_days": 30,
        "hmm_probs": {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.1},
        "indicator_probs": {1: 0.5, 2: 0.25, 3: 0.15, 4: 0.1},
        "blended_probs": {1: 0.55, 2: 0.22, 3: 0.13, 4: 0.1},
        "leading_indicators": {"vix_term_structure_slope": -0.02},
        "transition_risk": 0.45,
        "most_likely_next": 2,
        "confidence": 0.72,
    }

    system.analyze.return_value["multi_timeframe"] = {
        "timeframes": {
            "daily": {"regime": 1, "regime_name": "Risk-On Growth",
                      "confidence": 0.85, "disagreement": 0.15, "duration": 20},
            "weekly": {"regime": 1, "regime_name": "Risk-On Growth",
                       "confidence": 0.80, "disagreement": 0.20, "duration": 8},
            "monthly": {"regime": 1, "regime_name": "Risk-On Growth",
                        "confidence": 0.75, "disagreement": 0.25, "duration": 3},
        },
        "conflict_detected": False,
        "highest_conviction": "daily",
        "trade_signal": "strong_bullish_all_timeframes_risk_on",
        "agreement_score": 1.0,
    }

    system.analyze.return_value["narrative"] = {
        "headline": "Risk-On Growth (85% confidence)",
        "regime_section": "Markets in risk-on mode with strong confidence.",
        "signal_section": "Majority of signals are bullish.",
        "risk_section": "No immediate risks detected.",
        "positioning_section": "Favor equities, underweight bonds.",
        "full_text": "Full narrative text here.",
        "data_sources": {"headline": "regime_classifier"},
    }

    # Data pipeline
    data_pipeline = MagicMock()
    data_pipeline.get_latest_data.return_value = _make_market_data(60)
    system.data_pipeline = data_pipeline

    return system


@pytest.fixture
def phase2_api_client():
    """FastAPI TestClient with Phase 2 mock system."""
    import api.dependencies as deps
    from contextlib import asynccontextmanager

    orig_system = deps._system
    orig_time = deps._startup_time
    orig_analysis = deps._last_analysis

    deps._system = _create_phase2_mock_system()
    deps._startup_time = time.time()
    deps._last_analysis = None

    from api.main import app

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    saved_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan

    from api.middleware import RateLimitMiddleware, CSRFMiddleware
    _orig_dispatch = RateLimitMiddleware.dispatch
    _orig_csrf_dispatch = CSRFMiddleware.dispatch

    async def _passthrough(self, request, call_next):
        return await call_next(request)

    RateLimitMiddleware.dispatch = _passthrough
    CSRFMiddleware.dispatch = _passthrough

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        yield client

    RateLimitMiddleware.dispatch = _orig_dispatch
    CSRFMiddleware.dispatch = _orig_csrf_dispatch
    app.router.lifespan_context = saved_lifespan
    deps._system = orig_system
    deps._startup_time = orig_time
    deps._last_analysis = orig_analysis


class TestPhase2Endpoints:
    """Test Phase 2 API endpoints."""

    def test_transition_forecast(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/transition-forecast")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_regime"] == 1
        assert "blended_probs" in data
        assert data["transition_risk"] == pytest.approx(0.45)

    def test_transition_forecast_custom_horizon(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/transition-forecast?horizon=10")
        assert resp.status_code == 200

    def test_multi_timeframe(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/multi-timeframe")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily" in data
        assert "weekly" in data
        assert "monthly" in data
        assert data["agreement_score"] == 1.0

    def test_contagion_analyze(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/contagion/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "signal" in data
        assert "granger_network" in data
        assert "spillover" in data
        assert data["spillover"]["total_spillover_index"] == pytest.approx(42.5)
        assert data["n_significant_links"] == 1

    def test_contagion_spillover_only(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/contagion/spillover")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_spillover_index" in data

    def test_surprise_decay_index(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/surprise-decay/index")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == pytest.approx(1.5)
        assert data["direction"] == "positive"
        assert data["active_surprises"] == 2

    def test_surprise_decay_curves(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/surprise-decay/curves")
        assert resp.status_code == 200
        data = resp.json()
        assert "curves" in data
        assert "NFP" in data["curves"]

    def test_surprise_decay_active(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/surprise-decay/active")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_narrative(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/narrative")
        assert resp.status_code == 200
        data = resp.json()
        assert "headline" in data
        assert "Risk-On Growth" in data["headline"]
        assert "full_text" in data

    def test_module_summary_includes_new_modules(self, phase2_api_client):
        """VALID_MODULES should now include contagion and surprise_decay."""
        resp = phase2_api_client.get("/api/modules/summary")
        assert resp.status_code == 200
        data = resp.json()
        module_names = [s["module"] for s in data["signals"]]
        assert "contagion" in module_names
        assert "surprise_decay" in module_names

    def test_network_graph_response(self, phase2_api_client):
        resp = phase2_api_client.get("/api/phase2/contagion/analyze")
        assert resp.status_code == 200
        data = resp.json()
        graph = data["network_graph"]
        assert graph["total_nodes"] == 2
        assert graph["total_edges"] == 1

    def test_transition_forecast_no_model(self, phase2_api_client):
        """If transition model is None, should return 503."""
        import api.dependencies as deps
        system = deps._system
        system.transition_model = None
        # Clear cached analysis
        deps._last_analysis = None

        resp = phase2_api_client.get("/api/phase2/transition-forecast")
        assert resp.status_code == 503
