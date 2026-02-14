"""
Tests for Phase 1 Foundation Hardening features.

Covers:
- ModuleSignal 'cautious' direction
- YAML config loading in ensemble
- Recalibration engine with cooldown/rollback/shadow mode
- SQLite signal and classification persistence
- Options surface activation
- Factor exposure OLS regression
- Nelson-Siegel yield curve fitting
- GARCH volatility classifier
- REGIME_VIX_PROFILES usage
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── ModuleSignal 'cautious' ─────────────────────────────────────


class TestModuleSignalCautious:
    """Verify 'cautious' signal type works without KeyError."""

    def test_cautious_direction_is_zero(self):
        from src.modules.base import ModuleSignal

        sig = ModuleSignal(
            signal="cautious",
            strength=0.5,
            confidence=0.8,
            explanation="test cautious",
            regime_context="test",
        )
        assert sig.direction == 0

    def test_all_direction_values(self):
        from src.modules.base import ModuleSignal

        expected = {"bullish": 1, "bearish": -1, "neutral": 0, "cautious": 0}
        for signal_name, expected_dir in expected.items():
            sig = ModuleSignal(
                signal=signal_name,
                strength=0.5,
                confidence=0.8,
                explanation=f"test {signal_name}",
                regime_context="test",
            )
            assert sig.direction == expected_dir, f"{signal_name} direction wrong"


# ─── YAML Config Loading ─────────────────────────────────────────


class TestEnsembleYAMLConfig:
    """Test that ensemble loads weights from model_params.yaml."""

    def test_default_weights_match_yaml(self):
        from src.regime_detection.ensemble import RegimeEnsemble

        ensemble = RegimeEnsemble()
        # model_params.yaml has hmm:0.35, rf:0.25, corr:0.20, vol:0.20
        assert ensemble.DEFAULT_WEIGHTS["hmm"] == 0.35
        assert ensemble.DEFAULT_WEIGHTS["ml"] == 0.25
        assert ensemble.DEFAULT_WEIGHTS["correlation"] == 0.20
        assert ensemble.DEFAULT_WEIGHTS["volatility"] == 0.20

    def test_weights_load_from_yaml_file(self):
        from src.regime_detection.ensemble import RegimeEnsemble

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
        )
        if os.path.exists(os.path.join(config_path, "model_params.yaml")):
            ensemble = RegimeEnsemble(config_path=config_path)
            # Should have loaded weights
            assert sum(ensemble.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_load_yaml_config_static(self):
        from src.regime_detection.ensemble import RegimeEnsemble

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
        )
        yaml_path = os.path.join(config_path, "model_params.yaml")
        if os.path.exists(yaml_path):
            cfg = RegimeEnsemble._load_yaml_config(config_path)
            assert cfg is not None
            assert "ensemble" in cfg


# ─── Recalibration Engine ────────────────────────────────────────


class TestRecalibrationEngine:
    """Test recalibration cooldown, rollback, shadow mode."""

    def _create_meta_learner(self):
        from src.meta_learning.meta_learner import MetaLearner

        ml = MetaLearner()
        ml._last_recalibration = None  # Reset cooldown
        return ml

    def test_cooldown_prevents_rapid_recalibration(self):
        from src.meta_learning.recalibration import (
            RecalibrationDecision,
            RecalibrationReason,
        )

        ml = self._create_meta_learner()
        ensemble = MagicMock()
        ensemble.weights = {"hmm": 0.35, "ml": 0.25, "correlation": 0.20, "volatility": 0.20}
        ensemble.update_weights = MagicMock()

        decision = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.HIGH_ERROR_RATE],
            severity=0.7,
            timestamp=datetime.now(),
            recommendations=["Retrain ensemble"],
        )

        # First call should succeed
        result1 = ml.execute_recalibration(decision, ensemble=ensemble)
        assert result1 is True

        # Second call within cooldown should be skipped (returns False)
        result2 = ml.execute_recalibration(decision, ensemble=ensemble)
        assert result2 is False

    def test_normalize_weights(self):
        ml = self._create_meta_learner()
        ml.adaptive_weights = {"hmm": 0.5, "ml": 0.3, "correlation": 0.1, "volatility": 0.1}
        ml._normalize_weights()
        assert sum(ml.adaptive_weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_rollback_restores_snapshot(self):
        ml = self._create_meta_learner()
        original_weights = {"hmm": 0.35, "ml": 0.25, "correlation": 0.20, "volatility": 0.20}
        ml.adaptive_weights = {"hmm": 0.50, "ml": 0.20, "correlation": 0.15, "volatility": 0.15}
        ml._pre_recal_snapshot = {
            "weights": original_weights.copy(),
            "timestamp": datetime.now(),
            "accuracy": 0.8,
            "recalibration_count": 0,
        }

        ml._rollback_recalibration()
        assert ml.adaptive_weights == original_weights


# ─── SQLite Persistence ──────────────────────────────────────────


class TestSQLitePersistence:
    """Test signal and classification history storage."""

    def test_save_and_load_module_signal(self):
        from src.data_pipeline.storage import DatabaseStorage

        db_path = os.path.join(tempfile.gettempdir(), f"amrcais_test_{os.getpid()}_sig.db")
        try:
            storage = DatabaseStorage(db_path)

            storage.save_module_signal(
                module_name="yield_curve",
                signal="bullish",
                strength=0.8,
                confidence=0.9,
                explanation="test signal",
                regime_context="risk_on",
                regime_id=1,
            )

            df = storage.load_module_signals(module_name="yield_curve")
            assert len(df) >= 1
            assert df.iloc[0]["Signal"] == "bullish"
            assert df.iloc[0]["Strength"] == pytest.approx(0.8)
        finally:
            storage.engine.dispose()
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass

    def test_save_and_load_classification(self):
        from src.data_pipeline.storage import DatabaseStorage

        db_path = os.path.join(tempfile.gettempdir(), f"amrcais_test_{os.getpid()}_cls.db")
        try:
            storage = DatabaseStorage(db_path)

            storage.save_classification(
                regime=1,
                confidence=0.85,
                disagreement=0.2,
                individual_predictions={"hmm": 1, "ml": 1},
                market_state={"vix": 15.0},
            )

            df = storage.load_classifications(limit=10)
            assert len(df) >= 1
            assert df[0]["regime"] == 1
            assert df[0]["confidence"] == pytest.approx(0.85)
        finally:
            storage.engine.dispose()
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass


# ─── Options Surface ─────────────────────────────────────────────


class TestOptionsSurfaceActivation:
    """Options surface analyze() now computes skew and term structure."""

    def test_analyze_returns_skew_and_surface(self):
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor

        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.8)

        # Create data with VIX and SPX columns
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "VIX": np.random.uniform(12, 25, 100),
                "SPX": np.cumsum(np.random.randn(100)) + 4500,
            },
            index=dates,
        )

        result = monitor.analyze(data)
        assert "signal" in result
        # Should have computed surface data
        assert result["signal"] is not None

    def test_analyze_with_insufficient_data(self):
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor

        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.8)

        # Edge case: very short data
        data = pd.DataFrame({"VIX": [15.0, 16.0]})
        result = monitor.analyze(data)
        assert "signal" in result


# ─── Factor Exposure OLS ─────────────────────────────────────────


class TestFactorExposureOLS:
    """Test rolling OLS regression in factor exposure analyzer."""

    def _create_factor_data(self, n: int = 100) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        spx = np.cumsum(np.random.randn(n) * 0.01)
        momentum = spx * 0.5 + np.random.randn(n) * 0.005
        value = -spx * 0.3 + np.random.randn(n) * 0.005
        return pd.DataFrame(
            {
                "SPX_returns": spx,
                "momentum": momentum,
                "value": value,
                "quality": np.random.randn(n) * 0.01,
            },
            index=dates,
        )

    def test_rolling_ols_beta_computed(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.8)

        data = self._create_factor_data(100)
        result = analyzer.analyze(data)

        assert "ols_betas" in result
        # Momentum should have positive beta (correlated with SPX)
        if "momentum" in result.get("ols_betas", {}):
            assert result["ols_betas"]["momentum"] > 0

    def test_detect_factor_rotation_called(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.8)

        data = self._create_factor_data(100)
        result = analyzer.analyze(data)

        assert "rotation" in result

    def test_historical_stats_populated(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.8)

        data = self._create_factor_data(100)
        result = analyzer.analyze(data)

        assert "historical_stats" in result
        assert len(result["historical_stats"]) > 0

    def test_compute_rolling_ols_beta_method(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        data = self._create_factor_data(100)
        beta = analyzer._compute_rolling_ols_beta(data, "SPX_returns", "momentum")
        assert beta is not None
        assert isinstance(beta, float)

    def test_insufficient_data_returns_none(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        data = self._create_factor_data(10)
        beta = analyzer._compute_rolling_ols_beta(data, "SPX_returns", "momentum")
        # Should return None with only 10 obs (min is 40)
        assert beta is None

    def test_analyze_factor_accepts_beta(self):
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer

        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.8)

        result = analyzer.analyze_factor(
            factor_name="momentum",
            return_value=0.02,
            z_score=1.5,
            beta=0.45,
        )
        assert result["beta"] == 0.45


# ─── Nelson-Siegel Yield Curve ────────────────────────────────────


class TestNelsonSiegel:
    """Test Nelson-Siegel fitting in yield curve analyzer."""

    def _sample_yields(self) -> dict:
        return {
            "3M": 5.30,
            "6M": 5.20,
            "1Y": 4.90,
            "2Y": 4.60,
            "5Y": 4.30,
            "7Y": 4.35,
            "10Y": 4.40,
            "20Y": 4.60,
            "30Y": 4.65,
        }

    def test_fit_nelson_siegel_returns_params(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        params = analyzer.fit_nelson_siegel(self._sample_yields())

        assert params is not None
        assert "level" in params
        assert "slope" in params
        assert "curvature" in params
        assert "tau" in params
        assert "fit_rmse" in params
        assert params["fit_rmse"] < 0.5  # Reasonable fit

    def test_fit_nelson_siegel_insufficient_data(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        params = analyzer.fit_nelson_siegel({"2Y": 4.6, "10Y": 4.4})
        # Only 2 tenors — not enough
        assert params is None

    def test_analyze_includes_nelson_siegel(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.8)

        yields = self._sample_yields()
        col_map = {
            "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2",
            "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20",
            "30Y": "DGS30",
        }
        data_dict = {col_map[k]: [v] for k, v in yields.items()}
        data = pd.DataFrame(data_dict)

        result = analyzer.analyze(data)
        assert "nelson_siegel" in result
        assert result["nelson_siegel"] is not None

    def test_analyze_includes_forward_rates(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.8)

        yields = self._sample_yields()
        col_map = {
            "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1", "2Y": "DGS2",
            "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10", "20Y": "DGS20",
            "30Y": "DGS30",
        }
        data_dict = {col_map[k]: [v] for k, v in yields.items()}
        data = pd.DataFrame(data_dict)

        result = analyzer.analyze(data)
        assert "forward_rates" in result

    def test_ns_params_history_grows(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.8)

        yields = self._sample_yields()
        col_map = {
            "3M": "DGS3MO", "1Y": "DGS1", "2Y": "DGS2",
            "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30",
        }
        data_dict = {col_map[k]: [v] for k, v in yields.items() if k in col_map}
        data = pd.DataFrame(data_dict)

        analyzer.analyze(data)
        analyzer.analyze(data)
        assert len(analyzer._ns_params_history) == 2

    def test_dynamics_with_ns(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.8)

        yields = self._sample_yields()
        col_map = {
            "3M": "DGS3MO", "1Y": "DGS1", "2Y": "DGS2",
            "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30",
        }
        data_dict = {col_map[k]: [v] for k, v in yields.items() if k in col_map}
        data = pd.DataFrame(data_dict)

        analyzer.analyze(data)
        result = analyzer.analyze(data)
        dynamics = result["dynamics"]
        assert "ns_dynamics" in dynamics


# ─── GARCH Volatility Classifier ─────────────────────────────────


class TestGARCHVolatilityClassifier:
    """Test GARCH integration and realized vol."""

    def _sample_vix_data(self, n: int = 300) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        vix = np.abs(20 + np.cumsum(np.random.randn(n) * 0.5))
        spx = np.cumsum(np.random.randn(n) * 1.5) + 4500
        return pd.DataFrame({"VIX": vix, "SPX": spx}, index=dates)

    def test_garch_fitting_with_arch_package(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier(use_garch=True)
        data = self._sample_vix_data()
        classifier.fit(data)
        # GARCH should have been attempted
        assert classifier.is_fitted

    def test_garch_disabled(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier(use_garch=False)
        data = self._sample_vix_data()
        classifier.fit(data)
        assert classifier._garch_model is None

    def test_realized_vol_calculated(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier()
        data = self._sample_vix_data()
        classifier.fit(data)
        assert classifier._realized_vol is not None
        assert len(classifier._realized_vol) > 0

    def test_predict_includes_garch_and_realized_vol_metadata(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier()
        data = self._sample_vix_data()
        classifier.fit(data)
        result = classifier.predict(data)
        assert "realized_vol" in result.metadata

    def test_regime_vix_profiles_boost_confidence(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier()
        data = self._sample_vix_data()
        classifier.fit(data)

        # Low VIX should classify as risk-on (regime 1) with good conf
        result = classifier.predict(12.0)
        assert result.regime in (1, 4)
        assert result.confidence >= 0.3

    def test_feature_importance_dynamic(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier()
        data = self._sample_vix_data()
        classifier.fit(data)
        fi = classifier.get_feature_importance()
        assert "realized_vol" in fi
        total = sum(fi.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_high_vix_classifies_risk_off(self):
        from src.regime_detection.volatility_classifier import (
            VolatilityRegimeClassifier,
        )

        classifier = VolatilityRegimeClassifier()
        data = self._sample_vix_data()
        classifier.fit(data)
        result = classifier.predict(45.0)
        assert result.regime == 2  # Risk-Off Crisis


# ─── Spline Cache ────────────────────────────────────────────────


class TestSplineCache:
    """Test that spline cache is populated during analysis."""

    def test_spline_cache_populated(self):
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.8)

        yields = {
            "3M": 5.30, "1Y": 4.90, "2Y": 4.60,
            "5Y": 4.30, "10Y": 4.40, "30Y": 4.65,
        }
        col_map = {
            "3M": "DGS3MO", "1Y": "DGS1", "2Y": "DGS2",
            "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30",
        }
        data_dict = {col_map[k]: [v] for k, v in yields.items()}
        data = pd.DataFrame(data_dict)

        analyzer.analyze(data)
        assert "spline" in analyzer._spline_cache
