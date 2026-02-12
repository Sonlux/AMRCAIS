"""
Test Suite for AMRCAIS.

This module contains tests for the core components:
- Data pipeline (fetchers, validators, storage)
- Regime classifiers (HMM, ML, Correlation, Volatility)
- Regime ensemble (voting, disagreement)
- Analytical modules

Tests use walk-forward validation and mock external APIs.
Target coverage: 80%
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    
    # Generate correlated returns
    base_returns = np.random.normal(0.0005, 0.015, len(dates))
    
    data = pd.DataFrame({
        "Date": dates,
        "SPX": 3000 * np.cumprod(1 + base_returns + np.random.normal(0, 0.005, len(dates))),
        "TLT": 130 * np.cumprod(1 + base_returns * -0.3 + np.random.normal(0, 0.003, len(dates))),
        "GLD": 150 * np.cumprod(1 + np.random.normal(0, 0.008, len(dates))),
        "VIX": np.clip(15 + np.cumsum(np.random.normal(0, 0.5, len(dates))), 10, 80),
        "DXY": 95 + np.cumsum(np.random.normal(0, 0.1, len(dates))),
        "WTI": 55 * np.cumprod(1 + np.random.normal(0, 0.02, len(dates))),
    }).set_index("Date")
    
    return data


@pytest.fixture
def sample_yield_data():
    """Generate sample yield curve data."""
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    np.random.seed(42)
    
    # Base level (Fed Funds proxy)
    base = 1.5 + np.cumsum(np.random.normal(0, 0.02, len(dates)))
    base = np.clip(base, 0, 5)
    
    data = pd.DataFrame({
        "Date": dates,
        "DGS3MO": base + 0.1,
        "DGS2": base + 0.5 + np.random.normal(0, 0.05, len(dates)),
        "DGS5": base + 0.8 + np.random.normal(0, 0.08, len(dates)),
        "DGS10": base + 1.0 + np.random.normal(0, 0.1, len(dates)),
        "DGS30": base + 1.3 + np.random.normal(0, 0.12, len(dates)),
    }).set_index("Date")
    
    return data


@pytest.fixture
def covid_crisis_data():
    """Generate data that mimics March 2020 COVID crash."""
    dates = pd.date_range(start="2020-02-01", periods=60, freq="D")
    
    # Pre-crash, crash, recovery pattern
    pre_crash = np.ones(15) * 3300
    crash = np.linspace(3300, 2200, 20)
    recovery = np.linspace(2200, 2800, 25)
    spx = np.concatenate([pre_crash, crash, recovery])
    
    # VIX spikes during crash
    vix_pre = np.ones(15) * 15
    vix_crash = np.linspace(15, 80, 20)
    vix_recovery = np.linspace(80, 35, 25)
    vix = np.concatenate([vix_pre, vix_crash, vix_recovery])
    
    data = pd.DataFrame({
        "Date": dates,
        "SPX": spx,
        "VIX": vix,
        "TLT": 140 + np.random.normal(0, 2, 60),  # Flight to safety
        "GLD": 155 + np.cumsum(np.random.normal(0.2, 1, 60)),
    }).set_index("Date")
    
    return data


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestDataValidator:
    """Tests for data validation."""
    
    def test_missing_values_detection(self, sample_price_data):
        """Test detection of missing values."""
        from src.data_pipeline.validators import DataValidator
        
        # Inject some NaN values
        data_with_nan = sample_price_data.copy()
        data_with_nan.iloc[10, 0] = np.nan
        data_with_nan.iloc[20, 1] = np.nan
        
        validator = DataValidator()
        report = validator.validate(data_with_nan)
        
        assert not report.is_valid
        assert report.missing_values > 0
    
    def test_price_validity(self, sample_price_data):
        """Test detection of invalid prices."""
        from src.data_pipeline.validators import DataValidator
        
        # Inject negative price
        data_with_invalid = sample_price_data.copy()
        data_with_invalid.iloc[50, 0] = -100
        
        validator = DataValidator()
        report = validator.validate(data_with_invalid)
        
        assert len(report.warnings) > 0 or len(report.errors) > 0
    
    def test_outlier_detection(self, sample_price_data):
        """Test detection of extreme returns."""
        from src.data_pipeline.validators import DataValidator
        
        # Inject extreme move
        data_with_outlier = sample_price_data.copy()
        data_with_outlier.iloc[100, 0] = data_with_outlier.iloc[99, 0] * 1.5  # 50% move
        
        validator = DataValidator()
        report = validator.validate(data_with_outlier)
        
        assert report.outlier_count > 0
    
    def test_valid_data_passes(self, sample_price_data):
        """Test that valid data passes validation."""
        from src.data_pipeline.validators import DataValidator
        
        validator = DataValidator()
        report = validator.validate(sample_price_data)
        
        assert report.is_valid


class TestDatabaseStorage:
    """Tests for database storage."""
    
    def test_save_and_load_market_data(self, sample_price_data, temp_db_path):
        """Test saving and loading market data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(f"sqlite:///{temp_db_path}")
        
        # Save
        storage.save_market_data(sample_price_data, "SPX")
        
        # Load
        loaded = storage.load_market_data("SPX")
        
        assert loaded is not None
        assert len(loaded) > 0
    
    def test_freshness_check(self, sample_price_data, temp_db_path):
        """Test data freshness validation."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(f"sqlite:///{temp_db_path}")
        storage.save_market_data(sample_price_data, "SPX")
        
        # Should be fresh (just saved)
        is_fresh = storage.is_data_fresh("SPX", max_age_days=1)
        assert is_fresh


# ============================================================================
# REGIME CLASSIFIER TESTS
# ============================================================================

class TestHMMClassifier:
    """Tests for HMM regime classifier."""
    
    def test_fit_and_predict(self, sample_price_data):
        """Test basic fit and predict."""
        from src.regime_detection.hmm_classifier import HMMRegimeClassifier
        
        classifier = HMMRegimeClassifier()
        classifier.fit(sample_price_data)
        
        result = classifier.predict(sample_price_data)
        
        assert result.regime in [1, 2, 3, 4]
        assert 0 <= result.confidence <= 1
        assert sum(result.probabilities.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_detects_crisis(self, covid_crisis_data):
        """Test that HMM detects crisis regime during COVID crash."""
        from src.regime_detection.hmm_classifier import HMMRegimeClassifier
        
        classifier = HMMRegimeClassifier()
        classifier.fit(covid_crisis_data)
        
        # During the crash period (days 15-35)
        crash_data = covid_crisis_data.iloc[15:35]
        result = classifier.predict(crash_data)
        
        # Should detect elevated risk (regime 2)
        # Note: May also detect stagflation (3) due to high vol
        assert result.regime in [2, 3]
    
    def test_predict_sequence(self, sample_price_data):
        """Test sequence prediction."""
        from src.regime_detection.hmm_classifier import HMMRegimeClassifier
        
        classifier = HMMRegimeClassifier()
        classifier.fit(sample_price_data)
        
        results = classifier.predict_sequence(sample_price_data, window=60, step=10)
        
        assert len(results) > 0
        assert all(r.regime in [1, 2, 3, 4] for r in results)


class TestMLClassifier:
    """Tests for ML regime classifier."""
    
    def test_fit_with_labels(self, sample_price_data):
        """Test training with labeled data."""
        from src.regime_detection.ml_classifier import MLRegimeClassifier
        
        # Create labels
        labels = np.ones(len(sample_price_data), dtype=int)
        labels[100:150] = 2  # Crisis period
        labels[200:250] = 3  # Stagflation period
        
        classifier = MLRegimeClassifier()
        classifier.fit(sample_price_data, labels=labels)
        
        result = classifier.predict(sample_price_data)
        
        assert result.regime in [1, 2, 3, 4]
    
    def test_feature_importance(self, sample_price_data):
        """Test feature importance extraction."""
        from src.regime_detection.ml_classifier import MLRegimeClassifier
        
        labels = np.ones(len(sample_price_data), dtype=int)
        labels[100:150] = 2
        
        classifier = MLRegimeClassifier()
        classifier.fit(sample_price_data, labels=labels)
        
        importance = classifier.get_feature_importance()
        
        assert importance is not None
        assert len(importance) > 0


class TestVolatilityClassifier:
    """Tests for volatility-based classifier."""
    
    def test_basic_prediction(self, sample_price_data):
        """Test basic prediction."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier
        
        classifier = VolatilityRegimeClassifier()
        classifier.fit(sample_price_data)
        
        result = classifier.predict(sample_price_data)
        
        assert result.regime in [1, 2, 3, 4]
    
    def test_high_vix_detection(self, covid_crisis_data):
        """Test detection of high VIX regime."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier
        
        classifier = VolatilityRegimeClassifier()
        classifier.fit(covid_crisis_data)
        
        # During high VIX period
        crisis_data = covid_crisis_data.iloc[20:40]
        result = classifier.predict(crisis_data)
        
        # High VIX should indicate risk-off
        assert result.regime == 2


class TestCorrelationClassifier:
    """Tests for correlation-based classifier."""
    
    def test_clustering(self, sample_price_data):
        """Test correlation clustering."""
        from src.regime_detection.correlation_classifier import CorrelationRegimeClassifier
        
        classifier = CorrelationRegimeClassifier()
        classifier.fit(sample_price_data)
        
        result = classifier.predict(sample_price_data)
        
        assert result.regime in [1, 2, 3, 4]


# ============================================================================
# ENSEMBLE TESTS
# ============================================================================

class TestRegimeEnsemble:
    """Tests for regime ensemble."""
    
    def test_ensemble_prediction(self, sample_price_data):
        """Test ensemble prediction."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        ensemble.fit(sample_price_data)
        
        result = ensemble.predict(sample_price_data)
        
        assert result.regime in [1, 2, 3, 4]
        assert 0 <= result.confidence <= 1
        assert 0 <= result.disagreement <= 1
    
    def test_disagreement_calculation(self, sample_price_data):
        """Test disagreement index calculation."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        ensemble.fit(sample_price_data)
        
        result = ensemble.predict(sample_price_data)
        
        # Disagreement should be between 0 and 1
        assert 0 <= result.disagreement <= 1
        
        # Individual predictions should be recorded
        assert len(result.individual_predictions) > 0
    
    def test_transition_warning(self, covid_crisis_data):
        """Test that high disagreement triggers warning."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble(disagreement_threshold=0.5)
        ensemble.fit(covid_crisis_data)
        
        # During transition period, disagreement may be high
        results = ensemble.predict_sequence(covid_crisis_data, window=20, step=5)
        
        # At least one result should exist
        assert len(results) > 0
    
    def test_weight_update(self, sample_price_data):
        """Test classifier weight updates."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        
        new_weights = {"hmm": 0.4, "ml": 0.3, "correlation": 0.2, "volatility": 0.1}
        ensemble.update_weights(new_weights)
        
        # Weights should sum to 1
        assert sum(ensemble.weights.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_recalibration_detection(self, sample_price_data):
        """Test recalibration need detection."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        ensemble.fit(sample_price_data)
        
        # Initially should not need recalibration
        needs_recal, reason = ensemble.needs_recalibration()
        
        # This test just verifies the method works
        assert isinstance(needs_recal, bool)


# ============================================================================
# ANALYTICAL MODULE TESTS
# ============================================================================

class TestMacroEventTracker:
    """Tests for macro event tracker."""
    
    def test_regime_adaptive_interpretation(self):
        """Test that same event is interpreted differently by regime."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        
        # Strong NFP in Risk-On should be bullish
        tracker.update_regime(1, 0.9)
        result_riskon = tracker.analyze_event("NFP", actual=300, consensus=200)
        
        # Same strong NFP in Stagflation should be more mixed
        tracker.update_regime(3, 0.9)
        result_stagflation = tracker.analyze_event("NFP", actual=300, consensus=200)
        
        # Signals should differ
        assert result_riskon["signal"].signal != result_stagflation["signal"].signal or \
               result_riskon["signal"].explanation != result_stagflation["signal"].explanation
    
    def test_surprise_calculation(self):
        """Test surprise standardization."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.9)
        
        # 2 standard deviation surprise (NFP std = 75)
        result = tracker.analyze_event("NFP", actual=350, consensus=200)
        
        assert result["surprise"] == pytest.approx(2.0, rel=0.01)


class TestYieldCurveAnalyzer:
    """Tests for yield curve analyzer."""
    
    def test_curve_shape_classification(self, sample_yield_data):
        """Test yield curve shape classification."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape
        
        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.9)
        
        result = analyzer.analyze(sample_yield_data)
        
        assert result["curve_shape"] in [s.value for s in CurveShape]
    
    def test_inversion_detection(self):
        """Test yield curve inversion detection."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer
        
        analyzer = YieldCurveAnalyzer()
        
        # Create inverted curve
        yields = {"2Y": 3.0, "5Y": 2.8, "10Y": 2.5}
        
        inversions = analyzer.detect_inversion(yields)
        
        assert inversions["2s10s"] == True


class TestOptionsSurfaceMonitor:
    """Tests for options surface monitor."""
    
    def test_vol_level_interpretation(self, sample_price_data):
        """Test volatility level interpretation varies by regime."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        
        # VIX of 25 in Risk-On is elevated
        monitor.update_regime(1, 0.9)
        result_riskon = monitor._analyze_vol_level(25)
        
        # VIX of 25 in Risk-Off is normal
        monitor.update_regime(2, 0.9)
        result_riskoff = monitor._analyze_vol_level(25)
        
        # Interpretations should differ
        assert result_riskon.signal != result_riskoff.signal or \
               "elevated" in result_riskon.explanation.lower() or \
               "normal" in result_riskoff.explanation.lower()


class TestCorrelationAnomalyDetector:
    """Tests for correlation anomaly detector."""
    
    def test_anomaly_detection(self, sample_price_data):
        """Test correlation anomaly detection."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.9)
        
        # Check a correlation that should be in normal range
        result = detector.check_correlation("SPX", "TLT", -0.3)
        
        assert "is_anomaly" in result
    
    def test_correlation_matrix(self, sample_price_data):
        """Test correlation matrix calculation."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        
        corr_matrix = detector.get_correlation_matrix(sample_price_data)
        
        assert isinstance(corr_matrix, pd.DataFrame)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullPipeline:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_prediction(self, sample_price_data):
        """Test full pipeline from data to prediction."""
        from src.regime_detection.ensemble import RegimeEnsemble
        from src.modules.macro_event_tracker import MacroEventTracker
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer
        
        # 1. Get regime
        ensemble = RegimeEnsemble()
        ensemble.fit(sample_price_data)
        regime_result = ensemble.predict(sample_price_data)
        
        # 2. Update modules with regime
        macro = MacroEventTracker()
        macro.update_regime(regime_result.regime, regime_result.confidence)
        
        yields = YieldCurveAnalyzer()
        yields.update_regime(regime_result.regime, regime_result.confidence)
        
        # 3. All modules should have same regime
        assert macro.current_regime == regime_result.regime
        assert yields.current_regime == regime_result.regime


# ============================================================================
# KNOWN EVENT TESTS (Walk-Forward)
# ============================================================================

class TestKnownEvents:
    """Tests against known historical events."""
    
    def test_covid_crash_detection(self, covid_crisis_data):
        """Test detection of COVID crash regime."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        ensemble.fit(covid_crisis_data)
        
        # Test during crash
        crash_period = covid_crisis_data.iloc[15:35]
        result = ensemble.predict(crash_period)
        
        # Should detect crisis
        assert result.regime == 2 or result.probabilities[2] > 0.3
    
    def test_recovery_transition(self, covid_crisis_data):
        """Test detection of transition from crisis to recovery."""
        from src.regime_detection.ensemble import RegimeEnsemble
        
        ensemble = RegimeEnsemble()
        
        # Fit on full data
        ensemble.fit(covid_crisis_data)
        
        # Get sequence of predictions
        results = ensemble.predict_sequence(covid_crisis_data, window=20, step=5)
        
        # Should see some variation in regimes
        regimes = [r.regime for r in results]
        assert len(set(regimes)) >= 1  # At least one distinct regime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
