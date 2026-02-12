"""
Test Suite for AMRCAIS Data Pipeline and Main Class.

Tests for:
- DataPipeline: Config loading, data fetching (mocked), calculate_returns, prepare_regime_features
- AMRCAIS: Initialization, analysis flow, state management
- FactorExposureAnalyzer: Full factor analysis coverage
- CorrelationAnomalyDetector: Anomaly detection coverage

Coverage target: 80%+ for src/main.py, src/data_pipeline/pipeline.py,
    src/modules/factor_exposure_analyzer.py, src/modules/correlation_anomaly_detector.py
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path
import tempfile
import shutil
import os


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
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
def sample_price_dict():
    """Generate sample data as dictionary (for pipeline methods)."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="D")
    base_returns = np.random.normal(0.0005, 0.015, len(dates))
    
    return {
        "SPX": pd.DataFrame({
            "Close": 3000 * np.cumprod(1 + base_returns + np.random.normal(0, 0.005, len(dates))),
        }, index=dates),
        "TLT": pd.DataFrame({
            "Close": 130 * np.cumprod(1 + base_returns * -0.3 + np.random.normal(0, 0.003, len(dates))),
        }, index=dates),
        "GLD": pd.DataFrame({
            "Close": 150 * np.cumprod(1 + np.random.normal(0, 0.008, len(dates))),
        }, index=dates),
        "VIX": pd.DataFrame({
            "Close": np.clip(15 + np.cumsum(np.random.normal(0, 0.5, len(dates))), 10, 80),
        }, index=dates),
    }


@pytest.fixture
def sample_factor_data():
    """Generate sample factor return data."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "momentum": np.random.normal(0.003, 0.01, 100),
        "value": np.random.normal(-0.001, 0.008, 100),
        "quality": np.random.normal(0.002, 0.007, 100),
        "size": np.random.normal(0.001, 0.009, 100),
        "volatility": np.random.normal(-0.001, 0.006, 100),
        "growth": np.random.normal(0.002, 0.01, 100),
    }, index=dates)


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for database."""
    tmpdir = tempfile.mkdtemp()
    db = os.path.join(tmpdir, "test.db")
    yield db
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestDataPipelineInit:
    """Tests for DataPipeline initialization."""

    def test_init_no_config(self, temp_db_path):
        """Test pipeline init without config file."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(
            config_path="nonexistent.yaml",
            db_path=temp_db_path,
        )
        assert pipeline.config == {}
        assert pipeline.cache_max_age_days == 7

    def test_init_with_config(self, temp_db_path):
        """Test pipeline init with real config."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(
            config_path="config/data_sources.yaml",
            db_path=temp_db_path,
        )
        # If config file exists, should load it
        assert isinstance(pipeline.config, dict)

    def test_init_default_config_search(self, temp_db_path):
        """Test pipeline searches default config paths."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(
            config_path=None,
            db_path=temp_db_path,
        )
        assert isinstance(pipeline.config, dict)

    def test_lazy_fetcher_initialization(self, temp_db_path):
        """Test that fetchers are lazily initialized."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        assert pipeline._fred_fetcher is None
        assert pipeline._yf_fetcher is None
        assert pipeline._av_fetcher is None
        
        # Accessing property should initialize
        yf = pipeline.yf_fetcher
        assert yf is not None
        assert pipeline._yf_fetcher is not None


class TestDataPipelineCalculations:
    """Tests for DataPipeline calculation methods."""

    def test_calculate_returns_log(self, temp_db_path, sample_price_dict):
        """Test log returns calculation."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        returns = pipeline.calculate_returns(sample_price_dict, method="log")
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) > 0
        assert set(returns.columns) == set(sample_price_dict.keys())
        # Log returns should be close to zero on average for short periods
        assert all(abs(returns.mean()) < 0.1)

    def test_calculate_returns_simple(self, temp_db_path, sample_price_dict):
        """Test simple returns calculation."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        returns = pipeline.calculate_returns(sample_price_dict, method="simple")
        
        assert isinstance(returns, pd.DataFrame)
        assert len(returns) > 0

    def test_calculate_returns_no_close(self, temp_db_path):
        """Test returns with missing Close column handled gracefully."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        data = {
            "ASSET_A": pd.DataFrame({"Price": [100, 101, 102]}),
        }
        returns = pipeline.calculate_returns(data)
        assert len(returns.columns) == 0  # No Close column to process

    def test_prepare_regime_features(self, temp_db_path, sample_price_dict):
        """Test regime feature preparation."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        features = pipeline.prepare_regime_features(sample_price_dict, lookback=20)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        # Should have return columns
        assert any("return" in c for c in features.columns)
        # Should have volatility columns
        assert any("vol" in c for c in features.columns)

    def test_prepare_regime_features_with_correlations(self, temp_db_path):
        """Test feature preparation with SPX-TLT and SPX-GLD correlations."""
        from src.data_pipeline.pipeline import DataPipeline
        
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        data = {
            "SPX": pd.DataFrame({"Close": 3000 * np.cumprod(1 + np.random.normal(0, 0.01, 200))}, index=dates),
            "TLT": pd.DataFrame({"Close": 130 * np.cumprod(1 + np.random.normal(0, 0.005, 200))}, index=dates),
            "GLD": pd.DataFrame({"Close": 150 * np.cumprod(1 + np.random.normal(0, 0.008, 200))}, index=dates),
            "VIX": pd.DataFrame({"Close": np.clip(20 + np.cumsum(np.random.normal(0, 0.3, 200)), 10, 60)}, index=dates),
        }
        
        pipeline = DataPipeline(db_path=temp_db_path)
        features = pipeline.prepare_regime_features(data, lookback=20)
        
        # Should include correlation features
        corr_cols = [c for c in features.columns if "corr" in c]
        assert len(corr_cols) >= 1  # At least SPX_TLT_corr
        
        # Should include momentum
        mom_cols = [c for c in features.columns if "mom" in c]
        assert len(mom_cols) >= 1

    def test_get_data_summary(self, temp_db_path):
        """Test data summary retrieval."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        summary = pipeline.get_data_summary()
        
        assert "available_assets" in summary
        assert "asset_details" in summary
        assert isinstance(summary["available_assets"], list)


class TestDataPipelineFetching:
    """Tests for DataPipeline fetching with mocked external APIs."""

    @patch("src.data_pipeline.pipeline.YFinanceFetcher")
    def test_fetch_single_asset_yfinance(self, mock_yf_cls, temp_db_path):
        """Test fetching from yfinance."""
        from src.data_pipeline.pipeline import DataPipeline
        
        # Create mock data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        mock_df = pd.DataFrame({"Close": np.random.uniform(100, 200, 100)}, index=dates)
        
        mock_fetcher = Mock()
        mock_fetcher.fetch.return_value = mock_df
        mock_yf_cls.return_value = mock_fetcher
        
        pipeline = DataPipeline(db_path=temp_db_path)
        pipeline._yf_fetcher = mock_fetcher
        
        result = pipeline._fetch_single_asset(
            "SPX", datetime(2023, 1, 1), datetime(2023, 4, 10), use_cache=False
        )
        
        assert result is not None
        assert len(result) == 100

    def test_fetch_single_asset_cache_miss(self, temp_db_path):
        """Test fetch when cache misses and API fails."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        
        # Mock all fetchers to fail
        mock_yf = Mock()
        mock_yf.fetch.side_effect = Exception("API error")
        pipeline._yf_fetcher = mock_yf
        
        result = pipeline._fetch_single_asset(
            "UNKNOWN_ASSET", datetime(2023, 1, 1), datetime(2023, 4, 10), use_cache=False
        )
        
        # Should return None or empty when all sources fail
        assert result is None or (hasattr(result, 'empty') and result.empty)

    def test_fetch_yield_curve_no_api_key(self, temp_db_path):
        """Test yield curve fetch without FRED API key."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        
        with patch.dict(os.environ, {}, clear=True):
            # Remove FRED_API_KEY if it exists
            os.environ.pop("FRED_API_KEY", None)
            result = pipeline.fetch_yield_curve()
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_fetch_macro_data_no_api_key(self, temp_db_path):
        """Test macro data fetch without FRED API key."""
        from src.data_pipeline.pipeline import DataPipeline
        
        pipeline = DataPipeline(db_path=temp_db_path)
        
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FRED_API_KEY", None)
            result = pipeline.fetch_macro_data(["CPI", "NFP"])
        
        assert result == {}


# ============================================================================
# AMRCAIS MAIN CLASS TESTS
# ============================================================================

class TestAMRCAISInit:
    """Tests for AMRCAIS class initialization."""

    def test_init_defaults(self):
        """Test AMRCAIS initialization with defaults."""
        from src.main import AMRCAIS
        
        system = AMRCAIS()
        assert system.config_path == "config"
        assert system.db_path == "data/amrcais.db"
        assert system._is_initialized is False
        assert system._current_regime is None
        assert system.modules == {}

    def test_init_custom_paths(self, temp_db_path):
        """Test AMRCAIS initialization with custom paths."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(config_path="custom_config", db_path=temp_db_path)
        assert system.config_path == "custom_config"
        assert system.db_path == temp_db_path

    def test_get_current_state_uninitialized(self):
        """Test state retrieval before initialization."""
        from src.main import AMRCAIS
        
        system = AMRCAIS()
        state = system.get_current_state()
        
        assert state["regime"] is None
        assert state["confidence"] == 0.0
        assert state["is_initialized"] is False


class TestAMRCAISAnalysis:
    """Tests for AMRCAIS analysis flow (mocked)."""

    def test_analyze_not_initialized(self):
        """Test analyze raises error when not initialized."""
        from src.main import AMRCAIS
        
        system = AMRCAIS()
        with pytest.raises(RuntimeError, match="not initialized"):
            system.analyze()

    def test_recalibrate_not_initialized(self):
        """Test recalibrate raises error when not initialized."""
        from src.main import AMRCAIS
        
        system = AMRCAIS()
        with pytest.raises(RuntimeError, match="not initialized"):
            system.recalibrate()

    def test_analyze_empty_data(self, temp_db_path):
        """Test analyze with empty data."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = pd.DataFrame()
        system.ensemble = Mock()
        system.meta_learner = Mock()
        system.modules = {}
        
        result = system.analyze()
        assert "error" in result

    def test_analyze_full_flow(self, sample_price_data, temp_db_path):
        """Test full analysis flow with mocked components."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = sample_price_data
        
        # Mock ensemble
        mock_regime_result = Mock()
        mock_regime_result.regime = 1
        mock_regime_result.regime_name = "Risk-On Growth"
        mock_regime_result.confidence = 0.85
        mock_regime_result.disagreement = 0.2
        mock_regime_result.transition_warning = False
        mock_regime_result.probabilities = {1: 0.7, 2: 0.1, 3: 0.1, 4: 0.1}
        mock_regime_result.individual_predictions = {"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1}
        
        mock_ensemble = Mock()
        mock_ensemble.predict.return_value = mock_regime_result
        system.ensemble = mock_ensemble
        
        # Mock meta-learner
        mock_recal_decision = Mock()
        mock_recal_decision.should_recalibrate = False
        mock_recal_decision.urgency_level = "NONE"
        mock_recal_decision.severity = 0.0
        mock_recal_decision.reasons = []
        
        mock_perf_metrics = Mock()
        mock_perf_metrics.to_dict.return_value = {"total_classifications": 10}
        
        mock_meta = Mock()
        mock_meta.check_recalibration_needed.return_value = mock_recal_decision
        mock_meta.generate_uncertainty_signal.return_value = {"signal": "low", "strength": 0.1}
        mock_meta.get_adaptive_weights.return_value = {"hmm": 0.3, "ml": 0.25, "correlation": 0.25, "volatility": 0.2}
        mock_meta.get_performance_metrics.return_value = mock_perf_metrics
        system.meta_learner = mock_meta
        
        # Mock modules
        mock_module = Mock()
        mock_module.analyze.return_value = {
            "signal": Mock(to_dict=lambda: {"signal": "bullish", "strength": 0.7}),
        }
        system.modules = {"macro": mock_module}
        
        result = system.analyze()
        
        assert "regime" in result
        assert result["regime"]["id"] == 1
        assert result["regime"]["name"] == "Risk-On Growth"
        assert result["regime"]["confidence"] == 0.85
        assert "modules" in result
        assert "meta" in result
        assert "summary" in result
        
        # Verify meta-learner was called
        mock_meta.log_classification.assert_called_once()
        mock_meta.check_recalibration_needed.assert_called_once()

    def test_analyze_module_failure_graceful(self, sample_price_data, temp_db_path):
        """Test that module failures don't crash analysis."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = sample_price_data
        
        # Mock ensemble
        mock_regime = Mock()
        mock_regime.regime = 1
        mock_regime.regime_name = "Risk-On Growth"
        mock_regime.confidence = 0.85
        mock_regime.disagreement = 0.2
        mock_regime.transition_warning = False
        mock_regime.probabilities = {1: 0.7}
        mock_regime.individual_predictions = {"hmm": 1}
        
        mock_ensemble = Mock()
        mock_ensemble.predict.return_value = mock_regime
        system.ensemble = mock_ensemble
        
        # Mock meta-learner
        mock_recal = Mock()
        mock_recal.should_recalibrate = False
        mock_recal.urgency_level = "NONE"
        mock_recal.severity = 0.0
        mock_recal.reasons = []
        
        mock_meta = Mock()
        mock_meta.check_recalibration_needed.return_value = mock_recal
        mock_meta.generate_uncertainty_signal.return_value = {"signal": "low"}
        mock_meta.get_adaptive_weights.return_value = {}
        mock_meta.get_performance_metrics.return_value = Mock(to_dict=lambda: {})
        system.meta_learner = mock_meta
        
        # Module that raises exception
        failing_module = Mock()
        failing_module.analyze.side_effect = Exception("Module crashed!")
        system.modules = {"failing_module": failing_module}
        
        # Should not raise
        result = system.analyze()
        assert "modules" in result
        assert "error" in result["modules"]["failing_module"]

    def test_generate_summary(self, temp_db_path):
        """Test summary generation."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        
        results = {
            "regime": {
                "name": "Risk-On Growth",
                "confidence": 0.85,
                "transition_warning": False,
            },
            "modules": {
                "macro": {"signal": {"signal": "bullish", "strength": 0.7}},
                "yield": {"signal": {"signal": "bearish", "strength": 0.5}},
                "options": {"signal": {"signal": "bullish", "strength": 0.6}},
            },
        }
        
        summary = system._generate_summary(results)
        assert summary["headline"] == "Current Regime: Risk-On Growth"
        assert summary["confidence_level"] == "High"
        assert summary["stability"] == "Stable"
        assert summary["overall_bias"] == "Bullish"
        assert summary["signal_counts"]["bullish"] == 2
        assert summary["signal_counts"]["bearish"] == 1

    def test_generate_summary_low_confidence(self, temp_db_path):
        """Test summary with low confidence."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        
        results = {
            "regime": {
                "name": "Stagflation",
                "confidence": 0.35,
                "transition_warning": True,
            },
            "modules": {},
        }
        
        summary = system._generate_summary(results)
        assert summary["confidence_level"] == "Low"
        assert summary["stability"] == "Unstable"

    def test_recalibrate(self, sample_price_data, temp_db_path):
        """Test recalibration."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = sample_price_data
        
        mock_ensemble = Mock()
        system.ensemble = mock_ensemble
        
        system.recalibrate()
        mock_ensemble.recalibrate.assert_called_once_with(sample_price_data)

    def test_recalibrate_with_new_data(self, sample_price_data, temp_db_path):
        """Test recalibration with new data."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = pd.DataFrame()
        
        mock_ensemble = Mock()
        system.ensemble = mock_ensemble
        
        system.recalibrate(new_data=sample_price_data)
        mock_ensemble.recalibrate.assert_called_once_with(sample_price_data)

    def test_analyze_without_metalearner(self, sample_price_data, temp_db_path):
        """Test analyze fallback when meta_learner is None."""
        from src.main import AMRCAIS
        
        system = AMRCAIS(db_path=temp_db_path)
        system._is_initialized = True
        system.market_data = sample_price_data
        system.meta_learner = None
        
        mock_regime = Mock()
        mock_regime.regime = 2
        mock_regime.regime_name = "Risk-Off Crisis"
        mock_regime.confidence = 0.7
        mock_regime.disagreement = 0.4
        mock_regime.transition_warning = True
        mock_regime.probabilities = {2: 0.6}
        mock_regime.individual_predictions = {"hmm": 2}
        
        mock_ensemble = Mock()
        mock_ensemble.predict.return_value = mock_regime
        mock_ensemble.needs_recalibration.return_value = (False, "All good")
        mock_ensemble.get_classifier_performance.return_value = {}
        system.ensemble = mock_ensemble
        
        system.modules = {}
        result = system.analyze()
        
        assert result["meta"]["needs_recalibration"] is False
        assert "classifier_performance" in result["meta"]


# ============================================================================
# FACTOR EXPOSURE ANALYZER TESTS
# ============================================================================

class TestFactorExposureAnalyzer:
    """Tests for FactorExposureAnalyzer."""

    def test_initialization(self):
        """Test factor analyzer initialization."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        assert len(analyzer.STANDARD_FACTORS) == 6
        assert "momentum" in analyzer.factor_history
        assert len(analyzer.factor_history["momentum"]) == 0

    def test_get_regime_parameters(self):
        """Test regime-specific parameters."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        
        params = analyzer.get_regime_parameters(1)
        assert "expectations" in params
        assert "momentum" in params["expectations"]
        
        params2 = analyzer.get_regime_parameters(2)
        assert params2["expectations"]["momentum"]["expected"] == "negative"

    def test_analyze_with_data(self, sample_factor_data):
        """Test full factor analysis."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)
        
        result = analyzer.analyze(sample_factor_data)
        
        assert "signal" in result
        assert result["signal"].signal in ["bullish", "bearish", "neutral"]
        assert "factor_results" in result

    def test_analyze_no_factor_data(self):
        """Test analysis with no matching factor columns."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)
        
        data = pd.DataFrame({"irrelevant": [1, 2, 3]})
        result = analyzer.analyze(data)
        
        assert result["signal"].signal == "neutral"
        assert result["signal"].strength == 0.0

    def test_analyze_factor_positive_expected_positive(self):
        """Test factor behaving as expected (positive in risk-on)."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)  # Risk-On
        
        result = analyzer.analyze_factor("momentum", return_value=0.03, z_score=1.5)
        
        assert result["signal"].signal == "bullish"
        assert result["behaving_as_expected"] is True
        assert result["expected"] == "positive"

    def test_analyze_factor_positive_expected_negative(self):
        """Test factor unexpectedly positive."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(2, 0.85)  # Risk-Off
        
        result = analyzer.analyze_factor("momentum", return_value=0.05, z_score=2.0)
        
        assert result["signal"].signal == "bullish"
        assert result["behaving_as_expected"] is False

    def test_analyze_factor_negative_expected_negative(self):
        """Test factor negative as expected (crisis)."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(2, 0.85)  # Risk-Off
        
        result = analyzer.analyze_factor("momentum", return_value=-0.03)
        
        assert result["behaving_as_expected"] is True
        assert result["signal"].signal == "neutral"

    def test_analyze_factor_negative_expected_positive(self):
        """Test factor unexpectedly negative in risk-on."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)  # Risk-On
        
        result = analyzer.analyze_factor("momentum", return_value=-0.03, z_score=-1.5)
        
        assert result["signal"].signal == "cautious"
        assert result["behaving_as_expected"] is False

    def test_analyze_factor_flat(self):
        """Test factor with flat return."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)
        
        result = analyzer.analyze_factor("momentum", return_value=0.005)
        
        assert result["behaving_as_expected"] is True
        assert result["signal"].signal == "neutral"

    def test_analyze_factor_neutral_expectation(self):
        """Test factor with neutral regime expectation."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(2, 0.85)  # Risk-Off
        
        result = analyzer.analyze_factor("value", return_value=0.02)  # neutral expectation
        
        assert result["signal"].signal == "neutral"
        assert result["behaving_as_expected"] is True

    def test_analyze_factor_stores_history(self):
        """Test that observations are stored in history."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        analyzer.update_regime(1, 0.85)
        
        analyzer.analyze_factor("momentum", return_value=0.02, z_score=1.0, percentile=75.0)
        analyzer.analyze_factor("momentum", return_value=0.03, z_score=1.5, percentile=80.0)
        
        assert len(analyzer.factor_history["momentum"]) == 2
        assert analyzer.factor_history["momentum"][0].return_value == 0.02

    def test_detect_factor_rotation_insufficient_data(self):
        """Test rotation detection with insufficient data."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        result = analyzer.detect_factor_rotation(lookback=20)
        
        assert result["rotations"] == {}

    def test_detect_factor_rotation_with_data(self):
        """Test rotation detection with sufficient history."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer, FactorReturn
        
        analyzer = FactorExposureAnalyzer()
        
        # Add enough history for rotation detection
        for i in range(40):
            obs = FactorReturn(
                factor="momentum",
                return_value=-0.02 if i < 20 else 0.03,  # Switch from negative to positive
                z_score=0.5,
                percentile=50,
            )
            analyzer.factor_history["momentum"].append(obs)
        
        result = analyzer.detect_factor_rotation(lookback=30)
        assert "rotations" in result
        assert "interpretation" in result

    def test_interpret_rotations(self):
        """Test rotation pattern interpretation."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        
        analyzer = FactorExposureAnalyzer()
        
        # Quality improving, momentum deteriorating → defensive rotation
        rotations1 = {"quality": "improving", "momentum": "deteriorating"}
        assert "defensives" in analyzer._interpret_rotations(rotations1)
        
        # Momentum improving, value deteriorating → growth continuing
        rotations2 = {"momentum": "improving", "value": "deteriorating"}
        assert "Growth" in analyzer._interpret_rotations(rotations2)
        
        # Value improving, growth deteriorating → value rotation
        rotations3 = {"value": "improving", "growth": "deteriorating"}
        assert "Value" in analyzer._interpret_rotations(rotations3)
        
        # No clear pattern
        rotations4 = {"size": "stable"}
        assert "No clear" in analyzer._interpret_rotations(rotations4)


# ============================================================================
# CORRELATION ANOMALY DETECTOR TESTS
# ============================================================================

class TestCorrelationAnomalyDetector:
    """Tests for CorrelationAnomalyDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector(window=30)
        assert detector.window == 30
        assert detector.anomaly_threshold == 2.0
        assert len(detector.correlation_history) == 0

    def test_get_regime_parameters(self):
        """Test regime-specific correlation baselines."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        params = detector.get_regime_parameters(1)
        
        assert "baselines" in params
        assert "SPX-TLT" in params["baselines"]

    def test_check_correlation_normal(self):
        """Test normal correlation (no anomaly)."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)  # Risk-On
        
        result = detector.check_correlation("SPX", "TLT", -0.35)
        
        assert result["is_anomaly"] is False
        assert result["pair"] == "SPX-TLT"

    def test_check_correlation_spike_anomaly(self):
        """Test correlation spike anomaly."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)  # Risk-On
        
        # SPX-TLT going very positive in risk-on is anomalous
        result = detector.check_correlation("SPX", "TLT", 0.5)
        
        assert result["is_anomaly"] is True
        assert result["anomaly_type"] == "spike"

    def test_check_correlation_inversion(self):
        """Test correlation inversion anomaly."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)  # Risk-On
        
        # SPX-TLT going very negative beyond normal range
        result = detector.check_correlation("SPX", "TLT", -0.9)
        
        assert result["is_anomaly"] is True
        assert result["anomaly_type"] == "inversion"

    def test_check_correlation_unknown_pair(self):
        """Test correlation check for unknown pair."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        result = detector.check_correlation("ABC", "XYZ", 0.5)
        # Unknown pair uses default baselines
        assert isinstance(result["is_anomaly"], bool)

    def test_interpret_anomaly_spx_tlt_riskon(self):
        """Test SPX-TLT correlation spike interpretation in risk-on."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)  # Risk-On
        
        anomaly = CorrelationAnomaly(
            pair="SPX-TLT",
            current_corr=0.5,
            baseline_corr=-0.35,
            deviation=3.0,
            anomaly_type="spike",
        )
        
        signal = detector._interpret_anomaly(anomaly)
        assert signal.signal == "bearish"
        assert signal.strength == 0.8

    def test_interpret_anomaly_spx_tlt_stagflation(self):
        """Test SPX-TLT correlation normalization in stagflation."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(3, 0.85)  # Stagflation
        
        anomaly = CorrelationAnomaly(
            pair="SPX-TLT",
            current_corr=-0.5,
            baseline_corr=0.2,
            deviation=-3.0,
            anomaly_type="inversion",
        )
        
        signal = detector._interpret_anomaly(anomaly)
        assert signal.signal == "bullish"

    def test_interpret_anomaly_spx_vix_breakdown(self):
        """Test SPX-VIX correlation breakdown."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        anomaly = CorrelationAnomaly(
            pair="SPX-VIX",
            current_corr=-0.3,
            baseline_corr=-0.75,
            deviation=3.0,
            anomaly_type="breakdown",
        )
        
        signal = detector._interpret_anomaly(anomaly)
        assert signal.signal == "cautious"
        assert signal.strength == 0.7

    def test_interpret_anomaly_generic(self):
        """Test generic anomaly interpretation."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        anomaly = CorrelationAnomaly(
            pair="DXY-GLD",
            current_corr=0.5,
            baseline_corr=-0.4,
            deviation=3.5,
            anomaly_type="spike",
        )
        
        signal = detector._interpret_anomaly(anomaly)
        assert signal.signal == "cautious"

    def test_generate_anomaly_signal_multiple(self):
        """Test signal generation with multiple anomalies."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        anomalies = [
            CorrelationAnomaly("SPX-TLT", 0.5, -0.35, 3.0, "spike"),
            CorrelationAnomaly("SPX-GLD", 0.7, 0.0, 3.5, "spike"),
            CorrelationAnomaly("TLT-GLD", -0.5, 0.2, -3.5, "inversion"),
        ]
        
        signal = detector._generate_anomaly_signal(anomalies)
        assert signal.signal == "cautious"
        assert signal.strength == 0.9  # Multiple anomalies = high strength

    def test_generate_anomaly_signal_two(self):
        """Test signal generation with exactly two anomalies."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        anomalies = [
            CorrelationAnomaly("SPX-TLT", 0.5, -0.35, 3.0, "spike"),
            CorrelationAnomaly("SPX-GLD", 0.7, 0.0, 3.5, "spike"),
        ]
        
        signal = detector._generate_anomaly_signal(anomalies)
        assert signal.signal == "cautious"
        assert signal.strength == 0.7

    def test_generate_anomaly_signal_single(self):
        """Test signal generation with one anomaly."""
        from src.modules.correlation_anomaly_detector import (
            CorrelationAnomalyDetector, CorrelationAnomaly
        )
        
        detector = CorrelationAnomalyDetector()
        detector.update_regime(1, 0.85)
        
        anomalies = [
            CorrelationAnomaly("SPX-TLT", 0.5, -0.35, 3.0, "spike"),
        ]
        
        signal = detector._generate_anomaly_signal(anomalies)
        assert signal.signal == "bearish"  # SPX-TLT spike in risk-on

    def test_analyze_with_price_data(self, sample_price_data):
        """Test full analysis with price data."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector(window=60)
        detector.update_regime(1, 0.85)
        
        result = detector.analyze(sample_price_data)
        
        assert "signal" in result
        assert "correlations" in result
        assert "anomaly_count" in result
        assert isinstance(result["anomaly_count"], int)

    def test_find_column(self, sample_price_data):
        """Test column name matching."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        
        assert detector._find_column(sample_price_data, "SPX") == "SPX"
        assert detector._find_column(sample_price_data, "UNKNOWN") is None

    def test_get_correlation_matrix(self, sample_price_data):
        """Test full correlation matrix calculation."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector(window=60)
        matrix = detector.get_correlation_matrix(sample_price_data, window=60)
        
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] == matrix.shape[1]  # Square matrix

    def test_detect_correlation_regime_shift_no_history(self):
        """Test regime shift detection with no history."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        result = detector.detect_correlation_regime_shift()
        
        assert result["shift_detected"] is False
        assert "Insufficient" in result["reason"]

    def test_detect_correlation_regime_shift_with_history(self):
        """Test regime shift detection with sufficient history."""
        from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector
        
        detector = CorrelationAnomalyDetector()
        
        # Add correlation history
        for i in range(50):
            if i < 25:
                # First half: negative SPX-TLT
                detector.correlation_history.append({"SPX-TLT": -0.4, "SPX-GLD": 0.1})
            else:
                # Second half: positive SPX-TLT (regime shift!)
                detector.correlation_history.append({"SPX-TLT": 0.3, "SPX-GLD": 0.5})
        
        result = detector.detect_correlation_regime_shift(lookback=20)
        assert result["shift_detected"] is True
        assert "SPX-TLT" in result["shifting_pairs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
