"""
Additional Tests to Reach 80%+ Coverage.

Tests for:
- DatabaseStorage: save/load market data, macro data, regime history
- MacroEventTracker: Event analysis, surprise interpretation, market reactions
- OptionsSurfaceMonitor: Vol level, skew analysis, term structure
- AMRCAIS.initialize (mocked): Initialization flow

Coverage target: Push total from 68% to 80%+
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary directory for database."""
    tmpdir = tempfile.mkdtemp()
    db = os.path.join(tmpdir, "test.db")
    yield db
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_close_df():
    """DataFrame with Close column for storage tests."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "Open": np.random.uniform(99, 105, 100),
        "High": np.random.uniform(103, 110, 100),
        "Low": np.random.uniform(95, 100, 100),
        "Close": np.random.uniform(100, 105, 100),
        "Volume": np.random.randint(1e6, 1e7, 100),
    }, index=dates)


@pytest.fixture
def sample_macro_df():
    """DataFrame for macro data storage tests."""
    dates = pd.date_range("2023-01-01", periods=50, freq="MS")
    return pd.DataFrame({
        "Value": np.random.uniform(0, 5, 50),
    }, index=dates)


# ============================================================================
# DATABASE STORAGE TESTS
# ============================================================================

class TestDatabaseStorageSaveLoad:
    """Tests for DatabaseStorage save and load operations."""

    def test_save_market_data(self, temp_db_path, sample_close_df):
        """Test saving market data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        rows = storage.save_market_data("SPX", sample_close_df)
        
        assert rows == 100

    def test_load_market_data(self, temp_db_path, sample_close_df):
        """Test loading market data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        
        loaded = storage.load_market_data("SPX")
        assert len(loaded) == 100
        assert "Close" in loaded.columns

    def test_load_market_data_with_dates(self, temp_db_path, sample_close_df):
        """Test loading market data with date range."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        
        loaded = storage.load_market_data(
            "SPX",
            start_date="2023-02-01",
            end_date="2023-03-01",
        )
        assert len(loaded) > 0
        assert len(loaded) < 100

    def test_load_market_data_empty(self, temp_db_path):
        """Test loading when no data exists."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        loaded = storage.load_market_data("NONEXISTENT")
        
        assert loaded.empty

    def test_save_market_data_empty_raises(self, temp_db_path):
        """Test saving empty DataFrame raises ValueError."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        with pytest.raises(ValueError, match="empty"):
            storage.save_market_data("SPX", pd.DataFrame())

    def test_save_market_data_no_close_raises(self, temp_db_path):
        """Test saving without Close column raises ValueError."""
        from src.data_pipeline.storage import DatabaseStorage
        
        df = pd.DataFrame({"Price": [100, 101]}, index=pd.date_range("2023-01-01", periods=2))
        storage = DatabaseStorage(temp_db_path)
        
        with pytest.raises(ValueError, match="Close"):
            storage.save_market_data("SPX", df)

    def test_save_market_data_replace(self, temp_db_path, sample_close_df):
        """Test replacing existing data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        
        # Save again with replace=True (default)
        rows = storage.save_market_data("SPX", sample_close_df)
        assert rows == 100
        
        # No duplicates
        loaded = storage.load_market_data("SPX")
        assert len(loaded) == 100

    def test_get_available_assets(self, temp_db_path, sample_close_df):
        """Test listing available assets."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        storage.save_market_data("TLT", sample_close_df)
        
        assets = storage.get_available_assets()
        assert "SPX" in assets
        assert "TLT" in assets

    def test_get_date_range(self, temp_db_path, sample_close_df):
        """Test getting data date range."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        
        date_range = storage.get_date_range("SPX")
        assert date_range is not None
        assert date_range[0] <= date_range[1]

    def test_get_date_range_no_data(self, temp_db_path):
        """Test date range for nonexistent asset."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        result = storage.get_date_range("NONEXISTENT")
        assert result is None

    def test_is_data_fresh(self, temp_db_path):
        """Test data freshness check."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        
        # Save data ending today
        dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
        df = pd.DataFrame({"Close": np.random.uniform(100, 105, 10)}, index=dates)
        storage.save_market_data("SPX", df)
        
        assert storage.is_data_fresh("SPX", max_age_days=2) is True
        assert storage.is_data_fresh("NONEXISTENT") is False

    def test_is_data_stale(self, temp_db_path):
        """Test stale data detection."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        
        # Save old data
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Close": np.random.uniform(100, 105, 10)}, index=dates)
        storage.save_market_data("OLD_ASSET", df)
        
        assert storage.is_data_fresh("OLD_ASSET", max_age_days=1) is False


class TestDatabaseStorageMacroAndRegime:
    """Tests for macro data and regime storage."""

    def test_save_macro_data(self, temp_db_path, sample_macro_df):
        """Test saving macro data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        rows = storage.save_macro_data("CPI", sample_macro_df)
        assert rows == 50

    def test_load_macro_data(self, temp_db_path, sample_macro_df):
        """Test loading macro data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_macro_data("CPI", sample_macro_df)
        
        loaded = storage.load_macro_data("CPI")
        assert len(loaded) == 50
        assert "Value" in loaded.columns

    def test_load_macro_data_with_dates(self, temp_db_path, sample_macro_df):
        """Test loading macro data with date range filter."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_macro_data("CPI", sample_macro_df)
        
        loaded = storage.load_macro_data(
            "CPI",
            start_date="2023-06-01",
            end_date="2023-12-01",
        )
        assert len(loaded) > 0
        assert len(loaded) < 50

    def test_load_macro_data_empty(self, temp_db_path):
        """Test loading nonexistent macro series."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        result = storage.load_macro_data("NONEXISTENT")
        assert result.empty

    def test_save_macro_data_empty_raises(self, temp_db_path):
        """Test saving empty macro DataFrame raises ValueError."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        with pytest.raises(ValueError, match="empty"):
            storage.save_macro_data("CPI", pd.DataFrame())

    def test_save_regime(self, temp_db_path):
        """Test saving regime classification."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_regime(
            date=datetime(2024, 1, 15),
            regime=1,
            confidence=0.85,
            disagreement=0.2,
            votes={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 2},
        )
        
        history = storage.load_regime_history()
        assert len(history) == 1
        assert history.iloc[0]["Regime"] == 1

    def test_save_regime_without_votes(self, temp_db_path):
        """Test saving regime without individual votes."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_regime(
            date=datetime(2024, 1, 15),
            regime=2,
            confidence=0.7,
            disagreement=0.4,
        )
        
        history = storage.load_regime_history()
        assert len(history) == 1

    def test_load_regime_history_with_dates(self, temp_db_path):
        """Test loading regime history with date filters."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        for i in range(10):
            storage.save_regime(
                date=datetime(2024, 1, 1) + timedelta(days=i),
                regime=1 if i < 5 else 2,
                confidence=0.8,
                disagreement=0.2,
            )
        
        loaded = storage.load_regime_history(
            start_date="2024-01-05",
            end_date="2024-01-08",
        )
        assert len(loaded) > 0
        assert len(loaded) < 10

    def test_load_regime_history_empty(self, temp_db_path):
        """Test loading empty regime history."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        result = storage.load_regime_history()
        assert result.empty

    def test_clear_all_data(self, temp_db_path, sample_close_df, sample_macro_df):
        """Test clearing all data."""
        from src.data_pipeline.storage import DatabaseStorage
        
        storage = DatabaseStorage(temp_db_path)
        storage.save_market_data("SPX", sample_close_df)
        storage.save_macro_data("CPI", sample_macro_df)
        storage.save_regime(datetime(2024, 1, 1), 1, 0.8, 0.2)
        
        storage.clear_all_data()
        
        assert storage.load_market_data("SPX").empty
        assert storage.load_macro_data("CPI").empty
        assert storage.load_regime_history().empty


# ============================================================================
# MACRO EVENT TRACKER TESTS
# ============================================================================

class TestMacroEventTrackerExtended:
    """Extended tests for MacroEventTracker."""

    def test_analyze_event_nfp_riskon(self):
        """Test NFP surprise interpretation in Risk-On."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)  # Risk-On
        
        result = tracker.analyze_event("NFP", actual=250, consensus=180)
        
        assert result["signal"].signal == "bullish"
        assert result["surprise"] > 0
        assert result["regime_weight"] == 1.2

    def test_analyze_event_nfp_stagflation(self):
        """Test NFP surprise in Stagflation (mixed signal)."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(3, 0.8)  # Stagflation
        
        result = tracker.analyze_event("NFP", actual=280, consensus=200)
        
        # In stagflation, strong NFP is mixed → neutral
        assert result["signal"].signal == "neutral"

    def test_analyze_event_cpi_stagflation(self):
        """Test CPI surprise in Stagflation."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(3, 0.8)
        
        result = tracker.analyze_event("CPI", actual=3.3, consensus=3.1)
        
        assert result["signal"].signal == "bearish"

    def test_analyze_event_negative_surprise(self):
        """Test negative surprise interpretation."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)  # Risk-On
        
        result = tracker.analyze_event("NFP", actual=100, consensus=200)
        
        assert result["surprise"] < 0
        assert result["signal"].signal == "bearish"

    def test_analyze_event_inline(self):
        """Test inline result (no surprise)."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)
        
        result = tracker.analyze_event("NFP", actual=200, consensus=200)
        
        assert abs(result["surprise"]) < 0.5
        assert result["signal"].signal == "neutral"

    def test_analyze_event_stores_history(self):
        """Test that events are stored in history."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)
        
        tracker.analyze_event("NFP", actual=250, consensus=200)
        tracker.analyze_event("CPI", actual=3.1, consensus=3.0)
        
        assert len(tracker.event_history) == 2

    def test_macro_event_calculate_surprise(self):
        """Test MacroEvent surprise calculation."""
        from src.modules.macro_event_tracker import MacroEvent
        
        event = MacroEvent(
            event_type="NFP",
            release_date=datetime(2024, 1, 5),
            actual=250,
            consensus=175,
        )
        surprise = event.calculate_surprise(historical_std=75.0)
        assert surprise == pytest.approx(1.0, abs=0.01)

    def test_macro_event_calculate_surprise_zero_std(self):
        """Test surprise with zero std defaults to 1."""
        from src.modules.macro_event_tracker import MacroEvent
        
        event = MacroEvent(
            event_type="NFP",
            release_date=datetime(2024, 1, 5),
            actual=250,
            consensus=200,
        )
        surprise = event.calculate_surprise(historical_std=0)
        assert surprise == 50.0  # (250-200)/1.0

    def test_analyze_with_indicator_data(self):
        """Test analyze with DataFrame containing macro indicators."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)
        
        dates = pd.date_range("2024-01-01", periods=10, freq="MS")
        data = pd.DataFrame({
            "NFP": np.random.uniform(150, 300, 10),
            "CPI": np.random.uniform(2.5, 4.0, 10),
        }, index=dates)
        
        result = tracker.analyze(data)
        assert "signal" in result
        assert "individual_signals" in result

    def test_analyze_with_no_indicators(self):
        """Test analyze with no matching indicators."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)
        
        data = pd.DataFrame({"irrelevant": [1, 2, 3]})
        result = tracker.analyze(data)
        assert result["signal"].signal == "neutral"

    def test_interpret_indicator_change_growth(self):
        """Test growth indicator change interpretation."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        
        # Risk-On: positive growth → bullish
        tracker.update_regime(1, 0.85)
        signal = tracker._interpret_indicator_change("NFP", 50.0)
        assert signal.signal == "bullish"
        
        # Risk-On: negative growth → bearish
        signal = tracker._interpret_indicator_change("NFP", -50.0)
        assert signal.signal == "bearish"

    def test_interpret_indicator_change_stagflation(self):
        """Test growth indicator change in stagflation."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(3, 0.8)  # Stagflation
        
        # Strong growth in stagflation → neutral (more Fed tightening)
        signal = tracker._interpret_indicator_change("NFP", 50.0)
        assert signal.signal == "neutral"
        
        # Weak growth in stagflation → bullish (less Fed tightening)
        signal = tracker._interpret_indicator_change("GDP", -0.5)
        assert signal.signal == "bullish"

    def test_interpret_inflation_indicator(self):
        """Test inflation indicator interpretation."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        
        # Stagflation: rising inflation → bearish
        tracker.update_regime(3, 0.8)
        signal = tracker._interpret_indicator_change("CPI", 0.5)
        assert signal.signal == "bearish"
        
        # Stagflation: falling inflation → bullish
        signal = tracker._interpret_indicator_change("CPI", -0.3)
        assert signal.signal == "bullish"

    def test_get_event_statistics(self):
        """Test event statistics retrieval."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        tracker.update_regime(1, 0.85)
        
        tracker.analyze_event("NFP", actual=250, consensus=200)
        tracker.analyze_event("NFP", actual=180, consensus=200)
        tracker.analyze_event("NFP", actual=220, consensus=200)
        
        stats = tracker.get_event_statistics("NFP")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["positive_surprise_count"] >= 0

    def test_get_event_statistics_empty(self):
        """Test event statistics for unknown event type."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        assert tracker.get_event_statistics("UNKNOWN") is None

    def test_get_regime_parameters(self):
        """Test macro event regime parameters."""
        from src.modules.macro_event_tracker import MacroEventTracker
        
        tracker = MacroEventTracker()
        params = tracker.get_regime_parameters(1)
        
        assert "weights" in params
        assert "interpretations" in params
        assert "NFP" in params["weights"]


# ============================================================================
# OPTIONS SURFACE MONITOR TESTS
# ============================================================================

class TestOptionsSurfaceMonitorExtended:
    """Extended tests for OptionsSurfaceMonitor."""

    def test_analyze_with_vix_data(self):
        """Test analysis with VIX column in data."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "SPX": np.random.uniform(4000, 4500, 100),
            "VIX": np.random.uniform(12, 20, 100),
        }, index=dates)
        
        result = monitor.analyze(data)
        assert "signal" in result
        assert "vix_level" in result

    def test_analyze_with_vixcls(self):
        """Test analysis with VIXCLS column."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(2, 0.85)
        
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "VIXCLS": np.random.uniform(25, 40, 100),
        }, index=dates)
        
        result = monitor.analyze(data)
        assert "signal" in result

    def test_analyze_no_vol_data(self):
        """Test analysis without any vol data."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        data = pd.DataFrame({"SPX": [4000, 4100]})
        result = monitor.analyze(data)
        assert result["signal"].signal == "neutral"

    def test_vol_level_low(self):
        """Test low vol interpretation."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        signal = monitor._analyze_vol_level(10.0)  # Below 12 for risk-on
        assert signal.signal == "bullish"

    def test_vol_level_normal(self):
        """Test normal vol interpretation."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        signal = monitor._analyze_vol_level(14.0)  # Between 12-16
        assert signal.signal == "neutral"

    def test_vol_level_elevated(self):
        """Test elevated vol interpretation."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        signal = monitor._analyze_vol_level(24.0)  # Between 22-28
        assert signal.signal == "bearish"

    def test_vol_level_high(self):
        """Test high vol interpretation."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        signal = monitor._analyze_vol_level(35.0)  # Above 28
        assert signal.signal == "bearish"
        assert signal.strength == 0.8

    def test_vol_level_crisis_regime(self):
        """Test vol level in crisis regime (higher thresholds)."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(2, 0.85)  # Risk-Off
        
        # 25 is LOW for crisis regime
        signal = monitor._analyze_vol_level(18.0)
        assert signal.signal == "bullish"

    def test_analyze_skew_extreme(self):
        """Test extreme put skew analysis."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_skew(put_skew=12.0)
        assert result["skew_state"] == "extreme"

    def test_analyze_skew_steepening(self):
        """Test steepening put skew."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_skew(put_skew=6.0, prior_put_skew=3.0)
        assert result["skew_state"] == "steepening"

    def test_analyze_skew_flattening(self):
        """Test flattening put skew."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_skew(put_skew=3.0, prior_put_skew=6.0)
        assert result["skew_state"] == "flattening"
        assert result["signal"].signal == "bullish"

    def test_analyze_skew_stable(self):
        """Test stable put skew."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_skew(put_skew=5.0, prior_put_skew=4.5)
        assert result["skew_state"] == "stable"
        assert result["signal"].signal == "neutral"

    def test_analyze_skew_crisis_extreme(self):
        """Test extreme skew in crisis (capitulation signal)."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(2, 0.85)  # Risk-Off
        
        result = monitor.analyze_skew(put_skew=15.0)
        assert result["skew_state"] == "extreme"
        assert result["signal"].signal == "bullish"  # Capitulation signal!

    def test_analyze_term_structure_contango(self):
        """Test vol term structure in contango."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_term_structure(near_vol=15, far_vol=20)
        assert result["structure"] == "contango"
        assert result["signal"].signal == "bullish"

    def test_analyze_term_structure_backwardation(self):
        """Test vol term structure in backwardation."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_term_structure(near_vol=25, far_vol=18)
        assert result["structure"] == "backwardation"
        assert result["signal"].signal == "bearish"

    def test_analyze_term_structure_flat(self):
        """Test vol term structure flat."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(1, 0.85)
        
        result = monitor.analyze_term_structure(near_vol=18, far_vol=19)
        assert result["structure"] == "flat"
        assert result["signal"].signal == "neutral"

    def test_analyze_term_structure_crisis_regime(self):
        """Test contango in crisis regime is neutral."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        monitor.update_regime(2, 0.85)  # Crisis
        
        result = monitor.analyze_term_structure(near_vol=30, far_vol=35)
        assert result["structure"] == "contango"
        assert result["signal"].signal == "neutral"

    def test_get_regime_parameters(self):
        """Test options regime parameters."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        
        monitor = OptionsSurfaceMonitor()
        params = monitor.get_regime_parameters(1)
        
        assert "vol_levels" in params
        assert "put_skew_rules" in params
        assert "low" in params["vol_levels"]


# ============================================================================
# AMRCAIS INITIALIZATION TESTS
# ============================================================================

class TestAMRCAISInitialization:
    """Tests for AMRCAIS.initialize flow (mocked)."""

    @patch("src.modules.base.AnalyticalModule._load_config", return_value={})
    @patch("src.main.DataPipeline")
    @patch("src.main.RegimeEnsemble")
    @patch("src.main.MetaLearner")
    def test_initialize_no_data(self, mock_ml, mock_ens, mock_pipe, mock_cfg, temp_db_path):
        """Test initialization when no data is available."""
        from src.main import AMRCAIS
        
        mock_pipeline = Mock()
        mock_pipeline.fetch_market_data.side_effect = Exception("No API keys")
        mock_pipe.return_value = mock_pipeline
        
        system = AMRCAIS(db_path=temp_db_path)
        system.initialize()
        
        assert system._is_initialized is True
        assert len(system.modules) == 5

    @patch("src.modules.base.AnalyticalModule._load_config", return_value={})
    @patch("src.main.DataPipeline")
    @patch("src.main.RegimeEnsemble")
    @patch("src.main.MetaLearner")
    def test_initialize_with_data(self, mock_ml, mock_ens, mock_pipe, mock_cfg, temp_db_path):
        """Test initialization with successful data fetch."""
        from src.main import AMRCAIS
        
        # Create mock data with enough rows
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        mock_data = pd.DataFrame({
            "SPX": np.random.uniform(3000, 4000, 500),
            "TLT": np.random.uniform(120, 140, 500),
        }, index=dates)
        
        mock_pipeline = Mock()
        mock_pipeline.fetch_market_data.return_value = mock_data
        mock_pipe.return_value = mock_pipeline
        
        system = AMRCAIS(db_path=temp_db_path)
        system.initialize()
        
        assert system._is_initialized is True
        # With >100 rows, ensemble.fit should be called
        mock_ens.return_value.fit.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
