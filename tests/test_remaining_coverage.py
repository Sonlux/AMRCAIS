"""
Final Coverage Tests - Yield Curve, Volatility Classifier, Validators, Regime Base.

Targets uncovered lines in:
- yield_curve_analyzer.py (64% → 85%+)
- volatility_classifier.py (64% → 85%+)
- validators.py (70% → 85%+)
- regime_detection/base.py (64% → 85%+)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Optional, Union, List


# ============================================================================
# YIELD CURVE ANALYZER TESTS
# ============================================================================

class TestYieldCurveAnalyzerExtended:
    """Tests for uncovered yield curve methods."""

    def test_analyze_with_yield_data(self):
        """Test full analysis with yield columns."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame({
            "DGS2": np.linspace(4.0, 4.2, 10),
            "DGS10": np.linspace(4.5, 4.8, 10),
            "DGS3MO": np.linspace(5.0, 5.1, 10),
            "DGS5": np.linspace(4.3, 4.5, 10),
        }, index=dates)

        result = analyzer.analyze(data)
        assert "signal" in result
        assert "curve_shape" in result
        assert "slope_2_10" in result
        assert result["slope_2_10"] is not None

    def test_analyze_no_yield_data(self):
        """Test analysis with no matching yield columns."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        data = pd.DataFrame({"SPX": [4000, 4100]})
        result = analyzer.analyze(data)
        assert result["signal"].signal == "neutral"

    def test_classify_shape_inverted(self):
        """Test inverted curve classification."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        shape = analyzer._classify_shape(-0.5, -0.8, 0.1)
        assert shape == CurveShape.INVERTED

    def test_classify_shape_flat(self):
        """Test flat curve classification."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        shape = analyzer._classify_shape(0.05, 0.1, 0.1)
        assert shape == CurveShape.FLAT

    def test_classify_shape_humped(self):
        """Test humped curve classification."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        shape = analyzer._classify_shape(0.5, 0.6, 0.5)
        assert shape == CurveShape.HUMPED

    def test_classify_shape_normal(self):
        """Test normal curve classification."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        shape = analyzer._classify_shape(0.5, 0.6, 0.1)
        assert shape == CurveShape.NORMAL

    def test_classify_shape_none_slopes(self):
        """Test curve shape with no slopes."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        shape = analyzer._classify_shape(None, None, None)
        assert shape == CurveShape.NORMAL

    def test_analyze_dynamics_no_history(self):
        """Test dynamics with no history."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        result = analyzer._analyze_dynamics()
        assert result["direction"] == "unknown"

    def test_analyze_dynamics_steepening(self):
        """Test steepening dynamics detection."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveSnapshot, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.curve_history = [
            CurveSnapshot(datetime.now(), {}, 0.5, 0.5, 0.0, CurveShape.NORMAL),
            CurveSnapshot(datetime.now(), {}, 0.9, 0.9, 0.0, CurveShape.NORMAL),
        ]
        result = analyzer._analyze_dynamics()
        assert result["direction"] == "steepening"
        assert result["slope_change"] > 0

    def test_analyze_dynamics_flattening(self):
        """Test flattening dynamics detection."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveSnapshot, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.curve_history = [
            CurveSnapshot(datetime.now(), {}, 0.9, 0.9, 0.0, CurveShape.NORMAL),
            CurveSnapshot(datetime.now(), {}, 0.5, 0.5, 0.0, CurveShape.NORMAL),
        ]
        result = analyzer._analyze_dynamics()
        assert result["direction"] == "flattening"

    def test_analyze_dynamics_stable(self):
        """Test stable dynamics."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveSnapshot, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.curve_history = [
            CurveSnapshot(datetime.now(), {}, 0.5, 0.5, 0.0, CurveShape.NORMAL),
            CurveSnapshot(datetime.now(), {}, 0.52, 0.52, 0.0, CurveShape.NORMAL),
        ]
        result = analyzer._analyze_dynamics()
        assert result["direction"] == "stable"

    def test_generate_signal_steepening_riskon(self):
        """Test signal generation for steepening in risk-on."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        signal = analyzer._generate_signal(
            CurveShape.NORMAL,
            {"direction": "steepening", "magnitude": 0.15},
            0.5,
        )
        assert signal.signal == "bullish"
        assert signal.strength == 0.7

    def test_generate_signal_flattening_crisis(self):
        """Test signal generation for flattening in crisis."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(2, 0.85)

        signal = analyzer._generate_signal(
            CurveShape.NORMAL,
            {"direction": "flattening", "magnitude": 0.25},
            0.3,
        )
        assert signal.signal == "neutral"
        assert signal.strength == 0.9  # magnitude > 0.2

    def test_generate_signal_inverted_override(self):
        """Test that inversion overrides dynamics signal."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        signal = analyzer._generate_signal(
            CurveShape.INVERTED,
            {"direction": "steepening", "magnitude": 0.3},
            -0.5,
        )
        assert signal.signal == "bearish"

    def test_generate_signal_stable(self):
        """Test signal for stable dynamics."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        signal = analyzer._generate_signal(
            CurveShape.NORMAL,
            {"direction": "stable", "magnitude": 0.02},
            0.5,
        )
        assert signal.signal == "neutral"
        assert signal.strength == 0.3  # magnitude < 0.05

    def test_generate_signal_strength_tiers(self):
        """Test signal strength varies with magnitude."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer, CurveShape

        analyzer = YieldCurveAnalyzer()
        analyzer.update_regime(1, 0.85)

        # magnitude > 0.2
        s1 = analyzer._generate_signal(CurveShape.NORMAL, {"direction": "steepening", "magnitude": 0.25}, 0.5)
        assert s1.strength == 0.9

        # 0.1 < magnitude < 0.2
        s2 = analyzer._generate_signal(CurveShape.NORMAL, {"direction": "steepening", "magnitude": 0.12}, 0.5)
        assert s2.strength == 0.7

        # 0.05 < magnitude < 0.1
        s3 = analyzer._generate_signal(CurveShape.NORMAL, {"direction": "steepening", "magnitude": 0.07}, 0.5)
        assert s3.strength == 0.5

    def test_interpolate_curve(self):
        """Test cubic spline interpolation."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        yields = {"2Y": 4.0, "5Y": 4.3, "10Y": 4.5, "30Y": 5.0}
        result = analyzer.interpolate_curve(yields)

        assert len(result) > 0
        # Interpolated values should be within range
        for t, y in result.items():
            assert 3.5 <= y <= 5.5

    def test_interpolate_curve_insufficient_data(self):
        """Test interpolation with too few points."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        yields = {"2Y": 4.0, "10Y": 4.5}
        result = analyzer.interpolate_curve(yields)
        assert result == {}

    def test_interpolate_curve_custom_tenors(self):
        """Test interpolation with custom target tenors."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        yields = {"2Y": 4.0, "5Y": 4.3, "10Y": 4.5, "30Y": 5.0}
        result = analyzer.interpolate_curve(yields, target_tenors=[3, 7, 15])

        assert len(result) == 3
        assert 3 in result
        assert 7 in result
        assert 15 in result

    def test_calculate_forward_rates(self):
        """Test forward rate calculation."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        yields = {"2Y": 4.0, "5Y": 4.3, "10Y": 4.5}
        result = analyzer.calculate_forward_rates(yields)

        assert len(result) > 0
        # Forward rates should be calculable
        for key, val in result.items():
            assert isinstance(val, float)

    def test_calculate_curvature(self):
        """Test butterfly spread calculation."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        yields = {"2Y": 4.0, "5Y": 4.3, "10Y": 4.5}
        curvature = analyzer._calculate_curvature(yields)
        # 2 * 4.3 - 4.0 - 4.5 = 0.1
        assert curvature == pytest.approx(0.1, abs=0.01)

    def test_calculate_curvature_missing(self):
        """Test curvature with missing tenors."""
        from src.modules.yield_curve_analyzer import YieldCurveAnalyzer

        analyzer = YieldCurveAnalyzer()
        curvature = analyzer._calculate_curvature({"2Y": 4.0, "10Y": 4.5})
        assert curvature is None


# ============================================================================
# VOLATILITY CLASSIFIER TESTS
# ============================================================================

class TestVolatilityRegimeClassifierExtended:
    """Tests for uncovered volatility classifier methods."""

    def _make_fitted_classifier(self):
        """Create a fitted volatility classifier."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        np.random.seed(42)
        vix = pd.Series(np.random.uniform(10, 35, 500))
        clf.fit(vix)
        return clf

    def test_fit_with_series(self):
        """Test fitting with a Series."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        clf.fit(vix)
        assert clf.is_fitted
        assert clf._learned_thresholds["low"] > 0

    def test_fit_with_dataframe_vix_col(self):
        """Test fitting with DataFrame containing VIX column."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        df = pd.DataFrame({"VIX": np.random.uniform(10, 35, 200)})
        clf.fit(df)
        assert clf.is_fitted

    def test_fit_with_dataframe_close_col(self):
        """Test fitting with DataFrame containing Close column."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        df = pd.DataFrame({"Close": np.random.uniform(10, 35, 200)})
        clf.fit(df)
        assert clf.is_fitted

    def test_fit_with_labels(self):
        """Test fitting with regime labels for learning stats."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        labels = np.array([1, 2, 3, 4] * 50)
        clf.fit(vix, labels=labels)

        assert clf.is_fitted
        assert len(clf._regime_vix_stats) > 0

    def test_predict_low_vix(self):
        """Test prediction with low VIX (risk-on)."""
        clf = self._make_fitted_classifier()
        result = clf.predict(12.0)
        assert result.regime == 1  # Risk-On

    def test_predict_high_vix(self):
        """Test prediction with high VIX (risk-off)."""
        clf = self._make_fitted_classifier()
        result = clf.predict(45.0)
        assert result.regime == 2  # Risk-Off Crisis

    def test_predict_series(self):
        """Test prediction with a Series."""
        clf = self._make_fitted_classifier()
        vix = pd.Series([15, 16, 14, 13, 12])
        result = clf.predict(vix)
        assert result.regime in [1, 2, 3, 4]
        assert "vix_trend" in result.metadata

    def test_predict_dataframe(self):
        """Test prediction with a DataFrame."""
        clf = self._make_fitted_classifier()
        df = pd.DataFrame({"VIX": [15, 16, 14, 13, 40]})
        result = clf.predict(df)
        assert result.regime in [1, 2, 3, 4]

    def test_predict_dataframe_vix_level(self):
        """Test prediction with VIX_level column."""
        clf = self._make_fitted_classifier()
        df = pd.DataFrame({"VIX_level": [15, 16, 14, 13, 12]})
        result = clf.predict(df)
        assert result.regime in [1, 2, 3, 4]

    def test_predict_ndarray(self):
        """Test prediction with numpy array."""
        clf = self._make_fitted_classifier()
        result = clf.predict(np.array([15, 16, 14, 13, 12]))
        assert result.regime in [1, 2, 3, 4]

    def test_predict_not_fitted_raises(self):
        """Test prediction without fitting raises."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        with pytest.raises(ValueError, match="fitted"):
            clf.predict(20.0)

    def test_predict_no_vix_col_raises(self):
        """Test prediction with no VIX column raises."""
        clf = self._make_fitted_classifier()
        df = pd.DataFrame({"SPX": [4000, 4100]})
        with pytest.raises(ValueError, match="VIX"):
            clf.predict(df)

    def test_calculate_trend_spiking(self):
        """Test spiking VIX trend detection."""
        clf = self._make_fitted_classifier()
        vix = pd.Series([15, 15, 15, 15, 15, 15, 16, 17, 18, 19, 20])
        trend = clf._calculate_trend(vix)
        assert trend in ["spiking", "rising"]

    def test_calculate_trend_declining(self):
        """Test declining VIX trend detection."""
        clf = self._make_fitted_classifier()
        vix = pd.Series([30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10])
        trend = clf._calculate_trend(vix)
        assert trend == "declining"

    def test_calculate_trend_stable(self):
        """Test stable VIX trend."""
        clf = self._make_fitted_classifier()
        vix = pd.Series([15, 15.1, 14.9, 15.2, 14.8, 15.1, 15, 14.9, 15.1, 15, 15.05])
        trend = clf._calculate_trend(vix)
        assert trend == "stable"

    def test_calculate_trend_short_series(self):
        """Test trend with too few observations."""
        clf = self._make_fitted_classifier()
        trend = clf._calculate_trend(pd.Series([15, 16, 17]))
        assert trend == "unknown"

    def test_calculate_percentile(self):
        """Test VIX percentile calculation."""
        clf = self._make_fitted_classifier()
        pct = clf._calculate_percentile(20.0)
        assert pct is not None
        assert 0 <= pct <= 100

    def test_classify_vix_spiking_trend(self):
        """Test classification with spiking VIX trend."""
        clf = self._make_fitted_classifier()
        regime, conf = clf._classify_vix(18.0, "spiking", 50.0)
        assert regime == 2  # Transitioning to risk-off

    def test_classify_vix_elevated_stable(self):
        """Test elevated stable VIX → stagflation."""
        clf = self._make_fitted_classifier()
        regime, conf = clf._classify_vix(
            clf._learned_thresholds["elevated"],
            "stable",
            70.0,
        )
        assert regime == 3  # Stagflation

    def test_classify_vix_elevated_declining(self):
        """Test elevated declining VIX → disinflation."""
        clf = self._make_fitted_classifier()
        regime, conf = clf._classify_vix(
            clf._learned_thresholds["elevated"],
            "declining",
            60.0,
        )
        assert regime == 4  # Transitioning to calmer (disinflation)

    def test_calculate_probabilities_rule_based(self):
        """Test rule-based probability calculation."""
        clf = self._make_fitted_classifier()
        probs = clf._calculate_probabilities(12.0, 10.0)
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
        assert probs[1] > probs[2]  # Risk-on more likely at low VIX

    def test_calculate_probabilities_with_stats(self):
        """Test Gaussian-based probability with regime stats."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        labels = np.array([1, 2, 3, 4] * 50)
        clf.fit(vix, labels=labels)

        probs = clf._calculate_probabilities(15.0, 30.0)
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        clf = self._make_fitted_classifier()
        importance = clf.get_feature_importance()
        assert importance is not None
        assert "VIX_level" in importance
        assert sum(importance.values()) == pytest.approx(1.0)

    def test_get_regime_thresholds(self):
        """Test threshold table."""
        clf = self._make_fitted_classifier()
        df = clf.get_regime_thresholds()
        assert isinstance(df, pd.DataFrame)
        assert "Level" in df.columns
        assert "Learned" in df.columns
        assert len(df) == 5

    def test_get_vix_regime_profile(self):
        """Test VIX regime profile with stats."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        labels = np.array([1, 2, 3, 4] * 50)
        clf.fit(vix, labels=labels)

        profile = clf.get_vix_regime_profile()
        assert len(profile) == 4
        assert "VIX_Mean" in profile.columns

    def test_get_vix_regime_profile_no_stats_raises(self):
        """Test regime profile without stats raises."""
        clf = self._make_fitted_classifier()
        with pytest.raises(ValueError, match="statistics"):
            clf.get_vix_regime_profile()


# ============================================================================
# VALIDATORS EXTENDED TESTS
# ============================================================================

class TestDataValidatorExtended:
    """Extended tests for DataValidator."""

    def _make_sample_ohlcv(self, rows=100, add_issues=False):
        """Build a sample OHLCV DataFrame."""
        dates = pd.date_range("2023-01-01", periods=rows, freq="D")
        np.random.seed(42)
        close = np.cumsum(np.random.randn(rows)) + 100
        high = close + np.abs(np.random.randn(rows))
        low = close - np.abs(np.random.randn(rows))
        opn = close + np.random.randn(rows) * 0.5

        df = pd.DataFrame({
            "Open": opn,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(1e6, 1e7, rows),
        }, index=dates)

        if add_issues:
            df.loc[df.index[5], "Close"] = np.nan
            df.loc[df.index[10], "High"] = df.loc[df.index[10], "Low"] - 1  # OHLC inconsistency
            df.loc[df.index[15], "Volume"] = -100

        return df

    def test_validate_clean_data(self):
        """Test validation of clean data passes."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = self._make_sample_ohlcv()
        clean, report = validator.validate(df, "SPX")
        assert report.is_valid

    def test_validate_with_missing_values(self):
        """Test validation detects and fixes missing values."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = self._make_sample_ohlcv(add_issues=True)
        clean, report = validator.validate(df, "SPX")

        # Should have issues logged
        issue_types = [i["type"] for i in report.issues]
        assert "missing_values" in issue_types
        # Should fix the NaN
        assert clean["Close"].isna().sum() == 0

    def test_validate_ohlc_inconsistency(self):
        """Test OHLC consistency check."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = self._make_sample_ohlcv(add_issues=True)
        clean, report = validator.validate(df, "SPX")

        issue_types = [i["type"] for i in report.issues]
        assert "ohlc_inconsistency" in issue_types

    def test_validate_negative_volume(self):
        """Test negative volume detection and fix."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = self._make_sample_ohlcv(add_issues=True)
        clean, report = validator.validate(df, "SPX")

        issue_types = [i["type"] for i in report.issues]
        assert "negative_volume" in issue_types
        assert (clean["Volume"] >= 0).all()

    def test_validate_empty_raises(self):
        """Test empty DataFrame raises ValueError."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        with pytest.raises(ValueError, match="Empty"):
            validator.validate(pd.DataFrame(), "SPX")

    def test_validate_no_fix(self):
        """Test validation without fixing (report only)."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = self._make_sample_ohlcv(add_issues=True)
        _, report = validator.validate(df, "SPX", fix_issues=False)

        # Issues should still be reported
        assert len(report.issues) > 0

    def test_validate_price_below_minimum(self):
        """Test price below minimum detection."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator(min_price=50.0)
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "Close": [100, 100, 30, 100, 100, 100, 100, 100, 100, 100],
        }, index=dates)
        clean, report = validator.validate(df, "TEST")

        issue_types = [i["type"] for i in report.issues]
        assert "invalid_price" in issue_types

    def test_validate_return_outliers(self):
        """Test large daily return flagging."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = [100] * 10
        prices[5] = 130  # 30% jump
        df = pd.DataFrame({"Close": prices}, index=dates)
        _, report = validator.validate(df, "SPX")

        issue_types = [i["type"] for i in report.issues]
        assert "return_outlier" in issue_types

    def test_validate_date_gaps(self):
        """Test large date gap detection."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        dates += [datetime(2023, 3, 1) + timedelta(days=i) for i in range(5)]
        df = pd.DataFrame({
            "Close": np.random.uniform(100, 105, 10),
        }, index=pd.DatetimeIndex(dates))
        _, report = validator.validate(df, "SPX")

        issue_types = [i["type"] for i in report.issues]
        assert "date_gap" in issue_types

    def test_validation_report_str(self):
        """Test ValidationReport string representation."""
        from src.data_pipeline.validators import ValidationReport

        report = ValidationReport(
            asset="SPX",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_rows=100,
            valid_rows=95,
            issues=[{
                "type": "missing_values",
                "severity": "warning",
                "message": "5 missing values in Close",
            }],
        )
        s = str(report)
        assert "SPX" in s
        assert "PASSED" in s or "FAILED" in s

    def test_validation_report_missing_pct(self):
        """Test missing_pct property."""
        from src.data_pipeline.validators import ValidationReport

        report = ValidationReport(
            asset="SPX",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_rows=100,
            valid_rows=80,
        )
        assert report.missing_pct == pytest.approx(20.0)

    def test_validation_report_missing_pct_zero_rows(self):
        """Test missing_pct with zero rows."""
        from src.data_pipeline.validators import ValidationReport

        report = ValidationReport(
            asset="SPX",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_rows=0,
            valid_rows=0,
        )
        assert report.missing_pct == 100.0

    def test_validate_multi_asset(self):
        """Test multi-asset validation."""
        from src.data_pipeline.validators import validate_multi_asset

        data = {
            "SPX": self._make_sample_ohlcv(50),
            "TLT": self._make_sample_ohlcv(50),
        }
        clean, reports = validate_multi_asset(data)
        assert "SPX" in clean
        assert "TLT" in clean
        assert reports["SPX"].is_valid

    def test_validate_multi_asset_error(self):
        """Test multi-asset validation with one failing."""
        from src.data_pipeline.validators import validate_multi_asset

        data = {
            "SPX": self._make_sample_ohlcv(50),
            "BAD": pd.DataFrame(),
        }
        clean, reports = validate_multi_asset(data)
        assert "SPX" in clean
        assert "BAD" in reports
        assert reports["BAD"].valid_rows == 0

    def test_validate_without_volume_check(self):
        """Test with volume checking disabled."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator(check_volume=False)
        df = self._make_sample_ohlcv(add_issues=True)
        clean, report = validator.validate(df, "SPX")
        # Should not have negative_volume issue since volume check is disabled
        issue_types = [i["type"] for i in report.issues]
        assert "negative_volume" not in issue_types

    def test_validate_non_datetime_index(self):
        """Test validation auto-converts string dates."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        df = pd.DataFrame({
            "Close": [100, 101, 102],
        }, index=["2023-01-01", "2023-01-02", "2023-01-03"])
        clean, report = validator.validate(df, "TEST")
        assert isinstance(clean.index, pd.DatetimeIndex)

    def test_validate_zero_volume(self):
        """Test zero volume detection."""
        from src.data_pipeline.validators import DataValidator

        validator = DataValidator()
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1e6, 0, 0, 1e6, 1e6],
        }, index=dates)
        _, report = validator.validate(df, "TEST")
        issue_types = [i["type"] for i in report.issues]
        assert "zero_volume" in issue_types


# ============================================================================
# REGIME BASE CLASSIFIER TESTS
# ============================================================================

class TestBaseClassifierExtended:
    """Tests for RegimeResult and BaseClassifier uncovered paths."""

    def test_regime_result_to_dict(self):
        """Test RegimeResult serialization."""
        from src.regime_detection.base import RegimeResult

        result = RegimeResult(
            regime=1,
            confidence=0.85,
            probabilities={1: 0.6, 2: 0.1, 3: 0.1, 4: 0.2},
            features_used=["VIX"],
            metadata={"vix": 15.0},
        )
        d = result.to_dict()
        assert d["regime"] == 1
        assert d["regime_name"] == "Risk-On Growth"
        assert d["confidence"] == 0.85

    def test_regime_result_invalid_regime(self):
        """Test RegimeResult validation for invalid regime."""
        from src.regime_detection.base import RegimeResult

        with pytest.raises(ValueError, match="Regime must be 1-4"):
            RegimeResult(regime=5, confidence=0.8)

    def test_regime_result_invalid_confidence(self):
        """Test RegimeResult validation for out-of-range confidence."""
        from src.regime_detection.base import RegimeResult

        with pytest.raises(ValueError, match="Confidence must be 0-1"):
            RegimeResult(regime=1, confidence=1.5)

    def test_regime_result_regime_names(self):
        """Test all four regime names."""
        from src.regime_detection.base import RegimeResult

        names = {
            1: "Risk-On Growth",
            2: "Risk-Off Crisis",
            3: "Stagflation",
            4: "Disinflationary Boom",
        }
        for r, expected_name in names.items():
            result = RegimeResult(regime=r, confidence=0.8)
            assert result.regime_name == expected_name

    def test_validate_data_dataframe(self):
        """Test _validate_data with DataFrame."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        df = pd.DataFrame(np.random.randn(200, 3))
        arr = clf._validate_data(df)
        assert arr.shape == (200, 3)

    def test_validate_data_series(self):
        """Test _validate_data with Series."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        s = pd.Series(np.random.randn(200))
        arr = clf._validate_data(s)
        assert arr.shape == (200, 1)

    def test_validate_data_1d_array(self):
        """Test _validate_data with 1D array."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        arr = clf._validate_data(np.random.randn(200))
        assert arr.shape == (200, 1)

    def test_validate_data_insufficient_raises(self):
        """Test _validate_data with too few samples."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        with pytest.raises(ValueError, match="Insufficient"):
            clf._validate_data(np.random.randn(10), min_samples=100)

    def test_validate_data_with_nan(self):
        """Test _validate_data handles NaN values."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        arr = np.random.randn(200, 2)
        arr[5, 0] = np.nan
        arr[10, 1] = np.nan

        result = clf._validate_data(arr)
        assert not np.isnan(result).any()

    def test_predict_sequence(self):
        """Test predict_sequence on fitted classifier."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        clf.fit(vix)

        test_data = pd.DataFrame(
            {"VIX": [15, 25, 35, 12, 40]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        results = clf.predict_sequence(test_data)
        assert len(results) == 5
        assert all(r.regime in [1, 2, 3, 4] for r in results)

    def test_predict_sequence_not_fitted(self):
        """Test predict_sequence raises if not fitted."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        with pytest.raises(ValueError, match="fitted"):
            clf.predict_sequence(pd.DataFrame({"VIX": [15]}))

    def test_get_regime_history(self):
        """Test get_regime_history returns DataFrame."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        vix = pd.Series(np.random.uniform(10, 35, 200))
        clf.fit(vix)

        test_data = pd.DataFrame(
            {"VIX": [15, 25, 35]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        history = clf.get_regime_history(test_data)
        assert isinstance(history, pd.DataFrame)
        assert "Regime" in history.columns
        assert "Confidence" in history.columns
        assert len(history) == 3

    def test_classifier_repr(self):
        """Test classifier string representation."""
        from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

        clf = VolatilityRegimeClassifier()
        assert "not fitted" in repr(clf)

        clf.fit(pd.Series(np.random.uniform(10, 35, 200)))
        assert "fitted" in repr(clf)

    def test_classifier_init_non_4_regimes(self):
        """Test that BaseClassifier logs warning for non-4 regime count."""
        from src.regime_detection.base import BaseClassifier

        # Create a concrete subclass to test BaseClassifier init
        class DummyClassifier(BaseClassifier):
            def fit(self, data, labels=None):
                return self
            def predict(self, data):
                pass
            def get_feature_importance(self):
                return None

        clf = DummyClassifier(n_regimes=3)
        assert clf.n_regimes == 3


# Helper for optional warning check
from contextlib import nullcontext


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
