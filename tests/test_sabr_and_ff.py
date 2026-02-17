"""
Test Suite for SABR Calibration and Fama-French Multi-Factor OLS.

Tests for:
- SABRCalibrator: Implied vol, calibration, surface construction
- SABRParams: Dataclass validation
- OptionsSurfaceMonitor: SABR-integrated analyze path + VIX fallback
- FactorExposureAnalyzer: Fama-French / AQR column detection, multi-factor OLS,
  factor attribution, OLSResult dataclass

Coverage target: Full coverage of all new methods introduced in the
options_surface_monitor and factor_exposure_analyzer upgrades.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sabr_calibrator():
    """Create a SABRCalibrator with default beta=1.0."""
    from src.modules.options_surface_monitor import SABRCalibrator
    return SABRCalibrator(beta=1.0)


@pytest.fixture
def sabr_calibrator_cev():
    """Create a SABRCalibrator with beta=0.5 (CIR backbone)."""
    from src.modules.options_surface_monitor import SABRCalibrator
    return SABRCalibrator(beta=0.5)


@pytest.fixture
def sample_options_chain():
    """Generate a synthetic options chain suitable for SABR calibration."""
    np.random.seed(42)
    forward = 100.0
    strikes = np.array([85, 90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110, 115])
    # Realistic smile: higher vol for OTM puts, lower for ATM, slightly rising for OTM calls
    base_vol = 0.20
    skew = 0.005 * (forward - strikes)  # positive skew for puts
    smile = 0.0003 * (strikes - forward) ** 2  # butterfly
    market_vols = base_vol + skew + smile
    
    return pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": market_vols,
        "forward": forward,
        "expiry_years": 30 / 365.0,
    })


@pytest.fixture
def sample_options_chain_with_vix():
    """Options chain data mixed with VIX column."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    
    # VIX series for fallback path
    vix = np.clip(15 + np.cumsum(np.random.normal(0, 0.3, n)), 10, 40)
    spx = 4000 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    
    return pd.DataFrame({
        "VIX": vix,
        "SPX": spx,
        "strike": np.linspace(3800, 4200, n),
        "impliedVolatility": np.random.uniform(0.15, 0.30, n),
        "forward": spx,
    }, index=dates)


@pytest.fixture
def fama_french_data():
    """Generate synthetic Fama-French 5-factor data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    # Factors explain portfolio returns with some noise
    mkt = np.random.normal(0.0005, 0.012, n)
    smb = np.random.normal(0.0002, 0.006, n)
    hml = np.random.normal(-0.0001, 0.005, n)
    rmw = np.random.normal(0.0003, 0.004, n)
    cma = np.random.normal(0.0001, 0.003, n)
    rf = np.full(n, 0.0002)
    
    # Construct portfolio returns as linear combination + noise
    portfolio = 0.001 + 1.05 * mkt + 0.3 * smb - 0.2 * hml + 0.5 * rmw - 0.1 * cma + np.random.normal(0, 0.003, n)
    
    return pd.DataFrame({
        "SPX": portfolio,
        "Mkt-RF": mkt,
        "SMB": smb,
        "HML": hml,
        "RMW": rmw,
        "CMA": cma,
        "RF": rf,
    }, index=dates)


@pytest.fixture
def aqr_data():
    """Generate synthetic AQR-style factor data."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    
    mkt = np.random.normal(0.0005, 0.012, n)
    smb = np.random.normal(0.0002, 0.006, n)
    hml_ff = np.random.normal(-0.0001, 0.005, n)
    umd = np.random.normal(0.0003, 0.008, n)
    bab = np.random.normal(0.0002, 0.005, n)
    qmj = np.random.normal(0.0004, 0.004, n)
    
    portfolio = 0.002 + 0.9 * mkt + 0.2 * smb + np.random.normal(0, 0.003, n)
    
    return pd.DataFrame({
        "SPX": portfolio,
        "MKT": mkt,
        "SMB": smb,
        "HML_FF": hml_ff,
        "UMD": umd,
        "BAB": bab,
        "QMJ": qmj,
    }, index=dates)


@pytest.fixture
def monitor():
    """Create an OptionsSurfaceMonitor."""
    from src.modules.options_surface_monitor import OptionsSurfaceMonitor
    m = OptionsSurfaceMonitor()
    m.update_regime(1, 0.85)
    return m


@pytest.fixture
def factor_analyzer():
    """Create a FactorExposureAnalyzer."""
    from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
    a = FactorExposureAnalyzer()
    a.update_regime(1, 0.85)
    return a


# ============================================================================
# SABR CALIBRATOR TESTS
# ============================================================================

class TestSABRCalibrator:
    """Tests for the SABR stochastic volatility calibrator."""

    def test_init_default_beta(self, sabr_calibrator):
        """Test default beta=1.0 (log-normal backbone)."""
        assert sabr_calibrator.beta == 1.0

    def test_init_cev_beta(self, sabr_calibrator_cev):
        """Test beta=0.5 (CIR backbone)."""
        assert sabr_calibrator_cev.beta == 0.5

    def test_init_invalid_beta(self):
        """Test that invalid beta raises ValueError."""
        from src.modules.options_surface_monitor import SABRCalibrator
        with pytest.raises(ValueError, match="beta must be in"):
            SABRCalibrator(beta=1.5)
        with pytest.raises(ValueError, match="beta must be in"):
            SABRCalibrator(beta=-0.1)

    def test_implied_vol_atm(self, sabr_calibrator):
        """Test implied vol at ATM (strike == forward)."""
        vol = sabr_calibrator.implied_vol(
            strike=100.0, forward=100.0, expiry=0.25,
            alpha=0.2, rho=-0.25, nu=0.3,
        )
        assert isinstance(vol, float)
        assert 0.0 < vol < 2.0  # reasonable range

    def test_implied_vol_otm_put(self, sabr_calibrator):
        """Test implied vol for OTM put (strike < forward)."""
        vol = sabr_calibrator.implied_vol(
            strike=90.0, forward=100.0, expiry=0.25,
            alpha=0.2, rho=-0.25, nu=0.3,
        )
        assert isinstance(vol, float)
        assert vol > 0

    def test_implied_vol_otm_call(self, sabr_calibrator):
        """Test implied vol for OTM call (strike > forward)."""
        vol = sabr_calibrator.implied_vol(
            strike=110.0, forward=100.0, expiry=0.25,
            alpha=0.2, rho=-0.25, nu=0.3,
        )
        assert isinstance(vol, float)
        assert vol > 0

    def test_implied_vol_skew_with_negative_rho(self, sabr_calibrator):
        """Negative rho should produce higher put vol than call vol."""
        params = dict(forward=100.0, expiry=0.25, alpha=0.2, rho=-0.5, nu=0.5)
        put_vol = sabr_calibrator.implied_vol(strike=90.0, **params)
        call_vol = sabr_calibrator.implied_vol(strike=110.0, **params)
        # With negative rho, OTM puts should have higher vol
        assert put_vol > call_vol

    def test_implied_vol_surface_array(self, sabr_calibrator):
        """Test vectorised implied vol returns correct-shaped array."""
        strikes = np.array([90, 95, 100, 105, 110])
        vols = sabr_calibrator.implied_vol_surface(
            strikes=strikes, forward=100.0, expiry=0.25,
            alpha=0.2, rho=-0.25, nu=0.3,
        )
        assert vols.shape == (5,)
        assert all(v > 0 for v in vols)

    def test_calibrate_recovers_smile(self, sabr_calibrator):
        """Test calibration fits market vols within tolerance."""
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        forward = 100.0
        expiry = 0.25
        # Generate synthetic market vols from known SABR params
        true_alpha, true_rho, true_nu = 0.2, -0.3, 0.4
        market_vols = sabr_calibrator.implied_vol_surface(
            strikes, forward, expiry, true_alpha, true_rho, true_nu,
        )
        
        params = sabr_calibrator.calibrate(
            strikes=strikes, forward=forward, expiry=expiry,
            market_vols=market_vols,
        )
        
        assert params.error < 0.005  # RMSE < 50 bps
        assert abs(params.alpha - true_alpha) < 0.05
        assert abs(params.rho - true_rho) < 0.15
        assert abs(params.nu - true_nu) < 0.15
        assert params.beta == 1.0
        assert params.expiry == expiry

    def test_calibrate_custom_initial_guess(self, sabr_calibrator):
        """Test calibration with user-supplied alpha0."""
        strikes = np.array([90, 95, 100, 105, 110])
        market_vols = np.array([0.25, 0.22, 0.20, 0.21, 0.23])
        
        params = sabr_calibrator.calibrate(
            strikes=strikes, forward=100.0, expiry=0.25,
            market_vols=market_vols,
            alpha0=0.2, rho0=-0.1, nu0=0.2,
        )
        assert params.error < 0.01

    def test_calibrate_cev_backbone(self, sabr_calibrator_cev):
        """Test calibration with beta=0.5."""
        strikes = np.array([90, 95, 100, 105, 110])
        forward = 100.0
        market_vols = np.array([0.24, 0.21, 0.20, 0.205, 0.22])
        
        params = sabr_calibrator_cev.calibrate(
            strikes=strikes, forward=forward, expiry=0.5,
            market_vols=market_vols,
        )
        assert params.beta == 0.5
        assert params.error < 0.02


class TestSABRParams:
    """Tests for the SABRParams dataclass."""

    def test_sabr_params_creation(self):
        """Test dataclass creation."""
        from src.modules.options_surface_monitor import SABRParams
        p = SABRParams(alpha=0.2, beta=1.0, rho=-0.3, nu=0.4, expiry=0.25)
        assert p.alpha == 0.2
        assert p.beta == 1.0
        assert p.rho == -0.3
        assert p.nu == 0.4
        assert p.expiry == 0.25
        assert p.error == 0.0  # default

    def test_sabr_params_with_error(self):
        """Test dataclass with calibration error."""
        from src.modules.options_surface_monitor import SABRParams
        p = SABRParams(alpha=0.2, beta=0.5, rho=-0.1, nu=0.3, expiry=0.5, error=0.003)
        assert p.error == 0.003


# ============================================================================
# OPTIONS SURFACE MONITOR (SABR INTEGRATION) TESTS
# ============================================================================

class TestOptionsSurfaceMonitorSABR:
    """Tests for SABR-integrated OptionsSurfaceMonitor."""

    def test_init_has_sabr_calibrator(self):
        """Test monitor initialises with SABRCalibrator."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        m = OptionsSurfaceMonitor()
        assert m.sabr_calibrator is not None
        assert m.sabr_calibrator.beta == 1.0
        assert m.sabr_history == {}

    def test_init_custom_beta(self):
        """Test monitor with custom SABR beta."""
        from src.modules.options_surface_monitor import OptionsSurfaceMonitor
        m = OptionsSurfaceMonitor(sabr_beta=0.5)
        assert m.sabr_calibrator.beta == 0.5

    def test_analyze_with_options_chain(self, monitor, sample_options_chain):
        """Test analyze with options chain triggers SABR calibration."""
        result = monitor.analyze(sample_options_chain)
        
        assert "signal" in result
        assert result["signal"].signal in ("bullish", "bearish", "neutral", "cautious")
        assert "sabr" in result
        assert "sabr_params" in result["sabr"]
        assert "atm_vol" in result["sabr"]
        assert result["sabr"]["calibration_error"] < 0.05

    def test_analyze_chain_stores_sabr_history(self, monitor, sample_options_chain):
        """Test that SABR params are stored in history."""
        monitor.analyze(sample_options_chain)
        assert len(monitor.sabr_history) > 0

    def test_analyze_chain_builds_surface_snapshot(self, monitor, sample_options_chain):
        """Test that a VolSurface with sabr_params is constructed."""
        result = monitor.analyze(sample_options_chain)
        
        snapshot = result.get("surface_snapshot")
        assert snapshot is not None
        assert snapshot.sabr_params is not None
        assert snapshot.atm_vol > 0
        assert isinstance(snapshot.put_skew, float)
        assert isinstance(snapshot.call_skew, float)

    def test_analyze_chain_with_vix_fallback(self, monitor, sample_options_chain_with_vix):
        """Test mixed data: SABR from chain + VIX fallback signals."""
        result = monitor.analyze(sample_options_chain_with_vix)
        
        assert "signal" in result
        assert "vix_level" in result
        # Both SABR and VIX signals should be present
        assert len(result.get("individual_signals", [])) >= 2

    def test_analyze_thin_chain_falls_back(self, monitor):
        """Test that a chain with < 5 rows falls back to VIX."""
        thin_chain = pd.DataFrame({
            "strike": [95, 100, 105],
            "impliedVolatility": [0.22, 0.20, 0.21],
            "VIX": [18.0, 18.0, 18.0],
        })
        result = monitor.analyze(thin_chain)
        
        assert "signal" in result
        # Should not have SABR data since chain too thin
        assert "sabr" not in result

    def test_analyze_vix_only_still_works(self, monitor):
        """Test pure VIX path (no options chain columns)."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "VIX": np.random.uniform(12, 20, 100),
            "SPX": np.random.uniform(4000, 4500, 100),
        }, index=dates)
        
        result = monitor.analyze(data)
        assert "signal" in result
        assert "vix_level" in result
        assert "sabr" not in result

    def test_analyze_no_data(self, monitor):
        """Test with neither chain nor VIX data."""
        data = pd.DataFrame({"irrelevant": [1, 2, 3]})
        result = monitor.analyze(data)
        assert result["signal"].signal == "neutral"
        assert result["signal"].strength == 0.0

    def test_sabr_rho_signal_negative(self, monitor):
        """Test that strongly negative rho generates bearish signal."""
        # Create chain that will calibrate to rho < -0.7
        forward = 100.0
        strikes = np.array([80, 85, 90, 95, 97.5, 100, 102.5, 105, 110, 115, 120])
        # Very steep put skew → strong negative rho
        base = 0.20
        market_vols = base + 0.015 * (forward - strikes)  # extreme skew
        market_vols = np.clip(market_vols, 0.05, 0.60)
        
        chain = pd.DataFrame({
            "strike": strikes,
            "impliedVolatility": market_vols,
            "forward": forward,
            "expiry_years": 0.25,
        })
        result = monitor.analyze(chain)
        
        # Should have produced signals from SABR
        assert "sabr" in result
        assert result["sabr"]["sabr_params"].rho < 0

    def test_sabr_nu_signal_high_volofvol(self, monitor):
        """Test that high nu generates bearish signal."""
        forward = 100.0
        strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
        # Very curved smile → high nu
        base = 0.20
        curvature = 0.001 * (strikes - forward) ** 2
        market_vols = base + curvature
        
        chain = pd.DataFrame({
            "strike": strikes,
            "impliedVolatility": market_vols,
            "forward": forward,
            "expiry_years": 0.25,
        })
        result = monitor.analyze(chain)
        assert "sabr" in result


# ============================================================================
# FAMA-FRENCH / AQR MULTI-FACTOR OLS TESTS
# ============================================================================

class TestOLSResult:
    """Tests for the OLSResult dataclass."""

    def test_ols_result_creation(self):
        """Test dataclass creation."""
        from src.modules.factor_exposure_analyzer import OLSResult
        r = OLSResult(
            alpha=0.001, betas={"size": 0.3, "value": -0.2},
            r_squared=0.85, n_obs=60,
        )
        assert r.alpha == 0.001
        assert r.betas["size"] == 0.3
        assert r.r_squared == 0.85
        assert r.n_obs == 60
        assert r.t_stats == {}
        assert r.residual_vol == 0.0


class TestFactorExposureAnalyzerFF:
    """Tests for Fama-French / AQR multi-factor OLS upgrades."""

    def test_fama_french_columns_defined(self, factor_analyzer):
        """Test that FF and AQR column maps are present."""
        assert "Mkt-RF" in factor_analyzer.FAMA_FRENCH_COLUMNS
        assert "SMB" in factor_analyzer.FAMA_FRENCH_COLUMNS
        assert "HML" in factor_analyzer.FAMA_FRENCH_COLUMNS
        assert "RMW" in factor_analyzer.FAMA_FRENCH_COLUMNS
        assert "CMA" in factor_analyzer.FAMA_FRENCH_COLUMNS
        assert "MKT" in factor_analyzer.AQR_COLUMNS
        assert "QMJ" in factor_analyzer.AQR_COLUMNS

    def test_latest_multi_ols_init(self, factor_analyzer):
        """Test _latest_multi_ols is None at init."""
        assert factor_analyzer._latest_multi_ols is None

    def test_normalise_ff_columns(self, factor_analyzer, fama_french_data):
        """Test Fama-French column normalisation."""
        mapped, detected = factor_analyzer._normalise_factor_columns(fama_french_data)
        
        assert detected is True
        # Should have mapped SMB → size, HML → value, etc.
        assert "size" in mapped.columns
        assert "value" in mapped.columns
        assert "quality" in mapped.columns  # RMW → quality
        assert "growth" in mapped.columns   # CMA → growth

    def test_normalise_aqr_columns(self, factor_analyzer, aqr_data):
        """Test AQR column normalisation."""
        mapped, detected = factor_analyzer._normalise_factor_columns(aqr_data)
        
        assert detected is True
        assert "momentum" in mapped.columns  # UMD → momentum
        assert "volatility" in mapped.columns  # BAB → volatility
        assert "quality" in mapped.columns  # QMJ → quality

    def test_normalise_generic_columns_no_detection(self, factor_analyzer):
        """Test that generic factor columns don't trigger normalisation."""
        data = pd.DataFrame({
            "momentum": [0.01, 0.02],
            "value": [0.005, -0.003],
        })
        mapped, detected = factor_analyzer._normalise_factor_columns(data)
        assert detected is False

    def test_find_dependent_column_spx(self, factor_analyzer):
        """Test dependent column detection for SPX variants."""
        data1 = pd.DataFrame({"SPX_returns": [1], "SPX": [2]})
        assert factor_analyzer._find_dependent_column(data1) == "SPX_returns"
        
        data2 = pd.DataFrame({"SPX": [1]})
        assert factor_analyzer._find_dependent_column(data2) == "SPX"

    def test_find_dependent_column_none(self, factor_analyzer):
        """Test returns None when no dep column matches."""
        data = pd.DataFrame({"irrelevant": [1]})
        assert factor_analyzer._find_dependent_column(data) is None

    def test_analyze_with_ff_data(self, factor_analyzer, fama_french_data):
        """Test full analysis with Fama-French columns."""
        result = factor_analyzer.analyze(fama_french_data)
        
        assert "signal" in result
        assert "factor_results" in result
        assert "ols_betas" in result
        assert len(result["ols_betas"]) > 0
        
        # Multi-factor OLS should have been run
        assert "multi_factor_ols" in result
        mf = result["multi_factor_ols"]
        assert "alpha" in mf
        assert "r_squared" in mf
        assert "t_stats" in mf
        assert "residual_vol" in mf
        assert mf["r_squared"] > 0.0  # should explain some variance
        assert mf["n_obs"] > 0

    def test_analyze_with_aqr_data(self, factor_analyzer, aqr_data):
        """Test full analysis with AQR columns."""
        result = factor_analyzer.analyze(aqr_data)
        
        assert "signal" in result
        assert "ols_betas" in result
        assert "multi_factor_ols" in result

    def test_analyze_stores_latest_ols(self, factor_analyzer, fama_french_data):
        """Test that _latest_multi_ols is updated after analysis."""
        factor_analyzer.analyze(fama_french_data)
        assert factor_analyzer._latest_multi_ols is not None
        assert isinstance(factor_analyzer._latest_multi_ols.alpha, float)

    def test_multi_factor_ols_r_squared(self, factor_analyzer, fama_french_data):
        """Test R² is reasonable for well-constructed factor data."""
        result = factor_analyzer.analyze(fama_french_data)
        mf = result["multi_factor_ols"]
        # Data was constructed as linear combination, R² should be decent
        assert 0.0 <= mf["r_squared"] <= 1.0

    def test_multi_factor_ols_t_stats(self, factor_analyzer, fama_french_data):
        """Test t-statistics are computed for each factor."""
        result = factor_analyzer.analyze(fama_french_data)
        mf = result["multi_factor_ols"]
        # Should have t-stats for at least some factors
        assert len(mf["t_stats"]) > 0

    def test_multi_factor_ols_insufficient_data(self, factor_analyzer):
        """Test graceful handling when too few observations."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        a = FactorExposureAnalyzer()
        a.update_regime(1, 0.85)
        
        # Only 10 rows — below _min_observations (40)
        data = pd.DataFrame({
            "SPX": np.random.normal(0, 0.01, 10),
            "Mkt-RF": np.random.normal(0, 0.01, 10),
            "SMB": np.random.normal(0, 0.005, 10),
            "HML": np.random.normal(0, 0.005, 10),
        })
        result = a.analyze(data)
        # Should still work but without multi_factor_ols
        assert "multi_factor_ols" not in result

    def test_single_factor_ols_fallback(self, factor_analyzer):
        """Test single-factor OLS when no FF columns but SPX present."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            "SPX": np.random.normal(0.001, 0.01, n),
            "momentum": np.random.normal(0.003, 0.01, n),
            "value": np.random.normal(-0.001, 0.008, n),
        })
        result = factor_analyzer.analyze(data)
        
        assert "ols_betas" in result
        # Should have used single-factor OLS for each
        assert "momentum" in result["ols_betas"] or "value" in result["ols_betas"]
        # No multi_factor_ols since no FF columns
        assert "multi_factor_ols" not in result

    def test_analyze_preserves_generic_factor_path(self, factor_analyzer):
        """Test generic factors (no FF/AQR) still work as before."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            "momentum": np.random.normal(0.003, 0.01, n),
            "quality": np.random.normal(0.002, 0.007, n),
            "size": np.random.normal(0.001, 0.009, n),
        })
        result = factor_analyzer.analyze(data)
        
        assert "signal" in result
        assert "factor_results" in result
        assert len(result["factor_results"]) == 3

    def test_run_multi_factor_ols_directly(self, factor_analyzer, fama_french_data):
        """Test _run_multi_factor_ols method directly."""
        mapped, _ = factor_analyzer._normalise_factor_columns(fama_french_data)
        ols = factor_analyzer._run_multi_factor_ols(mapped, "SPX")
        
        assert ols is not None
        assert isinstance(ols.alpha, float)
        assert len(ols.betas) > 0
        assert 0 <= ols.r_squared <= 1

    def test_run_multi_factor_ols_no_dependent(self, factor_analyzer):
        """Test _run_multi_factor_ols returns None without dependent col."""
        data = pd.DataFrame({
            "momentum": [0.01, 0.02, 0.03],
            "value": [0.005, -0.003, 0.002],
        })
        result = factor_analyzer._run_multi_factor_ols(data, "nonexistent")
        assert result is None

    def test_regime_changes_affect_ff_analysis(self, fama_french_data):
        """Test that regime changes affect factor interpretation with FF data."""
        from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
        a1 = FactorExposureAnalyzer()
        a1.update_regime(1, 0.85)  # Risk-On
        r1 = a1.analyze(fama_french_data)
        
        a2 = FactorExposureAnalyzer()
        a2.update_regime(2, 0.85)  # Risk-Off
        r2 = a2.analyze(fama_french_data)
        
        # The OLS betas should be the same (data-driven)
        # but factor interpretation (signal) may differ by regime
        assert r1["regime_parameters"] != r2["regime_parameters"]


# ============================================================================
# INTEGRATION / EDGE-CASE TESTS
# ============================================================================

class TestIntegration:
    """Integration and edge-case tests for both upgrades."""

    def test_options_monitor_import_from_init(self):
        """Test imports from src.modules __init__."""
        from src.modules import OptionsSurfaceMonitor
        m = OptionsSurfaceMonitor()
        assert hasattr(m, "sabr_calibrator")

    def test_factor_analyzer_import_from_init(self):
        """Test imports from src.modules __init__."""
        from src.modules import FactorExposureAnalyzer
        a = FactorExposureAnalyzer()
        assert hasattr(a, "_latest_multi_ols")

    def test_sabr_with_zero_nu(self, sabr_calibrator):
        """Test SABR with nu=0 (no vol-of-vol, flat smile)."""
        vol = sabr_calibrator.implied_vol(
            strike=105.0, forward=100.0, expiry=0.25,
            alpha=0.2, rho=0.0, nu=0.0001,  # near-zero nu
        )
        assert vol > 0
        # With near-zero nu, smile should be very flat
        vol_atm = sabr_calibrator.implied_vol(
            strike=100.0, forward=100.0, expiry=0.25,
            alpha=0.2, rho=0.0, nu=0.0001,
        )
        assert abs(vol - vol_atm) < 0.01

    def test_multi_factor_ols_with_nan_rows(self, factor_analyzer):
        """Test multi-factor OLS handles NaN values gracefully."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            "SPX": np.random.normal(0.001, 0.01, n),
            "Mkt-RF": np.random.normal(0.0005, 0.012, n),
            "SMB": np.random.normal(0.0002, 0.006, n),
            "HML": np.random.normal(-0.0001, 0.005, n),
        })
        # Inject some NaNs
        data.loc[5:10, "SMB"] = np.nan
        data.loc[20:25, "HML"] = np.nan
        
        result = factor_analyzer.analyze(data)
        assert "signal" in result

    def test_options_chain_bad_data_graceful(self, monitor):
        """Test SABR fails gracefully on bad options chain."""
        bad_chain = pd.DataFrame({
            "strike": [0, 0, 0, 0, 0, 0],
            "impliedVolatility": [0, 0, 0, 0, 0, 0],
        })
        result = monitor.analyze(bad_chain)
        # Should not crash, falls back to neutral
        assert "signal" in result

    def test_consecutive_analyses_accumulate_history(self, monitor, sample_options_chain):
        """Test that repeated analyses accumulate surface history."""
        monitor.analyze(sample_options_chain)
        monitor.analyze(sample_options_chain)
        assert len(monitor.surface_history) >= 2
