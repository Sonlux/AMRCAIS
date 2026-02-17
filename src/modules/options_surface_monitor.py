"""
Options Surface Monitor for AMRCAIS.

Monitors implied volatility surfaces, skew dynamics, and term structure
to extract regime-adaptive signals about market sentiment and risk pricing.

Includes SABR stochastic volatility model (Hagan et al., 2002) for
calibrating a smooth volatility surface from market-observed option
prices.  When a full options chain is available the monitor builds
the surface via SABR; otherwise it falls back to VIX-proxy analytics.

The options market often leads the underlying - skew changes can signal
sentiment shifts before they appear in price. However, interpretation
is regime-dependent:

- Risk-On Growth: Put skew flattening → bullish continuation
- Risk-Off Crisis: Put skew steepening → fear, but may signal capitulation
- Stagflation: Call skew rising → inflation hedging
- Disinflationary Boom: Low vol, flat skew → Goldilocks

Classes:
    SABRCalibrator: SABR stochastic volatility model calibration
    OptionsSurfaceMonitor: Regime-adaptive options surface analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SABR Stochastic Volatility Model (Hagan et al., 2002)
# ---------------------------------------------------------------------------

@dataclass
class SABRParams:
    """Calibrated SABR parameters for a single expiry slice.

    Attributes:
        alpha: Initial volatility level (ATM vol backbone).
        beta: CEV exponent – typically fixed (1.0 for log-normal, 0.5 for CIR).
        rho: Correlation between forward and vol Brownian motions (−1,1).
        nu: Vol-of-vol – controls smile curvature.
        expiry: Time to expiry in years that the parameters were fitted for.
        error: Calibration RMSE (vol points).
    """
    alpha: float
    beta: float
    rho: float
    nu: float
    expiry: float
    error: float = 0.0


class SABRCalibrator:
    """SABR implied-volatility model.

    Implements the Hagan et al. (2002) closed-form approximation for
    implied Black volatility under the SABR stochastic-volatility model,
    together with a least-squares calibrator that fits ``(alpha, rho, nu)``
    to market-observed implied vols while holding ``beta`` fixed.

    Example:
        >>> cal = SABRCalibrator(beta=1.0)
        >>> params = cal.calibrate(
        ...     strikes=np.array([90, 95, 100, 105, 110]),
        ...     forward=100.0,
        ...     expiry=0.25,
        ...     market_vols=np.array([0.22, 0.20, 0.18, 0.19, 0.21]),
        ... )
        >>> print(f"alpha={params.alpha:.4f}, rho={params.rho:.4f}, nu={params.nu:.4f}")
    """

    def __init__(self, beta: float = 1.0) -> None:
        """Initialise with a fixed CEV exponent.

        Args:
            beta: CEV exponent (0 ≤ beta ≤ 1).  Common choices:
                  1.0 for equity/FX (log-normal backbone),
                  0.5 for rates (CIR-like backbone).
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.beta = beta

    # ---- Hagan closed-form implied vol -----------------------------------

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> float:
        """Compute SABR implied Black volatility (Hagan approximation).

        Args:
            strike: Option strike price.
            forward: Forward price of the underlying.
            expiry: Time to expiry in years.
            alpha: SABR alpha parameter.
            rho: SABR rho parameter (−1, 1).
            nu: SABR nu (vol-of-vol).

        Returns:
            Implied Black volatility (decimal, e.g. 0.20 for 20 %).
        """
        beta = self.beta
        eps = 1e-12  # guard divisions by zero

        if abs(forward - strike) < eps:
            # ATM limit
            fk = forward ** (1 - beta)
            term1 = ((1 - beta) ** 2 / 24) * alpha ** 2 / (fk ** 2 + eps)
            term2 = 0.25 * rho * beta * nu * alpha / (fk + eps)
            term3 = (2 - 3 * rho ** 2) / 24 * nu ** 2
            return alpha / (fk + eps) * (1 + (term1 + term2 + term3) * expiry)

        fk_mid = (forward * strike) ** ((1 - beta) / 2)
        log_fk = np.log(forward / strike)

        z = (nu / (alpha + eps)) * fk_mid * log_fk
        sqrt_term = np.sqrt(1 - 2 * rho * z + z ** 2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho + eps))
        if abs(x_z) < eps:
            x_z = 1.0  # z / x(z) → 1 when z → 0
            z_over_x = 1.0
        else:
            z_over_x = z / x_z

        denom = fk_mid * (
            1
            + (1 - beta) ** 2 / 24 * log_fk ** 2
            + (1 - beta) ** 4 / 1920 * log_fk ** 4
        )

        term1 = ((1 - beta) ** 2 / 24) * alpha ** 2 / (fk_mid ** 2 + eps)
        term2 = 0.25 * rho * beta * nu * alpha / (fk_mid + eps)
        term3 = (2 - 3 * rho ** 2) / 24 * nu ** 2

        return (alpha / (denom + eps)) * z_over_x * (
            1 + (term1 + term2 + term3) * expiry
        )

    def implied_vol_surface(
        self,
        strikes: np.ndarray,
        forward: float,
        expiry: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> np.ndarray:
        """Vectorised SABR implied vol for an array of strikes.

        Args:
            strikes: Array of strike prices.
            forward: Forward price.
            expiry: Time to expiry in years.
            alpha: SABR alpha.
            rho: SABR rho.
            nu: SABR nu.

        Returns:
            Array of implied Black volatilities.
        """
        return np.array([
            self.implied_vol(float(k), forward, expiry, alpha, rho, nu)
            for k in strikes
        ])

    # ---- Calibration (least-squares) -------------------------------------

    def calibrate(
        self,
        strikes: np.ndarray,
        forward: float,
        expiry: float,
        market_vols: np.ndarray,
        alpha0: Optional[float] = None,
        rho0: float = -0.25,
        nu0: float = 0.3,
    ) -> SABRParams:
        """Fit (alpha, rho, nu) to market implied vols for a single expiry.

        Uses ``scipy.optimize.minimize`` (L-BFGS-B) with bounds on each
        parameter.

        Args:
            strikes: Observed strike prices.
            forward: Forward price of the underlying.
            expiry: Time to expiry in years.
            market_vols: Observed Black implied vols (decimal).
            alpha0: Initial guess for alpha.  Defaults to ATM vol * forward^(1-beta).
            rho0: Initial guess for rho (default −0.25).
            nu0: Initial guess for nu (default 0.3).

        Returns:
            Fitted SABRParams dataclass.
        """
        if alpha0 is None:
            # Rough ATM vol seed
            atm_idx = int(np.argmin(np.abs(strikes - forward)))
            alpha0 = float(market_vols[atm_idx]) * forward ** (1 - self.beta)

        def objective(params: np.ndarray) -> float:
            a, r, n = params
            model_vols = self.implied_vol_surface(
                strikes, forward, expiry, a, r, n,
            )
            return float(np.sum((model_vols - market_vols) ** 2))

        result = minimize(
            objective,
            x0=[alpha0, rho0, nu0],
            method="L-BFGS-B",
            bounds=[
                (1e-6, 5.0),       # alpha
                (-0.999, 0.999),   # rho
                (1e-4, 5.0),       # nu
            ],
        )

        a_opt, r_opt, n_opt = result.x
        rmse = float(np.sqrt(result.fun / len(strikes)))
        logger.info(
            "SABR calibrated T=%.3f: alpha=%.4f rho=%.4f nu=%.4f RMSE=%.5f",
            expiry, a_opt, r_opt, n_opt, rmse,
        )

        return SABRParams(
            alpha=float(a_opt),
            beta=self.beta,
            rho=float(r_opt),
            nu=float(n_opt),
            expiry=expiry,
            error=rmse,
        )


@dataclass
class VolSurface:
    """Volatility surface snapshot.
    
    Attributes:
        timestamp: Snapshot time
        atm_vol: At-the-money implied volatility
        put_skew: 25-delta put skew (25d put - ATM)
        call_skew: 25-delta call skew (25d call - ATM)
        term_structure: Dict mapping expiry to ATM vol
        butterfly: Volatility smile curvature
        sabr_params: SABR calibration parameters (when available)
    """
    timestamp: datetime
    atm_vol: float
    put_skew: float
    call_skew: float
    term_structure: Dict[int, float]  # days to expiry -> vol
    butterfly: float = 0.0
    sabr_params: Optional[SABRParams] = None


class OptionsSurfaceMonitor(AnalyticalModule):
    """Monitors options implied volatility surfaces.
    
    Options prices embed forward-looking information about expected
    moves and tail risks. This module extracts signals from:
    
    1. ATM Volatility Level: Overall uncertainty
    2. Put Skew: Downside protection demand
    3. Call Skew: Upside speculation
    4. Term Structure: Near vs. far uncertainty
    5. Butterfly: Tail risk pricing
    
    REGIME-ADAPTIVE INTERPRETATION:
    
    Risk-On Growth (1):
        - Low vol + flat skew → bullish continuation
        - Rising put skew → warning sign
        - Vol contango normal
    
    Risk-Off Crisis (2):
        - High vol + steep put skew → fear
        - Extreme put skew → potential capitulation signal
        - Vol backwardation typical
    
    Stagflation (3):
        - Rising call skew → inflation hedging
        - Elevated vol baseline
        - Watch commodity options
    
    Disinflationary Boom (4):
        - Very low vol + flat skew → Goldilocks
        - Complacency risk if too low
    
    Attributes:
        surface_history: List of historical surface snapshots
        
    Example:
        >>> monitor = OptionsSurfaceMonitor()
        >>> monitor.update_regime(2, 0.85)  # Risk-Off Crisis
        >>> 
        >>> result = monitor.analyze_skew(put_skew=8.5, atm_vol=28)
        >>> print(result["signal"].signal)  # Regime-specific interpretation
    """
    
    # Regime-specific vol level interpretations
    VOL_LEVELS = {
        1: {"low": 12, "normal": 16, "elevated": 22, "high": 28},
        2: {"low": 20, "normal": 30, "elevated": 40, "high": 50},
        3: {"low": 16, "normal": 22, "elevated": 28, "high": 35},
        4: {"low": 10, "normal": 14, "elevated": 18, "high": 24},
    }
    
    # Put skew interpretation by regime
    PUT_SKEW_INTERPRETATION = {
        1: {
            "flattening": {"signal": "bullish", "explanation": "Reduced downside fear in growth regime"},
            "steepening": {"signal": "cautious", "explanation": "Rising downside protection demand"},
            "extreme": {"signal": "bearish", "explanation": "Significant fear emerging"},
        },
        2: {
            "flattening": {"signal": "bullish", "explanation": "Fear subsiding, possible recovery"},
            "steepening": {"signal": "neutral", "explanation": "Continued fear, expected in crisis"},
            "extreme": {"signal": "bullish", "explanation": "Potential capitulation signal"},
        },
        3: {
            "flattening": {"signal": "neutral", "explanation": "Downside fear stable"},
            "steepening": {"signal": "bearish", "explanation": "Stagflation fears intensifying"},
            "extreme": {"signal": "bearish", "explanation": "Severe stagflation concerns"},
        },
        4: {
            "flattening": {"signal": "bullish", "explanation": "Goldilocks vol environment"},
            "steepening": {"signal": "cautious", "explanation": "Some concern emerging"},
            "extreme": {"signal": "bearish", "explanation": "Market turning defensive"},
        },
    }
    
    def __init__(self, config_path: Optional[str] = None, sabr_beta: float = 1.0):
        """Initialize the options surface monitor.

        Args:
            config_path: Path to YAML configuration file.
            sabr_beta: Fixed CEV exponent for the SABR calibrator
                       (1.0 = log-normal, 0.5 = CIR backbone).
        """
        super().__init__(name="OptionsSurfaceMonitor", config_path=config_path)
        
        self.surface_history: List[VolSurface] = []
        self.sabr_calibrator = SABRCalibrator(beta=sabr_beta)
        self.sabr_history: Dict[float, SABRParams] = {}  # expiry -> latest params
        
        # Thresholds
        self.extreme_skew_threshold = 10.0  # vol points
        self.skew_change_threshold = 2.0
    
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get options parameters for a specific regime."""
        return {
            "vol_levels": self.VOL_LEVELS.get(regime, {}),
            "put_skew_rules": self.PUT_SKEW_INTERPRETATION.get(regime, {}),
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze options surface data.
        
        If an options chain is embedded in *data* (columns ``strike``,
        ``impliedVolatility``, ``expiry_years``, and optionally ``forward``),
        the SABR model is calibrated and a proper vol surface is
        constructed.  Otherwise the analyser falls back to VIX-proxy
        analytics.
        
        Args:
            data: DataFrame with VIX, vol surface data, or options chain.
            
        Returns:
            Comprehensive options surface analysis.
        """
        signals: List[ModuleSignal] = []
        surface_snapshot: Optional[VolSurface] = None
        sabr_result: Optional[Dict[str, Any]] = None

        # ----- Path A: Options-chain + SABR calibration -------------------
        chain_cols = {"strike", "impliedVolatility"}
        if chain_cols.issubset(set(data.columns)):
            sabr_result = self._analyze_options_chain(data)
            if sabr_result is not None:
                signals.extend(sabr_result.get("signals", []))
                surface_snapshot = sabr_result.get("surface_snapshot")
        
        # ----- Path B (fallback): VIX-proxy analytics ---------------------
        vix = None
        if "VIX" in data.columns:
            vix_series = data["VIX"].dropna()
            vix = float(vix_series.iloc[-1]) if len(vix_series) > 0 else None
        elif "VIXCLS" in data.columns:
            vix_series = data["VIXCLS"].dropna()
            vix = float(vix_series.iloc[-1]) if len(vix_series) > 0 else None
        
        if vix is not None:
            vol_signal = self._analyze_vol_level(vix)
            signals.append(vol_signal)
        
        # Compute put skew proxy from VIX dynamics
        put_skew_value = None
        if vix is not None and ("VIX" in data.columns or "VIXCLS" in data.columns):
            vix_col = "VIX" if "VIX" in data.columns else "VIXCLS"
            vix_series = data[vix_col].dropna()
            if len(vix_series) >= 20:
                vix_mean = float(vix_series.iloc[-60:].mean()) if len(vix_series) >= 60 else float(vix_series.mean())
                put_skew_value = vix - vix_mean
                
                prior_vix = float(vix_series.iloc[-2]) if len(vix_series) >= 2 else None
                prior_skew = (prior_vix - vix_mean) if prior_vix is not None else None
                
                skew_result = self.analyze_skew(
                    put_skew=put_skew_value,
                    atm_vol=vix,
                    prior_put_skew=prior_skew,
                )
                signals.append(skew_result["signal"])
        
        # Analyze term structure using VIX vs realized vol
        if vix is not None and "SPX" in data.columns:
            spx = data["SPX"].dropna()
            if len(spx) >= 20:
                realized_vol = float(spx.pct_change().iloc[-20:].std() * np.sqrt(252) * 100)
                term_result = self.analyze_term_structure(
                    near_vol=realized_vol,
                    far_vol=vix,
                    near_days=20,
                    far_days=30,
                )
                signals.append(term_result["signal"])
        
        # Use SABR skew if chain was available; else VIX-proxy
        if surface_snapshot is None and vix is not None:
            surface_snapshot = VolSurface(
                timestamp=datetime.now(),
                atm_vol=vix,
                put_skew=put_skew_value or 0.0,
                call_skew=-(put_skew_value or 0.0) * 0.5,
                term_structure={30: vix},
                butterfly=abs(put_skew_value or 0.0) * 0.3,
            )
        
        if surface_snapshot is not None:
            self.surface_history.append(surface_snapshot)
        
        if signals:
            avg_strength = np.mean([s.strength for s in signals])
            bullish = sum(1 for s in signals if s.signal == "bullish")
            bearish = sum(1 for s in signals if s.signal == "bearish")
            
            overall = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
            
            result: Dict[str, Any] = {
                "signal": self.create_signal(
                    signal=overall,
                    strength=avg_strength,
                    explanation=f"Options surface analysis based on {len(signals)} metrics",
                ),
                "vix_level": vix,
                "put_skew": put_skew_value,
                "surface_snapshot": surface_snapshot,
                "individual_signals": signals,
                "regime_parameters": self.get_current_parameters(),
            }
            if sabr_result is not None:
                result["sabr"] = {
                    k: v for k, v in sabr_result.items()
                    if k not in ("signals", "surface_snapshot")
                }
            return result
        
        return {
            "signal": self.create_signal(
                signal="neutral",
                strength=0.0,
                explanation="Insufficient options data",
            ),
        }

    # ---- SABR options-chain analysis -------------------------------------

    def _analyze_options_chain(
        self, data: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        """Calibrate SABR to an options chain and extract surface metrics.

        Expects *data* to contain at minimum ``strike`` and
        ``impliedVolatility`` columns.  ``expiry_years`` (float, years
        to expiry) and ``forward`` (forward price) are used when present;
        defaults are applied otherwise.

        Args:
            data: DataFrame with options chain data.

        Returns:
            Dictionary with SABR params, surface snapshot, and signals,
            or *None* if calibration fails.
        """
        try:
            chain = data.dropna(subset=["strike", "impliedVolatility"])
            if len(chain) < 5:
                logger.warning("Options chain too thin (%d rows) for SABR", len(chain))
                return None

            strikes = chain["strike"].values.astype(float)
            market_vols = chain["impliedVolatility"].values.astype(float)

            # Determine forward price
            if "forward" in chain.columns:
                forward = float(chain["forward"].iloc[0])
            elif "SPX" in chain.columns:
                forward = float(chain["SPX"].iloc[-1])
            else:
                forward = float(np.median(strikes))

            # Determine expiry
            if "expiry_years" in chain.columns:
                expiry = float(chain["expiry_years"].iloc[0])
            else:
                expiry = 30 / 365.0  # default ~1 month

            # ----- Calibrate SABR -----
            params = self.sabr_calibrator.calibrate(
                strikes=strikes,
                forward=forward,
                expiry=expiry,
                market_vols=market_vols,
            )
            self.sabr_history[expiry] = params

            # ----- Extract surface metrics from fitted model -----
            atm_vol_decimal = self.sabr_calibrator.implied_vol(
                forward, forward, expiry, params.alpha, params.rho, params.nu,
            )
            atm_vol_pct = atm_vol_decimal * 100  # convert for thresholds

            # 25-delta proxy: strikes at ±5 % from forward
            k_put = forward * 0.95
            k_call = forward * 1.05
            put_vol = self.sabr_calibrator.implied_vol(
                k_put, forward, expiry, params.alpha, params.rho, params.nu,
            )
            call_vol = self.sabr_calibrator.implied_vol(
                k_call, forward, expiry, params.alpha, params.rho, params.nu,
            )
            put_skew = (put_vol - atm_vol_decimal) * 100  # vol-point difference
            call_skew = (call_vol - atm_vol_decimal) * 100
            butterfly = (put_vol + call_vol - 2 * atm_vol_decimal) * 100

            surface_snapshot = VolSurface(
                timestamp=datetime.now(),
                atm_vol=atm_vol_pct,
                put_skew=put_skew,
                call_skew=call_skew,
                term_structure={int(expiry * 365): atm_vol_pct},
                butterfly=butterfly,
                sabr_params=params,
            )

            # ----- Generate signals from SABR metrics -----
            sabr_signals: List[ModuleSignal] = []

            # Vol level signal
            sabr_signals.append(self._analyze_vol_level(atm_vol_pct))

            # Skew signal (real put skew, not VIX proxy)
            skew_result = self.analyze_skew(put_skew=put_skew, atm_vol=atm_vol_pct)
            sabr_signals.append(skew_result["signal"])

            # Rho signal: large negative rho → strong skew / fear
            rho_abs = abs(params.rho)
            if rho_abs > 0.7:
                rho_signal = "bearish" if params.rho < 0 else "bullish"
                sabr_signals.append(self.create_signal(
                    signal=rho_signal,
                    strength=min(1.0, rho_abs),
                    explanation=f"SABR rho={params.rho:.2f} indicates "
                                f"{'strong skew demand' if params.rho < 0 else 'call-side pressure'}",
                ))

            # Vol-of-vol (nu) signal: high nu → unstable vol surface
            if params.nu > 1.0:
                sabr_signals.append(self.create_signal(
                    signal="bearish",
                    strength=min(1.0, params.nu / 2),
                    explanation=f"SABR nu={params.nu:.2f} — elevated vol-of-vol",
                ))

            logger.info(
                "SABR surface: ATM=%.1f%% put_skew=%.1f call_skew=%.1f "
                "rho=%.3f nu=%.3f",
                atm_vol_pct, put_skew, call_skew, params.rho, params.nu,
            )

            return {
                "sabr_params": params,
                "atm_vol": atm_vol_pct,
                "put_skew": put_skew,
                "call_skew": call_skew,
                "butterfly": butterfly,
                "forward": forward,
                "expiry": expiry,
                "surface_snapshot": surface_snapshot,
                "signals": sabr_signals,
                "calibration_error": params.error,
            }

        except Exception as exc:
            logger.warning("SABR calibration failed, falling back to VIX: %s", exc)
            return None
    
    def _analyze_vol_level(self, vol: float) -> ModuleSignal:
        """Analyze volatility level in regime context."""
        levels = self.VOL_LEVELS.get(self.current_regime, self.VOL_LEVELS[1])
        regime_name = self.REGIME_NAMES[self.current_regime]
        
        if vol < levels["low"]:
            signal = "bullish"
            strength = 0.7
            explanation = f"Vol ({vol:.1f}) is low for {regime_name} - complacent/bullish"
        elif vol < levels["normal"]:
            signal = "neutral"
            strength = 0.5
            explanation = f"Vol ({vol:.1f}) is normal for {regime_name}"
        elif vol < levels["elevated"]:
            signal = "bearish"
            strength = 0.6
            explanation = f"Vol ({vol:.1f}) is elevated for {regime_name}"
        else:
            signal = "bearish"
            strength = 0.8
            explanation = f"Vol ({vol:.1f}) is high for {regime_name} - fear elevated"
        
        return self.create_signal(
            signal=signal,
            strength=strength,
            explanation=explanation,
            vol_level=vol,
        )
    
    def analyze_skew(
        self,
        put_skew: float,
        call_skew: Optional[float] = None,
        atm_vol: Optional[float] = None,
        prior_put_skew: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze put/call skew with regime context.
        
        Args:
            put_skew: 25-delta put skew (vol points above ATM)
            call_skew: 25-delta call skew
            atm_vol: At-the-money volatility
            prior_put_skew: Previous put skew for change analysis
            
        Returns:
            Skew analysis with regime-adaptive interpretation
        """
        # Determine skew state
        if put_skew > self.extreme_skew_threshold:
            skew_state = "extreme"
        elif prior_put_skew and (put_skew - prior_put_skew) > self.skew_change_threshold:
            skew_state = "steepening"
        elif prior_put_skew and (prior_put_skew - put_skew) > self.skew_change_threshold:
            skew_state = "flattening"
        else:
            skew_state = "stable"
        
        # Get regime interpretation
        if skew_state == "stable":
            interp = {"signal": "neutral", "explanation": "Skew stable"}
        else:
            interp = self.PUT_SKEW_INTERPRETATION.get(
                self.current_regime, {}
            ).get(skew_state, {"signal": "neutral", "explanation": "Unknown"})
        
        # Calculate strength
        skew_magnitude = abs(put_skew) / self.extreme_skew_threshold
        strength = min(1.0, 0.5 + skew_magnitude * 0.5)
        
        signal = self.create_signal(
            signal=interp["signal"],
            strength=strength,
            explanation=interp["explanation"],
            put_skew=put_skew,
            skew_state=skew_state,
        )
        
        return {
            "signal": signal,
            "skew_state": skew_state,
            "put_skew": put_skew,
            "interpretation": interp["explanation"],
        }
    
    def analyze_term_structure(
        self,
        near_vol: float,
        far_vol: float,
        near_days: int = 30,
        far_days: int = 90,
    ) -> Dict[str, Any]:
        """Analyze volatility term structure.
        
        Args:
            near_vol: Near-term ATM vol
            far_vol: Far-term ATM vol
            near_days: Days to near expiry
            far_days: Days to far expiry
            
        Returns:
            Term structure analysis
        """
        spread = far_vol - near_vol
        
        if spread > 2:
            structure = "contango"
            explanation = "Vol term structure in contango - normal/bullish"
            signal = "bullish" if self.current_regime in [1, 4] else "neutral"
        elif spread < -2:
            structure = "backwardation"
            explanation = "Vol term structure in backwardation - near-term fear"
            signal = "bearish" if self.current_regime in [1, 4] else "neutral"
        else:
            structure = "flat"
            explanation = "Vol term structure flat"
            signal = "neutral"
        
        return {
            "signal": self.create_signal(
                signal=signal,
                strength=min(1.0, abs(spread) / 5),
                explanation=explanation,
            ),
            "structure": structure,
            "spread": spread,
        }
