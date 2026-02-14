"""
Options Surface Monitor for AMRCAIS.

Monitors implied volatility surfaces, skew dynamics, and term structure
to extract regime-adaptive signals about market sentiment and risk pricing.

The options market often leads the underlying - skew changes can signal
sentiment shifts before they appear in price. However, interpretation
is regime-dependent:

- Risk-On Growth: Put skew flattening → bullish continuation
- Risk-Off Crisis: Put skew steepening → fear, but may signal capitulation
- Stagflation: Call skew rising → inflation hedging
- Disinflationary Boom: Low vol, flat skew → Goldilocks

Classes:
    OptionsSurfaceMonitor: Regime-adaptive options surface analysis
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


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
    """
    timestamp: datetime
    atm_vol: float
    put_skew: float
    call_skew: float
    term_structure: Dict[int, float]  # days to expiry -> vol
    butterfly: float = 0.0


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
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the options surface monitor."""
        super().__init__(name="OptionsSurfaceMonitor", config_path=config_path)
        
        self.surface_history: List[VolSurface] = []
        
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
        
        Extracts VIX as ATM vol proxy, computes put skew from available
        data, analyzes term structure, and constructs VolSurface snapshots.
        
        Args:
            data: DataFrame with VIX, vol surface data, etc.
            
        Returns:
            Comprehensive options surface analysis
        """
        signals = []
        surface_snapshot = None
        
        # Extract VIX as ATM vol proxy
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
                # Use VIX mean-reversion as skew proxy: how far above mean
                vix_mean = float(vix_series.iloc[-60:].mean()) if len(vix_series) >= 60 else float(vix_series.mean())
                put_skew_value = vix - vix_mean  # positive = elevated skew
                
                # Get prior skew for change detection
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
                # Realized vol as "near term" proxy, VIX as "far term"
                realized_vol = float(spx.pct_change().iloc[-20:].std() * np.sqrt(252) * 100)
                term_result = self.analyze_term_structure(
                    near_vol=realized_vol,
                    far_vol=vix,
                    near_days=20,
                    far_days=30,
                )
                signals.append(term_result["signal"])
        
        # Construct VolSurface snapshot
        if vix is not None:
            surface_snapshot = VolSurface(
                timestamp=datetime.now(),
                atm_vol=vix,
                put_skew=put_skew_value or 0.0,
                call_skew=-(put_skew_value or 0.0) * 0.5,  # approx
                term_structure={30: vix},
                butterfly=abs(put_skew_value or 0.0) * 0.3,
            )
            self.surface_history.append(surface_snapshot)
        
        if signals:
            avg_strength = np.mean([s.strength for s in signals])
            bullish = sum(1 for s in signals if s.signal == "bullish")
            bearish = sum(1 for s in signals if s.signal == "bearish")
            
            overall = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
            
            return {
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
        
        return {
            "signal": self.create_signal(
                signal="neutral",
                strength=0.0,
                explanation="Insufficient options data",
            ),
        }
    
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
