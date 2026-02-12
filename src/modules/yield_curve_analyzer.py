"""
Yield Curve Structure Analyzer for AMRCAIS.

Analyzes the shape, steepness, and dynamics of the yield curve with
regime-adaptive interpretation. The yield curve is a powerful leading
indicator, but its meaning changes dramatically across regimes.

Key insight: A steepening yield curve signals different things:
- Risk-On Growth: Economic expansion continuing
- Risk-Off Crisis: Fed accommodation, potential recovery
- Stagflation: Inflation expectations rising faster than policy
- Disinflationary Boom: Fed has room to be accommodative

Classes:
    YieldCurveAnalyzer: Regime-adaptive yield curve analysis

Example:
    >>> analyzer = YieldCurveAnalyzer()
    >>> analyzer.update_regime(2, 0.9)  # Risk-Off Crisis
    >>> result = analyzer.analyze(yield_data)
    >>> print(result["curve_shape"])  # "inverted"
    >>> print(result["signal"].signal)  # Regime-specific interpretation
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from scipy import interpolate

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


class CurveShape(Enum):
    """Yield curve shape classifications."""
    NORMAL = "normal"          # Upward sloping
    FLAT = "flat"              # Minimal slope
    INVERTED = "inverted"      # Downward sloping
    HUMPED = "humped"          # Peak in the middle
    TWISTED = "twisted"        # Complex shape changes


@dataclass
class CurveSnapshot:
    """Point-in-time yield curve snapshot.
    
    Attributes:
        date: Snapshot date
        yields: Dict mapping tenor to yield
        slope_2_10: 10Y - 2Y spread
        slope_3m_10: 10Y - 3M spread
        curvature: Butterfly spread
        shape: Classified curve shape
    """
    date: datetime
    yields: Dict[str, float]
    slope_2_10: float
    slope_3m_10: float
    curvature: float
    shape: CurveShape


class YieldCurveAnalyzer(AnalyticalModule):
    """Analyzes yield curve structure with regime-adaptive interpretation.
    
    The yield curve encodes market expectations for growth and inflation.
    However, its predictive power varies by regime:
    
    REGIME-SPECIFIC INTERPRETATION:
    
    Risk-On Growth (1):
        - Steepening: Bullish (expansion continuing)
        - Flattening: Caution (late cycle, Fed tightening)
        - Inversion: Warning (recession signal)
    
    Risk-Off Crisis (2):
        - Steepening: Bullish (recovery expectations)
        - Flattening: Neutral (crisis ongoing)
        - Inversion: Less meaningful (flight to safety)
    
    Stagflation (3):
        - Steepening: Bearish (inflation expectations unanchored)
        - Flattening: Neutral (Fed credibility maintained)
        - Bear steepener: Very bearish (worst case)
    
    Disinflationary Boom (4):
        - Steepening: Bullish (growth without inflation)
        - Bull flattener: Very bullish (Goldilocks)
        - Inversion: Warning (overheating risk)
    
    Attributes:
        curve_history: List of historical curve snapshots
        tenors: Standard tenors to track
        
    Example:
        >>> analyzer = YieldCurveAnalyzer()
        >>> 
        >>> # In stagflation, steepening is bearish
        >>> analyzer.update_regime(3, 0.85)
        >>> result = analyzer.analyze_steepening(current=0.8, prior=0.5)
        >>> print(result["signal"].signal)  # "bearish"
        >>> 
        >>> # In risk-on, same steepening is bullish
        >>> analyzer.update_regime(1, 0.85)
        >>> result = analyzer.analyze_steepening(current=0.8, prior=0.5)
        >>> print(result["signal"].signal)  # "bullish"
    """
    
    STANDARD_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    TENOR_TO_YEARS = {
        "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
        "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30
    }
    
    # Regime-specific interpretation of curve dynamics
    STEEPENING_INTERPRETATION = {
        1: {"signal": "bullish", "explanation": "Steepening in growth regime confirms expansion"},
        2: {"signal": "bullish", "explanation": "Steepening in crisis signals recovery expectations"},
        3: {"signal": "bearish", "explanation": "Steepening in stagflation suggests inflation expectations rising"},
        4: {"signal": "bullish", "explanation": "Steepening in disinflation supports growth narrative"},
    }
    
    FLATTENING_INTERPRETATION = {
        1: {"signal": "cautious", "explanation": "Flattening in growth signals late cycle/Fed tightening"},
        2: {"signal": "neutral", "explanation": "Flattening in crisis is neutral (ongoing stress)"},
        3: {"signal": "neutral", "explanation": "Flattening in stagflation suggests Fed credibility"},
        4: {"signal": "bullish", "explanation": "Bull flattening in disinflation is Goldilocks"},
    }
    
    INVERSION_INTERPRETATION = {
        1: {"signal": "bearish", "explanation": "Inversion in growth is classic recession warning"},
        2: {"signal": "neutral", "explanation": "Inversion in crisis driven by flight to safety"},
        3: {"signal": "bearish", "explanation": "Inversion in stagflation signals policy credibility loss"},
        4: {"signal": "cautious", "explanation": "Inversion in disinflation warns of overheating"},
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the yield curve analyzer."""
        super().__init__(name="YieldCurveAnalyzer", config_path=config_path)
        
        self.curve_history: List[CurveSnapshot] = []
        self._spline_cache: Dict[str, Any] = {}
        
        # Thresholds for curve shape classification
        self.inversion_threshold = -0.1  # % for inversion
        self.flat_threshold = 0.25  # % for flat curve
    
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get yield curve parameters for a specific regime."""
        return {
            "steepening": self.STEEPENING_INTERPRETATION.get(regime, {}),
            "flattening": self.FLATTENING_INTERPRETATION.get(regime, {}),
            "inversion": self.INVERSION_INTERPRETATION.get(regime, {}),
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze yield curve data.
        
        Args:
            data: DataFrame with yield columns (e.g., 'DGS2', 'DGS10')
            
        Returns:
            Comprehensive yield curve analysis
        """
        # Extract yields for standard tenors
        yields = self._extract_yields(data)
        
        if not yields:
            return {
                "signal": self.create_signal(
                    signal="neutral",
                    strength=0.0,
                    explanation="Insufficient yield data for analysis",
                ),
                "error": "No valid yield data found",
            }
        
        # Calculate spreads
        slope_2_10 = yields.get("10Y", 0) - yields.get("2Y", 0) if "2Y" in yields and "10Y" in yields else None
        slope_3m_10 = yields.get("10Y", 0) - yields.get("3M", 0) if "3M" in yields and "10Y" in yields else None
        
        # Calculate curvature (butterfly)
        curvature = self._calculate_curvature(yields)
        
        # Classify shape
        shape = self._classify_shape(slope_2_10, slope_3m_10, curvature)
        
        # Create snapshot
        snapshot = CurveSnapshot(
            date=datetime.now(),
            yields=yields,
            slope_2_10=slope_2_10 or 0,
            slope_3m_10=slope_3m_10 or 0,
            curvature=curvature or 0,
            shape=shape,
        )
        self.curve_history.append(snapshot)
        
        # Analyze curve dynamics if we have history
        dynamics = self._analyze_dynamics()
        
        # Generate regime-adaptive signal
        signal = self._generate_signal(shape, dynamics, slope_2_10)
        
        return {
            "signal": signal,
            "curve_shape": shape.value,
            "slope_2_10": slope_2_10,
            "slope_3m_10": slope_3m_10,
            "curvature": curvature,
            "yields": yields,
            "dynamics": dynamics,
            "regime_parameters": self.get_current_parameters(),
        }
    
    def _extract_yields(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract yields from DataFrame for standard tenors."""
        yields = {}
        
        # Mapping from common column names to tenors
        column_mapping = {
            "DGS3MO": "3M", "TB3MS": "3M",
            "DGS6MO": "6M", "TB6MS": "6M",
            "DGS1": "1Y", "GS1": "1Y",
            "DGS2": "2Y", "GS2": "2Y",
            "DGS3": "3Y", "GS3": "3Y",
            "DGS5": "5Y", "GS5": "5Y",
            "DGS7": "7Y", "GS7": "7Y",
            "DGS10": "10Y", "GS10": "10Y",
            "DGS20": "20Y", "GS20": "20Y",
            "DGS30": "30Y", "GS30": "30Y",
        }
        
        for col, tenor in column_mapping.items():
            if col in data.columns:
                val = data[col].dropna().iloc[-1] if len(data[col].dropna()) > 0 else None
                if val is not None and not np.isnan(val):
                    yields[tenor] = float(val)
        
        return yields
    
    def _classify_shape(
        self,
        slope_2_10: Optional[float],
        slope_3m_10: Optional[float],
        curvature: Optional[float],
    ) -> CurveShape:
        """Classify the yield curve shape."""
        if slope_2_10 is None and slope_3m_10 is None:
            return CurveShape.NORMAL  # Default
        
        primary_slope = slope_2_10 if slope_2_10 is not None else slope_3m_10
        
        if primary_slope < self.inversion_threshold:
            return CurveShape.INVERTED
        elif abs(primary_slope) < self.flat_threshold:
            return CurveShape.FLAT
        elif curvature and abs(curvature) > 0.3:
            return CurveShape.HUMPED
        else:
            return CurveShape.NORMAL
    
    def _calculate_curvature(self, yields: Dict[str, float]) -> Optional[float]:
        """Calculate butterfly spread (curvature)."""
        if all(k in yields for k in ["2Y", "5Y", "10Y"]):
            # Butterfly = 2 * 5Y - 2Y - 10Y
            return 2 * yields["5Y"] - yields["2Y"] - yields["10Y"]
        return None
    
    def _analyze_dynamics(self) -> Dict[str, Any]:
        """Analyze recent curve dynamics."""
        if len(self.curve_history) < 2:
            return {"direction": "unknown", "magnitude": 0}
        
        recent = self.curve_history[-1]
        prior = self.curve_history[-2]
        
        slope_change = recent.slope_2_10 - prior.slope_2_10
        
        if abs(slope_change) < 0.05:
            direction = "stable"
        elif slope_change > 0:
            direction = "steepening"
        else:
            direction = "flattening"
        
        return {
            "direction": direction,
            "slope_change": slope_change,
            "magnitude": abs(slope_change),
        }
    
    def _generate_signal(
        self,
        shape: CurveShape,
        dynamics: Dict,
        slope_2_10: Optional[float],
    ) -> ModuleSignal:
        """Generate regime-adaptive signal from curve analysis."""
        direction = dynamics.get("direction", "unknown")
        magnitude = dynamics.get("magnitude", 0)
        
        # Get regime-specific interpretation
        if direction == "steepening":
            interp = self.STEEPENING_INTERPRETATION.get(
                self.current_regime, {"signal": "neutral", "explanation": "Unknown"}
            )
        elif direction == "flattening":
            interp = self.FLATTENING_INTERPRETATION.get(
                self.current_regime, {"signal": "neutral", "explanation": "Unknown"}
            )
        else:
            interp = {"signal": "neutral", "explanation": "Curve dynamics stable"}
        
        # Override for inversion
        if shape == CurveShape.INVERTED:
            interp = self.INVERSION_INTERPRETATION.get(
                self.current_regime, {"signal": "bearish", "explanation": "Curve inverted"}
            )
        
        # Calculate strength
        if magnitude > 0.2:
            strength = 0.9
        elif magnitude > 0.1:
            strength = 0.7
        elif magnitude > 0.05:
            strength = 0.5
        else:
            strength = 0.3
        
        return self.create_signal(
            signal=interp["signal"],
            strength=strength,
            explanation=interp["explanation"],
            curve_shape=shape.value,
            direction=direction,
            slope_2_10=slope_2_10,
        )
    
    def interpolate_curve(
        self,
        yields: Dict[str, float],
        target_tenors: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        """Interpolate the yield curve using cubic splines.
        
        Args:
            yields: Dict mapping tenor string to yield
            target_tenors: List of tenors (in years) to interpolate
            
        Returns:
            Dict mapping tenor (years) to interpolated yield
        """
        if target_tenors is None:
            target_tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        
        # Convert to arrays
        tenors_years = []
        yield_values = []
        
        for tenor_str, yield_val in yields.items():
            if tenor_str in self.TENOR_TO_YEARS:
                tenors_years.append(self.TENOR_TO_YEARS[tenor_str])
                yield_values.append(yield_val)
        
        if len(tenors_years) < 3:
            return {}
        
        # Sort by tenor
        sorted_pairs = sorted(zip(tenors_years, yield_values))
        tenors_years, yield_values = zip(*sorted_pairs)
        
        # Fit cubic spline
        try:
            spline = interpolate.CubicSpline(tenors_years, yield_values)
            interpolated = {}
            
            for t in target_tenors:
                if min(tenors_years) <= t <= max(tenors_years):
                    interpolated[t] = float(spline(t))
            
            return interpolated
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}")
            return {}
    
    def calculate_forward_rates(
        self,
        yields: Dict[str, float],
        forward_tenor: int = 1,
    ) -> Dict[str, float]:
        """Calculate forward rates from the spot curve.
        
        Args:
            yields: Spot yields by tenor
            forward_tenor: Forward period in years
            
        Returns:
            Forward rates for various starting points
        """
        forward_rates = {}
        
        sorted_tenors = sorted(
            [(self.TENOR_TO_YEARS.get(t, 0), y) for t, y in yields.items() if t in self.TENOR_TO_YEARS]
        )
        
        for i in range(len(sorted_tenors) - 1):
            t1, y1 = sorted_tenors[i]
            t2, y2 = sorted_tenors[i + 1]
            
            # Forward rate formula: f = (y2*t2 - y1*t1) / (t2 - t1)
            if t2 > t1:
                forward = ((y2 * t2) - (y1 * t1)) / (t2 - t1)
                forward_rates[f"{int(t1)}Y-{int(t2)}Y"] = forward
        
        return forward_rates
    
    def detect_inversion(self, yields: Dict[str, float]) -> Dict[str, bool]:
        """Detect inversions across various spreads.
        
        Args:
            yields: Yield curve data
            
        Returns:
            Dict indicating which spreads are inverted
        """
        inversions = {}
        
        spread_pairs = [
            ("2Y", "10Y", "2s10s"),
            ("3M", "10Y", "3m10y"),
            ("2Y", "5Y", "2s5s"),
            ("5Y", "30Y", "5s30s"),
        ]
        
        for short, long, name in spread_pairs:
            if short in yields and long in yields:
                spread = yields[long] - yields[short]
                inversions[name] = spread < 0
        
        return inversions
