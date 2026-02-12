"""
Correlation Anomaly Detector for AMRCAIS.

Detects unusual correlation patterns that may signal regime changes or
stress events. Cross-asset correlations are regime-dependent:

- Risk-On: Equity-bond negative correlation (diversification works)
- Risk-Off: All correlations spike to 1 (flight to safety)
- Stagflation: Unusual patterns (bonds and stocks both fall)
- Disinflationary Boom: Low correlations across the board

The "correlation breakdown" where historical relationships fail is often
one of the earliest regime transition signals.

Classes:
    CorrelationAnomalyDetector: Regime-adaptive correlation analysis
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
class CorrelationAnomaly:
    """Detected correlation anomaly.
    
    Attributes:
        pair: Asset pair (e.g., "SPX-TLT")
        current_corr: Current rolling correlation
        baseline_corr: Expected correlation for regime
        deviation: Standard deviations from baseline
        anomaly_type: Type of anomaly detected
    """
    pair: str
    current_corr: float
    baseline_corr: float
    deviation: float
    anomaly_type: str  # "breakdown", "spike", "inversion"


class CorrelationAnomalyDetector(AnalyticalModule):
    """Detects correlation anomalies across asset classes.
    
    Cross-asset correlations encode important information about market
    structure and regime. This module:
    
    1. Tracks rolling correlations across key pairs
    2. Compares to regime-specific baselines
    3. Detects anomalies that may signal transitions
    4. Flags correlation breakdowns (early warning)
    
    KEY CORRELATION PAIRS:
    - SPX-TLT: Equity-bond (risk sentiment)
    - SPX-GLD: Equity-gold (inflation/fear)
    - TLT-GLD: Bond-gold (real rate proxy)
    - SPX-VIX: Equity-vol (should be negative)
    - DXY-GLD: Dollar-gold (inverse relationship)
    
    REGIME BASELINES:
    
    Risk-On Growth (1):
        - SPX-TLT: -0.3 to -0.5 (negative)
        - SPX-GLD: Near 0
        - SPX-VIX: -0.7 to -0.8
    
    Risk-Off Crisis (2):
        - SPX-TLT: Can spike positive (flight)
        - All correlations elevated
        - SPX-VIX: Can approach -1
    
    Stagflation (3):
        - SPX-TLT: Positive (unusual)
        - TLT-GLD: Negative
        - Correlation regime breakdown
    
    Disinflationary Boom (4):
        - Low correlations generally
        - SPX-TLT: -0.2 to -0.3
        - Stable correlation structure
    
    Attributes:
        correlation_history: Historical correlation observations
        
    Example:
        >>> detector = CorrelationAnomalyDetector()
        >>> detector.update_regime(1, 0.85)  # Risk-On
        >>> 
        >>> # SPX-TLT going positive is anomalous in risk-on
        >>> result = detector.check_correlation("SPX", "TLT", 0.3)
        >>> if result["is_anomaly"]:
        ...     print(result["signal"].explanation)
    """
    
    # Key correlation pairs to monitor
    MONITOR_PAIRS = [
        ("SPX", "TLT"),   # Equity-Bond
        ("SPX", "GLD"),   # Equity-Gold
        ("TLT", "GLD"),   # Bond-Gold
        ("SPX", "VIX"),   # Equity-Vol
        ("DXY", "GLD"),   # Dollar-Gold
        ("SPX", "DXY"),   # Equity-Dollar
        ("WTI", "SPX"),   # Oil-Equity
    ]
    
    # Regime-specific correlation baselines
    CORRELATION_BASELINES = {
        1: {  # Risk-On Growth
            "SPX-TLT": {"mean": -0.35, "std": 0.15, "range": (-0.6, 0.0)},
            "SPX-GLD": {"mean": 0.0, "std": 0.2, "range": (-0.3, 0.3)},
            "TLT-GLD": {"mean": 0.2, "std": 0.15, "range": (-0.1, 0.5)},
            "SPX-VIX": {"mean": -0.75, "std": 0.1, "range": (-0.9, -0.5)},
            "DXY-GLD": {"mean": -0.4, "std": 0.15, "range": (-0.7, -0.1)},
        },
        2: {  # Risk-Off Crisis
            "SPX-TLT": {"mean": 0.0, "std": 0.3, "range": (-0.5, 0.5)},
            "SPX-GLD": {"mean": 0.2, "std": 0.25, "range": (-0.3, 0.7)},
            "TLT-GLD": {"mean": 0.3, "std": 0.2, "range": (0.0, 0.7)},
            "SPX-VIX": {"mean": -0.85, "std": 0.1, "range": (-1.0, -0.6)},
            "DXY-GLD": {"mean": -0.2, "std": 0.3, "range": (-0.6, 0.4)},
        },
        3: {  # Stagflation
            "SPX-TLT": {"mean": 0.2, "std": 0.2, "range": (-0.2, 0.5)},
            "SPX-GLD": {"mean": -0.1, "std": 0.25, "range": (-0.5, 0.3)},
            "TLT-GLD": {"mean": -0.1, "std": 0.2, "range": (-0.4, 0.3)},
            "SPX-VIX": {"mean": -0.7, "std": 0.15, "range": (-0.9, -0.4)},
            "DXY-GLD": {"mean": -0.3, "std": 0.2, "range": (-0.6, 0.1)},
        },
        4: {  # Disinflationary Boom
            "SPX-TLT": {"mean": -0.25, "std": 0.15, "range": (-0.5, 0.1)},
            "SPX-GLD": {"mean": 0.1, "std": 0.15, "range": (-0.2, 0.4)},
            "TLT-GLD": {"mean": 0.15, "std": 0.15, "range": (-0.1, 0.4)},
            "SPX-VIX": {"mean": -0.7, "std": 0.1, "range": (-0.85, -0.5)},
            "DXY-GLD": {"mean": -0.35, "std": 0.15, "range": (-0.6, -0.1)},
        },
    }
    
    def __init__(self, config_path: Optional[str] = None, window: int = 60):
        """Initialize the correlation anomaly detector.
        
        Args:
            config_path: Path to configuration
            window: Rolling window for correlation calculation
        """
        super().__init__(name="CorrelationAnomalyDetector", config_path=config_path)
        
        self.window = window
        self.correlation_history: List[Dict[str, float]] = []
        self._anomaly_history: List[CorrelationAnomaly] = []
        
        # Anomaly detection threshold (standard deviations)
        self.anomaly_threshold = 2.0
    
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get correlation baselines for a specific regime."""
        return {
            "baselines": self.CORRELATION_BASELINES.get(regime, {}),
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation structure.
        
        Args:
            data: DataFrame with asset price/return columns
            
        Returns:
            Comprehensive correlation analysis
        """
        anomalies = []
        current_correlations = {}
        
        # Calculate correlations for all pairs
        for asset1, asset2 in self.MONITOR_PAIRS:
            col1 = self._find_column(data, asset1)
            col2 = self._find_column(data, asset2)
            
            if col1 is not None and col2 is not None:
                # Calculate returns if needed
                if not col1.endswith("_ret"):
                    returns1 = data[col1].pct_change().dropna()
                else:
                    returns1 = data[col1].dropna()
                    
                if not col2.endswith("_ret"):
                    returns2 = data[col2].pct_change().dropna()
                else:
                    returns2 = data[col2].dropna()
                
                # Calculate rolling correlation
                if len(returns1) >= self.window and len(returns2) >= self.window:
                    # Align series
                    aligned = pd.concat([returns1, returns2], axis=1).dropna()
                    if len(aligned) >= self.window:
                        corr = aligned.iloc[-self.window:].corr().iloc[0, 1]
                        
                        pair_name = f"{asset1}-{asset2}"
                        current_correlations[pair_name] = corr
                        
                        # Check for anomaly
                        result = self.check_correlation(asset1, asset2, corr)
                        if result["is_anomaly"]:
                            anomalies.append(result["anomaly"])
        
        # Store history
        if current_correlations:
            self.correlation_history.append(current_correlations)
        
        # Generate overall signal
        if anomalies:
            signal = self._generate_anomaly_signal(anomalies)
        else:
            signal = self.create_signal(
                signal="neutral",
                strength=0.3,
                explanation="Correlation structure normal for regime",
            )
        
        return {
            "signal": signal,
            "correlations": current_correlations,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "regime_parameters": self.get_current_parameters(),
        }
    
    def _find_column(self, data: pd.DataFrame, asset: str) -> Optional[str]:
        """Find column name matching asset."""
        # Direct match
        if asset in data.columns:
            return asset
        
        # Common variations
        variations = [
            asset,
            f"{asset}_Close",
            f"{asset}_close",
            f"^{asset}",
            asset.lower(),
        ]
        
        for var in variations:
            if var in data.columns:
                return var
        
        return None
    
    def check_correlation(
        self,
        asset1: str,
        asset2: str,
        current_corr: float,
    ) -> Dict[str, Any]:
        """Check if correlation is anomalous for current regime.
        
        Args:
            asset1: First asset
            asset2: Second asset
            current_corr: Current correlation value
            
        Returns:
            Anomaly check results
        """
        pair_name = f"{asset1}-{asset2}"
        baselines = self.CORRELATION_BASELINES.get(self.current_regime, {})
        baseline = baselines.get(pair_name, {"mean": 0, "std": 0.3, "range": (-1, 1)})
        
        expected_mean = baseline["mean"]
        expected_std = baseline["std"]
        expected_range = baseline["range"]
        
        # Calculate deviation
        deviation = (current_corr - expected_mean) / expected_std if expected_std > 0 else 0
        
        # Check for anomaly
        is_anomaly = False
        anomaly_type = None
        
        if abs(deviation) > self.anomaly_threshold:
            is_anomaly = True
            if current_corr > expected_range[1]:
                anomaly_type = "spike"
            elif current_corr < expected_range[0]:
                anomaly_type = "inversion"
            else:
                anomaly_type = "breakdown"
        
        if is_anomaly:
            anomaly = CorrelationAnomaly(
                pair=pair_name,
                current_corr=current_corr,
                baseline_corr=expected_mean,
                deviation=deviation,
                anomaly_type=anomaly_type,
            )
            self._anomaly_history.append(anomaly)
            
            signal = self._interpret_anomaly(anomaly)
        else:
            anomaly = None
            signal = self.create_signal(
                signal="neutral",
                strength=0.2,
                explanation=f"{pair_name} correlation normal at {current_corr:.2f}",
            )
        
        return {
            "pair": pair_name,
            "current_corr": current_corr,
            "expected_mean": expected_mean,
            "deviation": deviation,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "anomaly": anomaly,
            "signal": signal,
        }
    
    def _interpret_anomaly(self, anomaly: CorrelationAnomaly) -> ModuleSignal:
        """Interpret a correlation anomaly."""
        regime_name = self.REGIME_NAMES[self.current_regime]
        
        # Specific interpretations
        if anomaly.pair == "SPX-TLT":
            if anomaly.anomaly_type == "spike" and self.current_regime == 1:
                return self.create_signal(
                    signal="bearish",
                    strength=0.8,
                    explanation=f"SPX-TLT correlation spiking in {regime_name} - "
                               "diversification breaking down, possible regime shift",
                    anomaly=anomaly.__dict__,
                )
            elif anomaly.anomaly_type == "inversion" and self.current_regime == 3:
                return self.create_signal(
                    signal="bullish",
                    strength=0.6,
                    explanation=f"SPX-TLT correlation normalizing in {regime_name} - "
                               "possible exit from stagflation",
                    anomaly=anomaly.__dict__,
                )
        
        elif anomaly.pair == "SPX-VIX":
            if anomaly.current_corr > -0.5:
                return self.create_signal(
                    signal="cautious",
                    strength=0.7,
                    explanation="SPX-VIX correlation breakdown - unusual market structure",
                    anomaly=anomaly.__dict__,
                )
        
        # Generic interpretation
        return self.create_signal(
            signal="cautious",
            strength=min(0.9, abs(anomaly.deviation) / 3),
            explanation=f"{anomaly.pair} correlation anomaly ({anomaly.anomaly_type}) - "
                       f"current {anomaly.current_corr:.2f} vs baseline {anomaly.baseline_corr:.2f}",
            anomaly=anomaly.__dict__,
        )
    
    def _generate_anomaly_signal(
        self,
        anomalies: List[CorrelationAnomaly],
    ) -> ModuleSignal:
        """Generate overall signal from multiple anomalies."""
        # Multiple anomalies suggest possible regime transition
        if len(anomalies) >= 3:
            return self.create_signal(
                signal="cautious",
                strength=0.9,
                explanation=f"Multiple correlation anomalies ({len(anomalies)}) - "
                           "possible regime transition",
                anomaly_count=len(anomalies),
            )
        elif len(anomalies) == 2:
            return self.create_signal(
                signal="cautious",
                strength=0.7,
                explanation=f"Two correlation anomalies detected - monitoring closely",
                anomaly_count=len(anomalies),
            )
        else:
            anomaly = anomalies[0]
            return self._interpret_anomaly(anomaly)
    
    def get_correlation_matrix(
        self,
        data: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """Calculate full correlation matrix.
        
        Args:
            data: DataFrame with asset returns
            window: Rolling window (uses default if not specified)
            
        Returns:
            Correlation matrix DataFrame
        """
        if window is None:
            window = self.window
        
        # Get return columns
        return_cols = [c for c in data.columns if "_ret" in c.lower() or 
                      any(a in c for a in ["SPX", "TLT", "GLD", "DXY", "VIX", "WTI"])]
        
        if not return_cols:
            # Convert prices to returns
            returns_df = data.pct_change().dropna()
        else:
            returns_df = data[return_cols].dropna()
        
        if len(returns_df) >= window:
            return returns_df.iloc[-window:].corr()
        
        return returns_df.corr()
    
    def detect_correlation_regime_shift(
        self,
        lookback: int = 20,
    ) -> Dict[str, Any]:
        """Detect if correlation structure is shifting.
        
        Args:
            lookback: Periods to compare
            
        Returns:
            Regime shift indicators
        """
        if len(self.correlation_history) < lookback * 2:
            return {"shift_detected": False, "reason": "Insufficient history"}
        
        # Compare recent vs prior correlations
        recent = self.correlation_history[-lookback:]
        prior = self.correlation_history[-lookback*2:-lookback]
        
        shifts = {}
        for pair in self.MONITOR_PAIRS:
            pair_name = f"{pair[0]}-{pair[1]}"
            
            recent_corrs = [h.get(pair_name) for h in recent if pair_name in h]
            prior_corrs = [h.get(pair_name) for h in prior if pair_name in h]
            
            if recent_corrs and prior_corrs:
                recent_mean = np.mean(recent_corrs)
                prior_mean = np.mean(prior_corrs)
                shift = recent_mean - prior_mean
                
                if abs(shift) > 0.2:
                    shifts[pair_name] = {
                        "recent_mean": recent_mean,
                        "prior_mean": prior_mean,
                        "shift": shift,
                    }
        
        shift_detected = len(shifts) >= 2
        
        return {
            "shift_detected": shift_detected,
            "shifting_pairs": shifts,
            "reason": f"{len(shifts)} pairs showing correlation shift" if shift_detected else "Stable",
        }
