"""
Volatility Regime Classifier for AMRCAIS.

This module implements a VIX and realized volatility-based classifier
that identifies regimes based on market fear and uncertainty levels.

The VIX is a key indicator of market regime:
- Low VIX (<15): Typically Risk-On or Disinflationary Boom
- Normal VIX (15-20): Stable markets, could be any regime
- Elevated VIX (20-30): Uncertainty, possible Stagflation
- High VIX (>30): Risk-Off Crisis

Classes:
    VolatilityRegimeClassifier: VIX-based regime detector

Example:
    >>> classifier = VolatilityRegimeClassifier()
    >>> classifier.fit(market_data)
    >>> result = classifier.predict(recent_vix)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np

from src.regime_detection.base import BaseClassifier, RegimeResult

logger = logging.getLogger(__name__)


class VolatilityRegimeClassifier(BaseClassifier):
    """Classifier based on VIX levels and realized volatility.
    
    Uses a rule-based approach with learned thresholds to classify
    regimes based on volatility characteristics:
    
    - Risk-On Growth: Low VIX, declining realized vol
    - Risk-Off Crisis: High VIX, spiking realized vol  
    - Stagflation: Elevated VIX, persistent high vol
    - Disinflationary Boom: Low VIX, stable low vol
    
    Also tracks VIX term structure (contango/backwardation) and
    VIX percentile for additional context.
    
    Attributes:
        vix_thresholds: Dictionary of VIX level thresholds
        use_percentile: Whether to use VIX percentile
        
    Example:
        >>> vol_classifier = VolatilityRegimeClassifier()
        >>> vol_classifier.fit(data_with_vix)
        >>> result = vol_classifier.predict(recent_data)
    """
    
    # Default VIX thresholds (can be learned from data)
    DEFAULT_THRESHOLDS = {
        "low": 15,
        "normal": 20,
        "elevated": 25,
        "high": 30,
        "extreme": 40,
    }
    
    # Regime mappings based on VIX characteristics
    REGIME_VIX_PROFILES = {
        1: {  # Risk-On Growth
            "vix_range": (10, 18),
            "vix_trend": "declining",
            "percentile_range": (0, 40),
        },
        2: {  # Risk-Off Crisis
            "vix_range": (30, 80),
            "vix_trend": "spiking",
            "percentile_range": (80, 100),
        },
        3: {  # Stagflation
            "vix_range": (20, 35),
            "vix_trend": "elevated_stable",
            "percentile_range": (50, 80),
        },
        4: {  # Disinflationary Boom
            "vix_range": (12, 18),
            "vix_trend": "low_stable",
            "percentile_range": (10, 50),
        },
    }
    
    def __init__(
        self,
        vix_thresholds: Optional[Dict[str, float]] = None,
        realized_vol_window: int = 20,
        vix_percentile_window: int = 252,
        use_vix_trend: bool = True,
        config: Optional[Dict] = None,
    ):
        """Initialize the volatility regime classifier.
        
        Args:
            vix_thresholds: Custom VIX thresholds for regime boundaries
            realized_vol_window: Window for realized volatility calculation
            vix_percentile_window: Window for VIX percentile calculation
            use_vix_trend: Whether to incorporate VIX trend in classification
            config: Optional configuration dictionary
        """
        super().__init__(n_regimes=4, name="Volatility Classifier", config=config)
        
        self.vix_thresholds = vix_thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.realized_vol_window = realized_vol_window
        self.vix_percentile_window = vix_percentile_window
        self.use_vix_trend = use_vix_trend
        
        self._vix_history: Optional[pd.Series] = None
        self._learned_thresholds: Dict[str, float] = {}
        self._regime_vix_stats: Dict[int, Dict] = {}
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> "VolatilityRegimeClassifier":
        """Train the volatility classifier (learn thresholds from data).
        
        Args:
            data: DataFrame with 'VIX' column, or VIX series directly
            labels: Optional regime labels for supervised threshold learning
            
        Returns:
            self: The fitted classifier
        """
        logger.info("Training volatility classifier")
        
        # Extract VIX data
        if isinstance(data, pd.DataFrame):
            if "VIX" in data.columns:
                vix = data["VIX"].dropna()
            elif "VIX_level" in data.columns:
                vix = data["VIX_level"].dropna()
            elif "Close" in data.columns:
                # Assume it's VIX data directly
                vix = data["Close"].dropna()
            else:
                raise ValueError("Data must contain 'VIX' or 'Close' column")
        elif isinstance(data, pd.Series):
            vix = data.dropna()
        else:
            vix = pd.Series(data.flatten())
        
        self._vix_history = vix
        
        # Learn thresholds from historical distribution
        self._learn_thresholds(vix)
        
        # If labels provided, learn regime-specific VIX statistics
        if labels is not None:
            self._learn_regime_stats(vix, labels)
        
        self.is_fitted = True
        self._fit_timestamp = datetime.now()
        self._fit_samples = len(vix)
        
        logger.info(
            f"Volatility classifier fitted. Learned thresholds: "
            f"low={self._learned_thresholds.get('low', 'N/A'):.1f}, "
            f"high={self._learned_thresholds.get('high', 'N/A'):.1f}"
        )
        
        return self
    
    def _learn_thresholds(self, vix: pd.Series) -> None:
        """Learn VIX thresholds from historical distribution."""
        # Use percentiles to determine thresholds
        self._learned_thresholds = {
            "low": float(vix.quantile(0.25)),
            "normal": float(vix.quantile(0.50)),
            "elevated": float(vix.quantile(0.75)),
            "high": float(vix.quantile(0.90)),
            "extreme": float(vix.quantile(0.95)),
        }
        
        # Store additional statistics
        self._vix_mean = float(vix.mean())
        self._vix_std = float(vix.std())
    
    def _learn_regime_stats(
        self,
        vix: pd.Series,
        labels: np.ndarray,
    ) -> None:
        """Learn VIX statistics for each regime from labeled data."""
        if len(vix) != len(labels):
            logger.warning("VIX and labels length mismatch, skipping regime stats")
            return
        
        for regime in range(1, 5):
            mask = labels == regime
            if mask.sum() > 0:
                regime_vix = vix[mask]
                self._regime_vix_stats[regime] = {
                    "mean": float(regime_vix.mean()),
                    "std": float(regime_vix.std()),
                    "median": float(regime_vix.median()),
                    "min": float(regime_vix.min()),
                    "max": float(regime_vix.max()),
                    "count": int(mask.sum()),
                }
                logger.debug(
                    f"Regime {regime} VIX stats: "
                    f"mean={self._regime_vix_stats[regime]['mean']:.1f}, "
                    f"median={self._regime_vix_stats[regime]['median']:.1f}"
                )
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, pd.Series, float],
    ) -> RegimeResult:
        """Predict regime based on current VIX and volatility characteristics.
        
        Args:
            data: Current VIX level or DataFrame with VIX data
            
        Returns:
            RegimeResult with regime and confidence
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Extract current VIX
        if isinstance(data, (int, float)):
            current_vix = float(data)
            vix_trend = "unknown"
            vix_percentile = None
        elif isinstance(data, pd.Series):
            current_vix = float(data.iloc[-1])
            vix_trend = self._calculate_trend(data)
            vix_percentile = self._calculate_percentile(current_vix)
        elif isinstance(data, pd.DataFrame):
            if "VIX" in data.columns:
                vix_series = data["VIX"]
            elif "VIX_level" in data.columns:
                vix_series = data["VIX_level"]
            elif "Close" in data.columns:
                vix_series = data["Close"]
            else:
                raise ValueError("DataFrame must have VIX data")
            
            current_vix = float(vix_series.iloc[-1])
            vix_trend = self._calculate_trend(vix_series)
            vix_percentile = self._calculate_percentile(current_vix)
        else:
            arr = np.asarray(data).flatten()
            current_vix = float(arr[-1])
            vix_trend = "unknown"
            vix_percentile = self._calculate_percentile(current_vix)
        
        # Classify based on VIX level
        regime, confidence = self._classify_vix(
            current_vix, vix_trend, vix_percentile
        )
        
        # Calculate probabilities
        probabilities = self._calculate_probabilities(
            current_vix, vix_percentile
        )
        
        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            features_used=["VIX_level", "VIX_percentile", "VIX_trend"],
            metadata={
                "vix_level": current_vix,
                "vix_percentile": vix_percentile,
                "vix_trend": vix_trend,
                "thresholds": self._learned_thresholds,
            },
        )
        
        self._log_classification(result)
        return result
    
    def _calculate_trend(self, vix: pd.Series, lookback: int = 10) -> str:
        """Calculate VIX trend direction."""
        if len(vix) < lookback:
            return "unknown"
        
        recent = vix.iloc[-lookback:]
        change = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
        
        if change > 0.2:
            return "spiking"
        elif change > 0.05:
            return "rising"
        elif change < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_percentile(self, current_vix: float) -> Optional[float]:
        """Calculate current VIX percentile relative to history."""
        if self._vix_history is None or len(self._vix_history) < 10:
            return None
        
        # Use recent history for percentile
        history = self._vix_history.iloc[-self.vix_percentile_window:]
        percentile = (history < current_vix).mean() * 100
        
        return float(percentile)
    
    def _classify_vix(
        self,
        vix: float,
        trend: str,
        percentile: Optional[float],
    ) -> Tuple[int, float]:
        """Classify regime based on VIX characteristics.
        
        Returns:
            Tuple of (regime_id, confidence)
        """
        thresholds = self._learned_thresholds
        
        # Risk-Off Crisis: Very high VIX
        if vix >= thresholds.get("high", 30):
            regime = 2
            confidence = min(0.95, 0.5 + (vix - thresholds["high"]) / 20)
            
        # Low VIX regimes
        elif vix <= thresholds.get("normal", 20):
            if trend == "spiking":
                # Transitioning to risk-off
                regime = 2
                confidence = 0.6
            elif percentile is not None and percentile > 30:
                # Relatively elevated for "low" - might be disinflation
                regime = 4
                confidence = 0.7
            else:
                # Classic risk-on
                regime = 1
                confidence = min(0.9, 0.5 + (thresholds["low"] - vix) / 10)
                
        # Elevated VIX (between normal and high)
        else:
            if trend == "spiking":
                regime = 2  # Moving toward crisis
                confidence = 0.7
            elif trend == "stable" or trend == "elevated_stable":
                regime = 3  # Stagflation - persistent uncertainty
                confidence = 0.7
            elif trend == "declining":
                regime = 4  # Transitioning to calmer markets
                confidence = 0.6
            else:
                regime = 3  # Default to stagflation for elevated VIX
                confidence = 0.6
        
        # Adjust confidence based on regime-specific statistics
        if self._regime_vix_stats:
            stats = self._regime_vix_stats.get(regime, {})
            if stats:
                mean = stats.get("mean", vix)
                std = stats.get("std", 5)
                # Higher confidence if VIX is near learned mean for this regime
                z_score = abs(vix - mean) / max(std, 1)
                confidence = confidence * (1 - min(0.3, z_score * 0.1))
        
        return regime, min(0.95, max(0.3, confidence))
    
    def _calculate_probabilities(
        self,
        vix: float,
        percentile: Optional[float],
    ) -> Dict[int, float]:
        """Calculate probability distribution over regimes."""
        probs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        if self._regime_vix_stats:
            # Use Gaussian likelihood based on learned stats
            total = 0
            for regime, stats in self._regime_vix_stats.items():
                mean = stats["mean"]
                std = max(stats["std"], 1)
                # Gaussian likelihood
                likelihood = np.exp(-0.5 * ((vix - mean) / std) ** 2)
                probs[regime] = likelihood
                total += likelihood
            
            # Normalize
            if total > 0:
                probs = {r: p / total for r, p in probs.items()}
        else:
            # Use rule-based probabilities
            thresholds = self._learned_thresholds
            
            if vix < thresholds.get("low", 15):
                probs = {1: 0.5, 2: 0.05, 3: 0.1, 4: 0.35}
            elif vix < thresholds.get("normal", 20):
                probs = {1: 0.35, 2: 0.1, 3: 0.2, 4: 0.35}
            elif vix < thresholds.get("elevated", 25):
                probs = {1: 0.15, 2: 0.2, 3: 0.4, 4: 0.25}
            elif vix < thresholds.get("high", 30):
                probs = {1: 0.05, 2: 0.35, 3: 0.45, 4: 0.15}
            else:
                probs = {1: 0.02, 2: 0.75, 3: 0.2, 4: 0.03}
        
        return probs
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance for volatility classifier."""
        return {
            "VIX_level": 0.6,
            "VIX_percentile": 0.25,
            "VIX_trend": 0.15,
        }
    
    def get_regime_thresholds(self) -> pd.DataFrame:
        """Get learned and default thresholds.
        
        Returns:
            DataFrame comparing default and learned thresholds
        """
        records = []
        for level in ["low", "normal", "elevated", "high", "extreme"]:
            records.append({
                "Level": level,
                "Default": self.DEFAULT_THRESHOLDS.get(level),
                "Learned": self._learned_thresholds.get(level),
            })
        
        return pd.DataFrame(records)
    
    def get_vix_regime_profile(self) -> pd.DataFrame:
        """Get VIX statistics for each regime.
        
        Returns:
            DataFrame with VIX statistics per regime
        """
        if not self._regime_vix_stats:
            raise ValueError("No regime statistics available. Fit with labels.")
        
        records = []
        for regime in range(1, 5):
            stats = self._regime_vix_stats.get(regime, {})
            records.append({
                "Regime": regime,
                "Regime_Name": self.REGIME_NAMES.get(regime),
                "VIX_Mean": stats.get("mean"),
                "VIX_Std": stats.get("std"),
                "VIX_Median": stats.get("median"),
                "VIX_Min": stats.get("min"),
                "VIX_Max": stats.get("max"),
                "Observation_Count": stats.get("count"),
            })
        
        return pd.DataFrame(records)
