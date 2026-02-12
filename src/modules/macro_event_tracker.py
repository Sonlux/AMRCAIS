"""
Macro Event Impact Tracker for AMRCAIS.

This module monitors scheduled economic releases (NFP, CPI, FOMC) and measures
market reactions across asset classes. CRITICALLY, it applies regime-dependent
weights to event impact interpretation.

The same macro surprise can be bullish in one regime and bearish in another:
- Risk-On Growth: Strong NFP → bullish (growth confirmation)
- Stagflation: Strong NFP → bearish (more Fed tightening expected)

Classes:
    MacroEventTracker: Regime-adaptive macro event analysis

Example:
    >>> tracker = MacroEventTracker()
    >>> tracker.update_regime(regime=3, confidence=0.8)  # Stagflation
    >>> result = tracker.analyze_event("CPI", actual=3.2, consensus=3.0)
    >>> print(result["signal"])  # "bearish" (inflation surprise in stagflation)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


@dataclass
class MacroEvent:
    """Represents a macroeconomic event/release.
    
    Attributes:
        event_type: Type of event (NFP, CPI, FOMC, etc.)
        release_date: When the data was released
        actual: Actual reported value
        consensus: Market consensus/expectation
        prior: Previous period's value
        surprise: Standardized surprise (actual - consensus) / historical_std
    """
    event_type: str
    release_date: datetime
    actual: float
    consensus: float
    prior: Optional[float] = None
    surprise: Optional[float] = None
    
    def calculate_surprise(self, historical_std: float = 1.0) -> float:
        """Calculate standardized surprise."""
        if historical_std <= 0:
            historical_std = 1.0
        self.surprise = (self.actual - self.consensus) / historical_std
        return self.surprise


class MacroEventTracker(AnalyticalModule):
    """Tracks macro events and interprets their impact by regime.
    
    REGIME-ADAPTIVE INTERPRETATION:
    - Risk-On Growth (1): Strong data → bullish, Fed tightening acceptable
    - Risk-Off Crisis (2): Data largely irrelevant, focus on liquidity/vol
    - Stagflation (3): Strong inflation → bearish, strong growth → bearish (Fed)
    - Disinflationary Boom (4): Weak inflation → bullish, growth → bullish
    
    Attributes:
        event_history: List of past macro events
        regime_weights: Regime-specific weights for each event type
        
    Example:
        >>> tracker = MacroEventTracker()
        >>> tracker.update_regime(1, 0.85)  # Risk-On Growth
        >>> 
        >>> # Analyze NFP release
        >>> result = tracker.analyze_event("NFP", actual=250, consensus=200)
        >>> print(result["signal"].signal)  # "bullish"
        >>> 
        >>> # Now in Stagflation, same NFP surprise would be different
        >>> tracker.update_regime(3, 0.8)
        >>> result = tracker.analyze_event("NFP", actual=250, consensus=200)
        >>> print(result["signal"].signal)  # More nuanced interpretation
    """
    
    # Default regime weights for event types
    DEFAULT_REGIME_WEIGHTS = {
        1: {"NFP": 1.2, "CPI": 0.8, "FOMC": 1.0, "PMI": 1.1, "GDP": 0.9},
        2: {"NFP": 0.3, "CPI": 0.5, "FOMC": 1.5, "PMI": 0.4, "GDP": 0.3},
        3: {"NFP": 0.7, "CPI": 1.5, "FOMC": 1.2, "PMI": 0.8, "GDP": 0.6},
        4: {"NFP": 1.0, "CPI": 0.6, "FOMC": 1.3, "PMI": 0.9, "GDP": 1.0},
    }
    
    # Interpretation rules by regime
    EVENT_INTERPRETATIONS = {
        "NFP": {
            1: {"positive_surprise": "bullish", "negative_surprise": "bearish"},
            2: {"positive_surprise": "neutral", "negative_surprise": "neutral"},
            3: {"positive_surprise": "mixed", "negative_surprise": "bullish"},
            4: {"positive_surprise": "bullish", "negative_surprise": "neutral"},
        },
        "CPI": {
            1: {"positive_surprise": "neutral", "negative_surprise": "bullish"},
            2: {"positive_surprise": "neutral", "negative_surprise": "neutral"},
            3: {"positive_surprise": "bearish", "negative_surprise": "bullish"},
            4: {"positive_surprise": "bearish", "negative_surprise": "bullish"},
        },
        "FOMC": {
            1: {"hawkish": "neutral", "dovish": "bullish"},
            2: {"hawkish": "bearish", "dovish": "bullish"},
            3: {"hawkish": "bearish", "dovish": "bullish"},
            4: {"hawkish": "bearish", "dovish": "bullish"},
        },
    }
    
    # Historical standard deviations for surprise calculation
    HISTORICAL_STDS = {
        "NFP": 75.0,  # thousands of jobs
        "CPI": 0.1,   # percentage points YoY
        "CORE_CPI": 0.1,
        "PPI": 0.2,
        "PMI": 1.5,
        "GDP": 0.5,
        "RETAIL_SALES": 0.3,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the macro event tracker."""
        super().__init__(name="MacroEventTracker", config_path=config_path)
        
        self.event_history: List[MacroEvent] = []
        self._market_reactions: Dict[str, List[Dict]] = {}
        
        # Load regime weights from config or use defaults
        self._load_regime_weights()
    
    def _load_regime_weights(self) -> None:
        """Load regime-specific event weights."""
        self.regime_weights = {}
        
        for regime in range(1, 5):
            regime_config = self._regime_params.get(regime, {})
            weights = regime_config.get(
                "macro_event_weights",
                self.DEFAULT_REGIME_WEIGHTS.get(regime, {})
            )
            self.regime_weights[regime] = weights
    
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get macro event parameters for a specific regime.
        
        Args:
            regime: Regime ID (1-4)
            
        Returns:
            Dictionary with event weights and interpretation rules
        """
        return {
            "weights": self.regime_weights.get(regime, {}),
            "interpretations": {
                event: rules.get(regime, {})
                for event, rules in self.EVENT_INTERPRETATIONS.items()
            },
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze macro data for signals.
        
        Args:
            data: DataFrame with macro indicator columns
            
        Returns:
            Analysis results with regime-adaptive signals
        """
        signals = []
        
        # Analyze each macro indicator if present
        for indicator in ["NFP", "CPI", "PMI", "GDP"]:
            if indicator in data.columns:
                # Get recent values
                recent = data[indicator].dropna().iloc[-2:]
                if len(recent) >= 2:
                    change = recent.iloc[-1] - recent.iloc[-2]
                    signal = self._interpret_indicator_change(indicator, change)
                    signals.append(signal)
        
        # Aggregate signals
        if signals:
            avg_strength = np.mean([s.strength for s in signals])
            bullish_count = sum(1 for s in signals if s.signal == "bullish")
            bearish_count = sum(1 for s in signals if s.signal == "bearish")
            
            if bullish_count > bearish_count:
                overall_signal = "bullish"
            elif bearish_count > bullish_count:
                overall_signal = "bearish"
            else:
                overall_signal = "neutral"
            
            return {
                "signal": self.create_signal(
                    signal=overall_signal,
                    strength=avg_strength,
                    explanation=f"Aggregate of {len(signals)} macro indicators",
                    individual_signals=[s.to_dict() for s in signals],
                ),
                "individual_signals": signals,
                "regime_parameters": self.get_current_parameters(),
            }
        
        return {
            "signal": self.create_signal(
                signal="neutral",
                strength=0.0,
                explanation="No macro indicators available for analysis",
            ),
            "regime_parameters": self.get_current_parameters(),
        }
    
    def analyze_event(
        self,
        event_type: str,
        actual: float,
        consensus: float,
        prior: Optional[float] = None,
        release_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Analyze a specific macro event release.
        
        CRITICAL: Interpretation is regime-dependent!
        
        Args:
            event_type: Type of event (NFP, CPI, FOMC, etc.)
            actual: Actual released value
            consensus: Market consensus expectation
            prior: Previous period's value (optional)
            release_date: When the event occurred (defaults to now)
            
        Returns:
            Dictionary with:
                - signal: ModuleSignal with regime-adapted interpretation
                - surprise: Standardized surprise value
                - regime_weight: How much this event matters in current regime
                - interpretation: Detailed regime-specific interpretation
        """
        if release_date is None:
            release_date = datetime.now()
        
        # Create event record
        event = MacroEvent(
            event_type=event_type,
            release_date=release_date,
            actual=actual,
            consensus=consensus,
            prior=prior,
        )
        
        # Calculate surprise
        historical_std = self.HISTORICAL_STDS.get(event_type, 1.0)
        surprise = event.calculate_surprise(historical_std)
        
        # Get regime-specific weight and interpretation
        regime_weight = self.regime_weights.get(
            self.current_regime, {}
        ).get(event_type, 1.0)
        
        # Interpret the surprise based on regime
        signal, explanation = self._interpret_surprise(event_type, surprise)
        
        # Adjust strength by regime weight
        raw_strength = min(1.0, abs(surprise) / 2.0)  # Normalize surprise
        weighted_strength = raw_strength * regime_weight
        
        # Store event
        self.event_history.append(event)
        
        result_signal = self.create_signal(
            signal=signal,
            strength=min(1.0, weighted_strength),
            explanation=explanation,
            event_type=event_type,
            actual=actual,
            consensus=consensus,
            surprise=surprise,
            regime_weight=regime_weight,
        )
        
        logger.info(
            f"MacroEvent: {event_type} actual={actual}, consensus={consensus}, "
            f"surprise={surprise:.2f}, signal={signal} in "
            f"{self.REGIME_NAMES[self.current_regime]}"
        )
        
        return {
            "signal": result_signal,
            "surprise": surprise,
            "regime_weight": regime_weight,
            "interpretation": explanation,
            "event": event,
        }
    
    def _interpret_surprise(
        self,
        event_type: str,
        surprise: float,
    ) -> Tuple[str, str]:
        """Interpret surprise value based on current regime.
        
        Args:
            event_type: Type of macro event
            surprise: Standardized surprise value
            
        Returns:
            Tuple of (signal, explanation)
        """
        interpretations = self.EVENT_INTERPRETATIONS.get(event_type, {})
        regime_rules = interpretations.get(self.current_regime, {})
        
        regime_name = self.REGIME_NAMES[self.current_regime]
        
        # Determine direction of surprise
        if surprise > 0.5:  # Positive surprise
            signal = regime_rules.get("positive_surprise", "neutral")
            direction = "above"
        elif surprise < -0.5:  # Negative surprise
            signal = regime_rules.get("negative_surprise", "neutral")
            direction = "below"
        else:  # Inline with expectations
            signal = "neutral"
            direction = "inline with"
        
        # Handle special signals
        if signal == "mixed":
            # In stagflation, strong growth is mixed (good growth, but more Fed)
            signal = "neutral"
            explanation = (
                f"{event_type} came in {direction} expectations. "
                f"In {regime_name}, this has mixed implications "
                "(growth vs. Fed policy concerns)"
            )
        else:
            explanation = (
                f"{event_type} surprise of {surprise:.2f} standard deviations. "
                f"In {regime_name}, this is interpreted as {signal}"
            )
        
        return signal, explanation
    
    def _interpret_indicator_change(
        self,
        indicator: str,
        change: float,
    ) -> ModuleSignal:
        """Interpret change in a macro indicator."""
        # Simple interpretation based on regime
        if indicator in ["NFP", "GDP", "PMI", "RETAIL_SALES"]:
            # Growth indicators
            if self.current_regime == 1:  # Risk-On
                signal = "bullish" if change > 0 else "bearish"
            elif self.current_regime == 3:  # Stagflation
                # In stagflation, strong growth means more Fed tightening
                signal = "neutral" if change > 0 else "bullish"
            else:
                signal = "bullish" if change > 0 else "neutral"
        elif indicator in ["CPI", "PPI", "CORE_CPI"]:
            # Inflation indicators
            if self.current_regime == 3:  # Stagflation
                signal = "bearish" if change > 0 else "bullish"
            else:
                signal = "neutral" if change > 0 else "bullish"
        else:
            signal = "neutral"
        
        return self.create_signal(
            signal=signal,
            strength=min(1.0, abs(change) * 10),  # Scale appropriately
            explanation=f"{indicator} changed by {change:.2f}",
        )
    
    def track_market_reaction(
        self,
        event_type: str,
        price_data: pd.DataFrame,
        event_time: datetime,
        pre_window_minutes: int = 30,
        post_window_minutes: int = 180,
    ) -> Dict[str, float]:
        """Track market reaction to a macro event.
        
        Args:
            event_type: Type of event
            price_data: DataFrame with price data (OHLC for multiple assets)
            event_time: When the event occurred
            pre_window_minutes: Minutes before event to capture
            post_window_minutes: Minutes after event to capture
            
        Returns:
            Dictionary with price reactions for each asset
        """
        reactions = {}
        
        pre_start = event_time - timedelta(minutes=pre_window_minutes)
        post_end = event_time + timedelta(minutes=post_window_minutes)
        
        for asset in price_data.columns:
            if "Close" in asset or len(price_data.columns) == 1:
                col = asset
            else:
                continue
            
            try:
                # Get pre-event price
                pre_data = price_data[col][pre_start:event_time]
                if len(pre_data) > 0:
                    pre_price = pre_data.iloc[-1]
                else:
                    continue
                
                # Get post-event price
                post_data = price_data[col][event_time:post_end]
                if len(post_data) > 0:
                    post_price = post_data.iloc[-1]
                else:
                    continue
                
                # Calculate reaction
                reaction_pct = (post_price - pre_price) / pre_price * 100
                reactions[col] = reaction_pct
                
            except Exception as e:
                logger.warning(f"Could not calculate reaction for {asset}: {e}")
        
        # Store reaction
        if event_type not in self._market_reactions:
            self._market_reactions[event_type] = []
        
        self._market_reactions[event_type].append({
            "event_time": event_time,
            "reactions": reactions,
            "regime": self.current_regime,
        })
        
        return reactions
    
    def get_event_statistics(self, event_type: str) -> Optional[Dict]:
        """Get historical statistics for an event type.
        
        Args:
            event_type: Type of macro event
            
        Returns:
            Dictionary with surprise statistics and market reactions
        """
        events = [e for e in self.event_history if e.event_type == event_type]
        
        if not events:
            return None
        
        surprises = [e.surprise for e in events if e.surprise is not None]
        
        return {
            "count": len(events),
            "surprise_mean": np.mean(surprises) if surprises else None,
            "surprise_std": np.std(surprises) if surprises else None,
            "positive_surprise_count": sum(1 for s in surprises if s > 0.5),
            "negative_surprise_count": sum(1 for s in surprises if s < -0.5),
        }
