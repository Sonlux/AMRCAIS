"""
Performance Tracker for Regime Classification.

This module tracks the accuracy of regime classifications over time and
provides metrics to evaluate whether the regime detection is working correctly.

Classes:
    RegimeClassification: Single regime classification record
    PerformanceMetrics: Aggregated performance statistics
    RegimePerformanceTracker: Main tracking and evaluation class
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimeClassification:
    """Record of a single regime classification.
    
    Attributes:
        timestamp: When the classification was made
        regime: Predicted regime (1-4)
        confidence: Classifier confidence (0-1)
        disagreement: Disagreement index across classifiers (0-1)
        individual_predictions: Dict of classifier name to regime prediction
        market_state: Market data snapshot at classification time
    """
    timestamp: datetime
    regime: int
    confidence: float
    disagreement: float
    individual_predictions: Dict[str, int] = field(default_factory=dict)
    market_state: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "individual_predictions": self.individual_predictions,
            "market_state": self.market_state,
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics.
    
    Attributes:
        period_start: Start of evaluation period
        period_end: End of evaluation period
        total_classifications: Total number of classifications made
        regime_stability_score: How stable regime classifications are (0-1)
        transition_count: Number of regime transitions
        avg_disagreement: Average disagreement index
        high_disagreement_days: Days with disagreement >0.6
        regime_distribution: Distribution of regimes in period
    """
    period_start: datetime
    period_end: datetime
    total_classifications: int
    regime_stability_score: float
    transition_count: int
    avg_disagreement: float
    high_disagreement_days: int
    regime_distribution: Dict[int, int] = field(default_factory=dict)
    
    @property
    def stability_rating(self) -> str:
        """Human-readable stability rating."""
        if self.regime_stability_score > 0.8:
            return "High Stability"
        elif self.regime_stability_score > 0.6:
            return "Moderate Stability"
        elif self.regime_stability_score > 0.4:
            return "Low Stability"
        else:
            return "Very Low Stability - Consider Recalibration"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_classifications": self.total_classifications,
            "regime_stability_score": self.regime_stability_score,
            "stability_rating": self.stability_rating,
            "transition_count": self.transition_count,
            "avg_disagreement": self.avg_disagreement,
            "high_disagreement_days": self.high_disagreement_days,
            "regime_distribution": self.regime_distribution,
        }


class RegimePerformanceTracker:
    """Tracks regime classification performance over time.
    
    This class maintains a history of regime classifications and provides
    methods to evaluate accuracy, stability, and identify when recalibration
    may be needed.
    
    Attributes:
        history: List of all regime classifications
        storage_path: Path to persist classification history
        
    Example:
        >>> tracker = RegimePerformanceTracker()
        >>> tracker.log_classification(
        ...     regime=1,
        ...     confidence=0.85,
        ...     disagreement=0.25,
        ...     individual_predictions={"hmm": 1, "ml": 1, "corr": 1, "vol": 2}
        ... )
        >>> metrics = tracker.get_performance_metrics(lookback_days=30)
        >>> print(f"Stability: {metrics.stability_rating}")
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the performance tracker.
        
        Args:
            storage_path: Path to persist classification history (optional)
        """
        self.history: List[RegimeClassification] = []
        self.storage_path = storage_path or Path("data/regime_history.csv")
        
        # Load existing history if available
        self._load_history()
        
        logger.info(f"RegimePerformanceTracker initialized with {len(self.history)} historical records")
    
    def log_classification(
        self,
        regime: int,
        confidence: float,
        disagreement: float,
        individual_predictions: Dict[str, int],
        market_state: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a regime classification.
        
        Args:
            regime: Predicted regime (1-4)
            confidence: Classifier confidence (0-1)
            disagreement: Disagreement index (0-1)
            individual_predictions: Dict mapping classifier name to prediction
            market_state: Optional market data snapshot
            timestamp: Classification timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        classification = RegimeClassification(
            timestamp=timestamp,
            regime=regime,
            confidence=confidence,
            disagreement=disagreement,
            individual_predictions=individual_predictions,
            market_state=market_state or {},
        )
        
        self.history.append(classification)
        
        # Log regime changes
        if len(self.history) > 1:
            prev_regime = self.history[-2].regime
            if prev_regime != regime:
                logger.info(
                    f"REGIME CHANGE: {prev_regime} → {regime} "
                    f"(confidence={confidence:.2f}, disagreement={disagreement:.2f})"
                )
            elif disagreement > 0.6:
                logger.warning(
                    f"HIGH DISAGREEMENT: {disagreement:.2f} "
                    f"(regime={regime}, confidence={confidence:.2f}) "
                    "- May indicate imminent regime transition"
                )
        
        # Periodically save to disk
        if len(self.history) % 10 == 0:
            self._save_history()
    
    def get_performance_metrics(
        self,
        lookback_days: int = 30,
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """Calculate performance metrics for a specific period.
        
        Args:
            lookback_days: Number of days to look back
            end_date: End date for evaluation (default: now)
            
        Returns:
            PerformanceMetrics object with aggregated statistics
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=lookback_days)
        
        # Filter history to period
        period_history = [
            c for c in self.history
            if start_date <= c.timestamp <= end_date
        ]
        
        if not period_history:
            logger.warning(f"No classifications found in period {start_date} to {end_date}")
            return PerformanceMetrics(
                period_start=start_date,
                period_end=end_date,
                total_classifications=0,
                regime_stability_score=0.0,
                transition_count=0,
                avg_disagreement=0.0,
                high_disagreement_days=0,
            )
        
        # Calculate metrics
        total = len(period_history)
        regimes = [c.regime for c in period_history]
        
        # Count transitions
        transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        
        # Stability score: 1.0 - (transitions / possible_transitions)
        possible_transitions = max(1, len(regimes) - 1)
        stability_score = 1.0 - (transitions / possible_transitions)
        
        # Average disagreement
        avg_disagreement = np.mean([c.disagreement for c in period_history])
        
        # High disagreement days
        high_disagreement = sum(1 for c in period_history if c.disagreement > 0.6)
        
        # Regime distribution
        unique, counts = np.unique(regimes, return_counts=True)
        regime_dist = dict(zip(unique.tolist(), counts.tolist()))
        
        metrics = PerformanceMetrics(
            period_start=start_date,
            period_end=end_date,
            total_classifications=total,
            regime_stability_score=stability_score,
            transition_count=transitions,
            avg_disagreement=avg_disagreement,
            high_disagreement_days=high_disagreement,
            regime_distribution=regime_dist,
        )
        
        logger.info(
            f"Performance metrics (last {lookback_days} days): "
            f"Stability={stability_score:.2f}, Transitions={transitions}, "
            f"Avg Disagreement={avg_disagreement:.2f}"
        )
        
        return metrics
    
    def check_regime_flip_frequency(self, window_days: int = 5) -> Tuple[int, bool]:
        """Check if regime is flipping too frequently.
        
        Args:
            window_days: Window to check (default: 5 days)
            
        Returns:
            Tuple of (flip_count, is_excessive)
            is_excessive is True if >3 flips in the window
        """
        if len(self.history) < 2:
            return 0, False
        
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = [c for c in self.history if c.timestamp >= cutoff]
        
        if len(recent) < 2:
            return 0, False
        
        flips = sum(
            1 for i in range(1, len(recent))
            if recent[i].regime != recent[i-1].regime
        )
        
        is_excessive = flips > 3
        
        if is_excessive:
            logger.warning(
                f"EXCESSIVE REGIME FLIPS: {flips} flips in past {window_days} days. "
                "Consider increasing confidence threshold."
            )
        
        return flips, is_excessive
    
    def check_persistent_disagreement(
        self,
        threshold: float = 0.7,
        min_days: int = 10,
    ) -> Tuple[int, bool]:
        """Check for persistent high disagreement across classifiers.
        
        Args:
            threshold: Disagreement threshold (default: 0.7)
            min_days: Minimum days of persistence (default: 10)
            
        Returns:
            Tuple of (days_high_disagreement, is_persistent)
        """
        if not self.history:
            return 0, False
        
        # Count consecutive days with high disagreement
        consecutive_days = 0
        for classification in reversed(self.history):
            if classification.disagreement > threshold:
                consecutive_days += 1
            else:
                break
        
        is_persistent = consecutive_days >= min_days
        
        if is_persistent:
            logger.warning(
                f"PERSISTENT DISAGREEMENT: {consecutive_days} consecutive days "
                f"with disagreement >{threshold:.1f}. Manual review recommended."
            )
        
        return consecutive_days, is_persistent
    
    def evaluate_prediction_accuracy(
        self,
        lookback_days: int = 14,
    ) -> float:
        """Evaluate regime prediction accuracy using market behavior.
        
        Uses market behavior consistency as ground truth:
        - Risk-On: SPX should generally be up, VIX down
        - Risk-Off: SPX down, VIX up, correlations spike
        - Stagflation: Commodities up, equities flat
        - Disinflationary Boom: Both equities and bonds up
        
        Args:
            lookback_days: Period to evaluate (default: 14 days)
            
        Returns:
            Accuracy score (0-1), higher means predictions match market behavior
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [c for c in self.history if c.timestamp >= cutoff]
        
        if not recent:
            return 0.5  # No data = neutral score
        
        # This is a simplified heuristic - in production, use actual market validation
        consistent_predictions = 0
        
        for classification in recent:
            market = classification.market_state
            regime = classification.regime
            
            if not market:
                continue  # Can't validate without market data
            
            # Simplified consistency checks
            is_consistent = False
            
            if regime == 1:  # Risk-On Growth
                is_consistent = market.get("SPX_returns", 0) > 0
            elif regime == 2:  # Risk-Off Crisis
                is_consistent = market.get("VIX_level", 15) > 25
            elif regime == 3:  # Stagflation
                is_consistent = market.get("WTI_returns", 0) > 0
            elif regime == 4:  # Disinflationary Boom
                is_consistent = (
                    market.get("SPX_returns", 0) > 0 and
                    market.get("TLT_returns", 0) > 0
                )
            
            if is_consistent:
                consistent_predictions += 1
        
        accuracy = consistent_predictions / len(recent) if recent else 0.5
        
        logger.info(f"Prediction accuracy (last {lookback_days} days): {accuracy:.1%}")
        
        return accuracy
    
    def get_classification_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get classification history as DataFrame.
        
        Args:
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with classification history
        """
        if not self.history:
            return pd.DataFrame()
        
        # Filter by date range
        filtered = self.history
        if start_date:
            filtered = [c for c in filtered if c.timestamp >= start_date]
        if end_date:
            filtered = [c for c in filtered if c.timestamp <= end_date]
        
        # Convert to DataFrame
        data = {
            "timestamp": [c.timestamp for c in filtered],
            "regime": [c.regime for c in filtered],
            "confidence": [c.confidence for c in filtered],
            "disagreement": [c.disagreement for c in filtered],
        }
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def _load_history(self) -> None:
        """Load classification history from disk."""
        if not self.storage_path.exists():
            logger.debug("No existing history file found")
            return
        
        try:
            df = pd.read_csv(self.storage_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            for _, row in df.iterrows():
                classification = RegimeClassification(
                    timestamp=row["timestamp"],
                    regime=int(row["regime"]),
                    confidence=float(row["confidence"]),
                    disagreement=float(row["disagreement"]),
                )
                self.history.append(classification)
            
            logger.info(f"Loaded {len(self.history)} historical classifications")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    
    def _save_history(self) -> None:
        """Save classification history to disk."""
        if not self.history:
            return
        
        try:
            # Create parent directory if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to DataFrame and save
            df = self.get_classification_history()
            df.to_csv(self.storage_path)
            
            logger.debug(f"Saved {len(self.history)} classifications to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def clear_history(self) -> None:
        """Clear all classification history (use with caution)."""
        self.history.clear()
        logger.warning("Classification history cleared")
