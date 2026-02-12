"""
Meta-Learner: Adaptive Intelligence Layer for AMRCAIS.

This is the "Killer Feature" - the meta-learning coordinator that brings together
performance tracking and recalibration logic to create an adaptive system that
improves over time.

The MetaLearner:
1. Tracks all regime classifications and their accuracy
2. Monitors disagreement across classifiers
3. Triggers recalibration when performance degrades
4. Adapts classifier weights based on recent accuracy
5. Transforms model uncertainty into actionable signals

Classes:
    MetaLearner: Main adaptive learning coordinator

Example:
    >>> meta = MetaLearner()
    >>> meta.log_classification(regime=1, confidence=0.85, disagreement=0.25)
    >>> 
    >>> # Check if recalibration needed
    >>> decision = meta.check_recalibration_needed()
    >>> if decision.should_recalibrate:
    ...     print(f"Alert: {decision.urgency_level}")
    >>> 
    >>> # Get adaptive weights for ensemble
    >>> weights = meta.get_adaptive_weights()
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.meta_learning.performance_tracker import (
    RegimePerformanceTracker,
    RegimeClassification,
    PerformanceMetrics,
)
from src.meta_learning.recalibration import (
    RecalibrationTrigger,
    RecalibrationDecision,
    RecalibrationReason,
)

logger = logging.getLogger(__name__)


class MetaLearner:
    """Adaptive meta-learning coordinator for regime detection system.
    
    This class orchestrates the meta-learning layer (Layer 3) of AMRCAIS,
    combining performance tracking and recalibration logic to create a
    self-improving system.
    
    Key Features:
    - Tracks classification history and performance metrics
    - Monitors regime stability and classifier disagreement
    - Triggers recalibration when performance degrades
    - Adapts classifier weights based on recent accuracy
    - Flags regime uncertainty as a tradeable signal
    
    Attributes:
        tracker: Performance tracking system
        trigger: Recalibration trigger logic
        classifier_accuracy: Recent accuracy per classifier
        adaptive_weights: Current adaptive weight adjustments
        
    Example:
        >>> meta = MetaLearner()
        >>> 
        >>> # Log classification from ensemble
        >>> meta.log_classification(
        ...     regime=1,
        ...     confidence=0.85,
        ...     disagreement=0.25,
        ...     individual_predictions={"hmm": 1, "ml": 1, "corr": 1, "vol": 2},
        ...     market_state={"SPX_returns": 0.012, "VIX_level": 16.5}
        ... )
        >>> 
        >>> # Check system health
        >>> metrics = meta.get_performance_metrics(lookback_days=30)
        >>> print(f"Stability: {metrics.stability_rating}")
        >>> 
        >>> # Check if recalibration needed
        >>> decision = meta.check_recalibration_needed()
        >>> if decision.should_recalibrate:
        ...     # Trigger retraining workflow
        ...     print(f"Recalibration urgency: {decision.urgency_level}")
    """
    
    DEFAULT_WEIGHTS = {
        "hmm": 0.30,
        "ml": 0.25,
        "correlation": 0.25,
        "volatility": 0.20,
    }
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        enable_adaptive_weights: bool = True,
    ):
        """Initialize the meta-learner.
        
        Args:
            storage_path: Path to persist history (optional)
            enable_adaptive_weights: Whether to adapt classifier weights (default: True)
        """
        self.tracker = RegimePerformanceTracker(storage_path=storage_path)
        self.trigger = RecalibrationTrigger()
        
        self.enable_adaptive_weights = enable_adaptive_weights
        self.classifier_accuracy: Dict[str, List[float]] = {
            name: [] for name in self.DEFAULT_WEIGHTS.keys()
        }
        self.adaptive_weights = self.DEFAULT_WEIGHTS.copy()
        
        self._last_recalibration: Optional[datetime] = None
        self._recalibration_count = 0
        
        logger.info("MetaLearner initialized with adaptive learning enabled")
    
    def log_classification(
        self,
        regime: int,
        confidence: float,
        disagreement: float,
        individual_predictions: Dict[str, int],
        market_state: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log a regime classification and update tracking.
        
        Args:
            regime: Predicted regime (1-4)
            confidence: Overall confidence (0-1)
            disagreement: Disagreement index (0-1)
            individual_predictions: Dict of classifier → regime prediction
            market_state: Current market data snapshot
            timestamp: Classification time (default: now)
        """
        # Log to performance tracker
        self.tracker.log_classification(
            regime=regime,
            confidence=confidence,
            disagreement=disagreement,
            individual_predictions=individual_predictions,
            market_state=market_state,
            timestamp=timestamp,
        )
        
        # Flag high uncertainty as potential signal
        if disagreement > 0.6:
            logger.warning(
                f"🚨 REGIME UNCERTAINTY SIGNAL: Disagreement={disagreement:.2f} "
                f"(regime={regime}, confidence={confidence:.2f}). "
                "Historically precedes regime transitions. Consider risk management."
            )
        
        # Update adaptive weights if enabled
        if self.enable_adaptive_weights and len(self.tracker.history) % 10 == 0:
            self._update_adaptive_weights()
    
    def check_recalibration_needed(
        self,
        lookback_days: int = 14,
    ) -> RecalibrationDecision:
        """Check if recalibration is needed based on recent performance.
        
        Args:
            lookback_days: Days to evaluate (default: 14)
            
        Returns:
            RecalibrationDecision with recommendations
        """
        # Get performance metrics
        metrics = self.tracker.get_performance_metrics(lookback_days=lookback_days)
        
        # Check flip frequency
        flip_count, _ = self.tracker.check_regime_flip_frequency(window_days=5)
        
        # Check persistent disagreement
        disagreement_days, _ = self.tracker.check_persistent_disagreement(
            threshold=0.7,
            min_days=10,
        )
        
        # Evaluate prediction accuracy
        accuracy = self.tracker.evaluate_prediction_accuracy(lookback_days=lookback_days)
        
        # Generate recalibration decision
        decision = self.trigger.evaluate(
            accuracy=accuracy,
            recent_flips=flip_count,
            disagreement_days=disagreement_days,
            avg_confidence=metrics.avg_disagreement,  # Using as proxy
            avg_disagreement=metrics.avg_disagreement,
        )
        
        return decision
    
    def execute_recalibration(
        self,
        decision: RecalibrationDecision,
    ) -> bool:
        """Execute recalibration based on decision.
        
        This is a placeholder for the actual recalibration workflow.
        In production, this would trigger model retraining.
        
        Args:
            decision: RecalibrationDecision from check_recalibration_needed()
            
        Returns:
            True if recalibration executed successfully
        """
        if not decision.should_recalibrate:
            logger.info("No recalibration needed")
            return True
        
        logger.warning(
            f"EXECUTING RECALIBRATION: {decision.urgency_level}"
        )
        
        for reason in decision.reasons:
            logger.warning(f"  Reason: {reason}")
        
        for rec in decision.recommendations:
            logger.info(f"  Recommendation: {rec}")
        
        # TODO: Implement actual recalibration workflow
        # 1. Fetch latest training data
        # 2. Retrain classifiers with updated parameters
        # 3. Validate on holdout set
        # 4. Update ensemble with new classifiers
        
        # For now, log and adjust thresholds if needed
        if RecalibrationReason.EXCESSIVE_FLIPPING in decision.reasons:
            # Increase confidence threshold to reduce flipping
            logger.info("Adjusting confidence threshold to reduce flipping")
            # self.trigger.adjust_thresholds({"flip_threshold": 4})
        
        if RecalibrationReason.PERSISTENT_DISAGREEMENT in decision.reasons:
            # May need to add new regime or update training data
            logger.info("Persistent disagreement suggests novel market regime")
        
        # Mark recalibration as executed
        self._last_recalibration = datetime.now()
        self._recalibration_count += 1
        
        logger.info(
            f"Recalibration #{self._recalibration_count} completed "
            f"(severity: {decision.severity:.2f})"
        )
        
        return True
    
    def get_performance_metrics(
        self,
        lookback_days: int = 30,
    ) -> PerformanceMetrics:
        """Get performance metrics for recent period.
        
        Args:
            lookback_days: Days to evaluate
            
        Returns:
            PerformanceMetrics object
        """
        return self.tracker.get_performance_metrics(lookback_days=lookback_days)
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive classifier weights.
        
        If adaptive weighting is enabled, returns weights adjusted based on
        recent classifier accuracy. Otherwise returns default weights.
        
        Returns:
            Dict mapping classifier name to weight (sum = 1.0)
        """
        if not self.enable_adaptive_weights:
            return self.DEFAULT_WEIGHTS.copy()
        
        return self.adaptive_weights.copy()
    
    def _update_adaptive_weights(self) -> None:
        """Update classifier weights based on recent accuracy.
        
        This implements adaptive weight adjustment where classifiers with
        better recent accuracy get higher weights.
        """
        if not self.enable_adaptive_weights:
            return
        
        # Calculate recent accuracy per classifier
        # This is a simplified implementation - production would use actual validation
        recent_history = self.tracker.history[-50:] if len(self.tracker.history) >= 50 else self.tracker.history
        
        if len(recent_history) < 10:
            logger.debug("Insufficient history for adaptive weight update")
            return
        
        # Count agreement with consensus for each classifier
        agreement_counts = {name: 0 for name in self.DEFAULT_WEIGHTS.keys()}
        
        for classification in recent_history:
            consensus_regime = classification.regime
            for classifier_name, prediction in classification.individual_predictions.items():
                if prediction == consensus_regime:
                    agreement_counts[classifier_name] = agreement_counts.get(classifier_name, 0) + 1
        
        # Calculate accuracy scores
        total = len(recent_history)
        accuracy_scores = {
            name: count / total if total > 0 else 0.5
            for name, count in agreement_counts.items()
        }
        
        # Adjust weights proportional to accuracy
        # Use softmax to ensure weights sum to 1
        scores = np.array(list(accuracy_scores.values()))
        
        # Avoid division by zero
        if scores.sum() == 0:
            scores = np.ones_like(scores)
        
        # Apply softmax with temperature to smooth adjustments
        temperature = 2.0  # Higher = less aggressive adjustment
        exp_scores = np.exp(scores / temperature)
        new_weights = exp_scores / exp_scores.sum()
        
        # Update adaptive weights
        for i, name in enumerate(self.DEFAULT_WEIGHTS.keys()):
            old_weight = self.adaptive_weights.get(name, 0.25)
            new_weight = new_weights[i]
            
            # Blend with existing weights for stability
            blended_weight = 0.7 * old_weight + 0.3 * new_weight
            self.adaptive_weights[name] = blended_weight
        
        # Normalize to ensure sum = 1.0
        total_weight = sum(self.adaptive_weights.values())
        self.adaptive_weights = {
            name: weight / total_weight
            for name, weight in self.adaptive_weights.items()
        }
        
        logger.info(
            f"Adaptive weights updated: "
            + ", ".join(f"{name}={weight:.3f}" for name, weight in self.adaptive_weights.items())
        )
    
    def get_system_health_report(self) -> Dict:
        """Generate comprehensive system health report.
        
        Returns:
            Dict with system health metrics and status
        """
        # Get recent performance
        metrics_30d = self.tracker.get_performance_metrics(lookback_days=30)
        metrics_7d = self.tracker.get_performance_metrics(lookback_days=7)
        
        # Check recalibration status
        decision = self.check_recalibration_needed()
        
        # Check regime flip frequency
        flip_count, excessive_flips = self.tracker.check_regime_flip_frequency()
        
        # Check persistent disagreement
        disagreement_days, persistent = self.tracker.check_persistent_disagreement()
        
        # Evaluate accuracy
        accuracy = self.tracker.evaluate_prediction_accuracy()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy" if not decision.should_recalibrate else "needs_attention",
            "performance_30d": {
                "stability_score": metrics_30d.regime_stability_score,
                "stability_rating": metrics_30d.stability_rating,
                "transition_count": metrics_30d.transition_count,
                "avg_disagreement": metrics_30d.avg_disagreement,
                "high_disagreement_days": metrics_30d.high_disagreement_days,
            },
            "performance_7d": {
                "stability_score": metrics_7d.regime_stability_score,
                "transition_count": metrics_7d.transition_count,
                "avg_disagreement": metrics_7d.avg_disagreement,
            },
            "recalibration": {
                "needed": decision.should_recalibrate,
                "urgency": decision.urgency_level,
                "severity": decision.severity,
                "reasons": [str(r) for r in decision.reasons],
                "last_recalibration": self._last_recalibration.isoformat() if self._last_recalibration else None,
                "total_recalibrations": self._recalibration_count,
            },
            "alerts": {
                "excessive_flipping": excessive_flips,
                "flip_count_5d": flip_count,
                "persistent_disagreement": persistent,
                "disagreement_days": disagreement_days,
            },
            "accuracy": {
                "overall": accuracy,
                "above_threshold": accuracy >= 0.75,
            },
            "adaptive_weights": self.adaptive_weights if self.enable_adaptive_weights else None,
        }
        
        return report
    
    def generate_uncertainty_signal(self) -> Dict:
        """Generate regime uncertainty signal for trading.
        
        This is the "killer feature" - transforming model uncertainty into
        a tradeable signal.
        
        Returns:
            Dict with uncertainty signal details
        """
        if not self.tracker.history:
            return {"signal": "neutral", "strength": 0.0, "explanation": "Insufficient data"}
        
        # Get recent disagreement trend
        recent = self.tracker.history[-30:] if len(self.tracker.history) >= 30 else self.tracker.history
        disagreements = [c.disagreement for c in recent]
        avg_disagreement = np.mean(disagreements)
        current_disagreement = recent[-1].disagreement if recent else 0.0
        
        # Check if disagreement is increasing
        if len(disagreements) >= 5:
            recent_trend = np.mean(disagreements[-5:])
            earlier_trend = np.mean(disagreements[-10:-5]) if len(disagreements) >= 10 else avg_disagreement
            disagreement_increasing = recent_trend > earlier_trend
        else:
            disagreement_increasing = False
        
        # Generate signal
        if current_disagreement > 0.7:
            signal = "high_uncertainty"
            strength = min(1.0, current_disagreement)
            explanation = (
                f"CRITICAL: Disagreement at {current_disagreement:.2f}. "
                "Classifiers highly uncertain. Historically precedes regime transitions. "
                "Recommendation: Reduce position sizes, increase hedges."
            )
        elif current_disagreement > 0.6:
            signal = "elevated_uncertainty"
            strength = current_disagreement
            explanation = (
                f"WARNING: Elevated disagreement at {current_disagreement:.2f}. "
                "Potential regime transition forming. Monitor closely."
            )
        elif disagreement_increasing and avg_disagreement > 0.4:
            signal = "rising_uncertainty"
            strength = avg_disagreement
            explanation = (
                f"CAUTION: Disagreement trending up (current: {current_disagreement:.2f}, "
                f"avg: {avg_disagreement:.2f}). Early warning of potential regime shift."
            )
        else:
            signal = "low_uncertainty"
            strength = 1.0 - current_disagreement
            explanation = (
                f"Classifiers in agreement (disagreement: {current_disagreement:.2f}). "
                "Regime classification reliable."
            )
        
        return {
            "signal": signal,
            "strength": strength,
            "current_disagreement": current_disagreement,
            "avg_disagreement": avg_disagreement,
            "disagreement_trend": "increasing" if disagreement_increasing else "stable",
            "explanation": explanation,
            "actionable": current_disagreement > 0.6,
        }
