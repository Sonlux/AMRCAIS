"""
Recalibration Trigger Logic for AMRCAIS.

This module determines when regime classifiers need to be retrained based on
performance degradation, excessive regime flipping, or persistent disagreement.

Classes:
    RecalibrationReason: Enum of reasons for recalibration
    RecalibrationTrigger: Main logic for determining when to recalibrate

Example:
    >>> trigger = RecalibrationTrigger()
    >>> should_recal, reasons = trigger.should_recalibrate(
    ...     performance_metrics=metrics,
    ...     flip_count=5,
    ...     persistent_disagreement_days=12
    ... )
    >>> if should_recal:
    ...     print(f"Recalibration needed: {reasons}")
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RecalibrationReason(Enum):
    """Reasons for triggering recalibration."""
    
    HIGH_ERROR_RATE = "high_error_rate"  # Accuracy <75% over 2 weeks
    EXCESSIVE_FLIPPING = "excessive_flipping"  # >3 regime changes in 5 days
    PERSISTENT_DISAGREEMENT = "persistent_disagreement"  # Disagreement >0.7 for 10+ days
    LOW_CONFIDENCE = "low_confidence"  # Average confidence <0.5 for extended period
    MARKET_REGIME_MISMATCH = "market_regime_mismatch"  # Predicted regime inconsistent with market
    
    def __str__(self) -> str:
        """Human-readable description."""
        descriptions = {
            self.HIGH_ERROR_RATE: "Prediction accuracy below 75% threshold",
            self.EXCESSIVE_FLIPPING: "Too many regime transitions in short period",
            self.PERSISTENT_DISAGREEMENT: "Classifiers disagree for extended period",
            self.LOW_CONFIDENCE: "Average confidence scores too low",
            self.MARKET_REGIME_MISMATCH: "Regime predictions inconsistent with market behavior",
        }
        return descriptions.get(self, self.value)


@dataclass
class RecalibrationDecision:
    """Decision on whether to recalibrate.
    
    Attributes:
        should_recalibrate: Whether recalibration is recommended
        reasons: List of reasons triggering recalibration
        severity: Severity score (0-1, higher = more urgent)
        timestamp: When decision was made
        recommendations: Suggested actions
    """
    should_recalibrate: bool
    reasons: List[RecalibrationReason]
    severity: float
    timestamp: datetime
    recommendations: List[str]
    
    @property
    def urgency_level(self) -> str:
        """Human-readable urgency level."""
        if self.severity >= 0.8:
            return "CRITICAL - Immediate recalibration required"
        elif self.severity >= 0.6:
            return "HIGH - Recalibrate within 24 hours"
        elif self.severity >= 0.4:
            return "MEDIUM - Schedule recalibration soon"
        elif self.severity >= 0.2:
            return "LOW - Monitor and consider recalibration"
        else:
            return "NONE - System performing normally"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "should_recalibrate": self.should_recalibrate,
            "reasons": [str(r) for r in self.reasons],
            "severity": self.severity,
            "urgency_level": self.urgency_level,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
        }


class RecalibrationTrigger:
    """Determines when regime classifiers need recalibration.
    
    This class implements the recalibration logic defined in the Development Rules:
    - TRIGGER: Regime changes >3 times in 5 days
    - TRIGGER: Disagreement stays >0.7 for >10 days
    - TRIGGER: Error rate >25% over 2-week period
    
    Attributes:
        error_rate_threshold: Accuracy threshold below which to trigger (default: 0.75)
        flip_threshold: Max regime flips in window (default: 3)
        flip_window_days: Window for counting flips (default: 5)
        disagreement_threshold: Disagreement level triggering concern (default: 0.7)
        disagreement_persistence_days: Days of persistence triggering recalibration (default: 10)
        
    Example:
        >>> trigger = RecalibrationTrigger()
        >>> decision = trigger.evaluate(
        ...     accuracy=0.68,
        ...     recent_flips=5,
        ...     disagreement_days=12,
        ...     avg_confidence=0.55
        ... )
        >>> if decision.should_recalibrate:
        ...     print(f"{decision.urgency_level}")
        ...     for reason in decision.reasons:
        ...         print(f"  - {reason}")
    """
    
    def __init__(
        self,
        error_rate_threshold: float = 0.75,
        flip_threshold: int = 3,
        flip_window_days: int = 5,
        disagreement_threshold: float = 0.7,
        disagreement_persistence_days: int = 10,
        low_confidence_threshold: float = 0.5,
    ):
        """Initialize recalibration trigger.
        
        Args:
            error_rate_threshold: Min acceptable accuracy (0-1)
            flip_threshold: Max regime flips in window
            flip_window_days: Window for counting flips (days)
            disagreement_threshold: High disagreement level
            disagreement_persistence_days: Days of persistence to trigger
            low_confidence_threshold: Min acceptable confidence
        """
        self.error_rate_threshold = error_rate_threshold
        self.flip_threshold = flip_threshold
        self.flip_window_days = flip_window_days
        self.disagreement_threshold = disagreement_threshold
        self.disagreement_persistence_days = disagreement_persistence_days
        self.low_confidence_threshold = low_confidence_threshold
        
        logger.info(
            f"RecalibrationTrigger initialized with thresholds: "
            f"accuracy>{error_rate_threshold:.0%}, "
            f"flips<={flip_threshold}/{flip_window_days}days, "
            f"disagreement<{disagreement_threshold:.1f}"
        )
    
    def evaluate(
        self,
        accuracy: float,
        recent_flips: int,
        disagreement_days: int,
        avg_confidence: float,
        avg_disagreement: Optional[float] = None,
    ) -> RecalibrationDecision:
        """Evaluate whether recalibration is needed.
        
        Args:
            accuracy: Recent prediction accuracy (0-1)
            recent_flips: Number of regime flips in recent window
            disagreement_days: Consecutive days of high disagreement
            avg_confidence: Average classifier confidence (0-1)
            avg_disagreement: Average disagreement index (optional)
            
        Returns:
            RecalibrationDecision with recommendation and reasons
        """
        reasons = []
        severity_scores = []
        recommendations = []
        
        # Check 1: High error rate (low accuracy)
        if accuracy < self.error_rate_threshold:
            reasons.append(RecalibrationReason.HIGH_ERROR_RATE)
            severity = 1.0 - accuracy  # Lower accuracy = higher severity
            severity_scores.append(severity)
            recommendations.append(
                f"Accuracy at {accuracy:.1%} (threshold: {self.error_rate_threshold:.0%}). "
                "Retrain classifiers with recent data."
            )
            logger.warning(
                f"HIGH ERROR RATE: Accuracy {accuracy:.1%} below threshold "
                f"{self.error_rate_threshold:.0%}"
            )
        
        # Check 2: Excessive regime flipping
        if recent_flips > self.flip_threshold:
            reasons.append(RecalibrationReason.EXCESSIVE_FLIPPING)
            severity = min(1.0, (recent_flips - self.flip_threshold) / 5)
            severity_scores.append(severity)
            recommendations.append(
                f"{recent_flips} regime flips detected in {self.flip_window_days} days. "
                "Consider increasing confidence threshold or smoothing predictions."
            )
            logger.warning(
                f"EXCESSIVE FLIPPING: {recent_flips} flips in "
                f"{self.flip_window_days} days (threshold: {self.flip_threshold})"
            )
        
        # Check 3: Persistent disagreement
        if disagreement_days >= self.disagreement_persistence_days:
            reasons.append(RecalibrationReason.PERSISTENT_DISAGREEMENT)
            severity = min(1.0, disagreement_days / (self.disagreement_persistence_days * 2))
            severity_scores.append(severity)
            recommendations.append(
                f"Classifiers disagree for {disagreement_days} consecutive days. "
                "May indicate new regime not captured in training data."
            )
            logger.warning(
                f"PERSISTENT DISAGREEMENT: {disagreement_days} days "
                f"(threshold: {self.disagreement_persistence_days})"
            )
        
        # Check 4: Low confidence
        if avg_confidence < self.low_confidence_threshold:
            reasons.append(RecalibrationReason.LOW_CONFIDENCE)
            severity = 1.0 - avg_confidence
            severity_scores.append(severity)
            recommendations.append(
                f"Average confidence at {avg_confidence:.2f} "
                f"(threshold: {self.low_confidence_threshold:.2f}). "
                "Classifiers uncertain about current regime."
            )
            logger.warning(
                f"LOW CONFIDENCE: {avg_confidence:.2f} below threshold "
                f"{self.low_confidence_threshold:.2f}"
            )
        
        # Check 5: Market behavior mismatch (if we have disagreement metric)
        if avg_disagreement is not None and avg_disagreement > 0.8:
            reasons.append(RecalibrationReason.MARKET_REGIME_MISMATCH)
            severity = avg_disagreement
            severity_scores.append(severity)
            recommendations.append(
                f"High average disagreement ({avg_disagreement:.2f}) suggests "
                "market may be in novel regime not seen during training."
            )
        
        # Determine if recalibration is needed
        should_recalibrate = len(reasons) > 0
        
        # Calculate overall severity (max of individual severities)
        overall_severity = max(severity_scores) if severity_scores else 0.0
        
        # Add general recommendation if recalibration needed
        if should_recalibrate:
            if overall_severity >= 0.8:
                recommendations.insert(
                    0,
                    "CRITICAL: Immediate recalibration required. "
                    "System reliability compromised."
                )
            elif len(reasons) >= 3:
                recommendations.insert(
                    0,
                    "Multiple issues detected. Comprehensive recalibration recommended."
                )
        else:
            recommendations.append("System performing within acceptable parameters.")
        
        decision = RecalibrationDecision(
            should_recalibrate=should_recalibrate,
            reasons=reasons,
            severity=overall_severity,
            timestamp=datetime.now(),
            recommendations=recommendations,
        )
        
        if should_recalibrate:
            logger.warning(
                f"RECALIBRATION RECOMMENDED: {decision.urgency_level}"
            )
            for reason in reasons:
                logger.warning(f"  Reason: {reason}")
        else:
            logger.info("No recalibration needed - system stable")
        
        return decision
    
    def check_individual_trigger(
        self,
        trigger_type: str,
        value: float,
    ) -> Tuple[bool, float]:
        """Check a single trigger condition.
        
        Args:
            trigger_type: Type of trigger to check
            value: Current value to evaluate
            
        Returns:
            Tuple of (is_triggered, severity)
        """
        if trigger_type == "accuracy":
            is_triggered = value < self.error_rate_threshold
            severity = 1.0 - value if is_triggered else 0.0
            return is_triggered, severity
        
        elif trigger_type == "flips":
            is_triggered = value > self.flip_threshold
            severity = min(1.0, (value - self.flip_threshold) / 5) if is_triggered else 0.0
            return is_triggered, severity
        
        elif trigger_type == "disagreement_days":
            is_triggered = value >= self.disagreement_persistence_days
            severity = min(1.0, value / (self.disagreement_persistence_days * 2)) if is_triggered else 0.0
            return is_triggered, severity
        
        elif trigger_type == "confidence":
            is_triggered = value < self.low_confidence_threshold
            severity = 1.0 - value if is_triggered else 0.0
            return is_triggered, severity
        
        else:
            logger.warning(f"Unknown trigger type: {trigger_type}")
            return False, 0.0
    
    def adjust_thresholds(
        self,
        new_thresholds: Dict[str, float],
    ) -> None:
        """Dynamically adjust recalibration thresholds.
        
        Args:
            new_thresholds: Dict mapping threshold names to new values
        """
        for key, value in new_thresholds.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Adjusted threshold '{key}': {old_value} → {value}")
            else:
                logger.warning(f"Unknown threshold: {key}")
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current recalibration thresholds.
        
        Returns:
            Dict of threshold names to values
        """
        return {
            "error_rate_threshold": self.error_rate_threshold,
            "flip_threshold": self.flip_threshold,
            "flip_window_days": self.flip_window_days,
            "disagreement_threshold": self.disagreement_threshold,
            "disagreement_persistence_days": self.disagreement_persistence_days,
            "low_confidence_threshold": self.low_confidence_threshold,
        }
