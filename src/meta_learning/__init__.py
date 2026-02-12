"""
Meta-Learning Module for AMRCAIS.

This module implements Layer 3: Meta-Learning & Adaptation, the "killer feature"
that tracks regime classification accuracy, monitors disagreement across
classifiers, and triggers recalibration when errors exceed thresholds.

The meta-learning layer transforms model uncertainty from a weakness into
a tradeable insight by flagging when classifiers disagree (Disagreement Index >0.6),
which historically precedes major market transitions.

Classes:
    RegimePerformanceTracker: Tracks regime classification accuracy
    RecalibrationTrigger: Determines when models need retraining
    MetaLearner: Main coordinator for adaptive learning
"""

from src.meta_learning.performance_tracker import RegimePerformanceTracker
from src.meta_learning.recalibration import RecalibrationTrigger, RecalibrationReason
from src.meta_learning.meta_learner import MetaLearner

__all__ = [
    "RegimePerformanceTracker",
    "RecalibrationTrigger",
    "RecalibrationReason",
    "MetaLearner",
]
