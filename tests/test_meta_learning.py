"""
Test Suite for AMRCAIS Meta-Learning Layer (Layer 3).

Tests for:
- RegimePerformanceTracker: Classification history & metrics
- RecalibrationTrigger: Recalibration decision logic
- MetaLearner: Adaptive intelligence coordinator

Coverage target: 80%+ for src/meta_learning/*
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for test storage."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir) / "test_regime_history.csv"
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_classifications():
    """Generate sample classification history."""
    classifications = []
    base_time = datetime(2026, 1, 1)
    
    # 30 days of mostly risk-on with a few transitions
    regimes = [1]*10 + [2]*5 + [1]*10 + [3]*3 + [1]*2
    for i, regime in enumerate(regimes):
        classifications.append({
            "regime": regime,
            "confidence": 0.8 if regime == 1 else 0.65,
            "disagreement": 0.2 if regime == 1 else 0.5,
            "individual_predictions": {"hmm": regime, "ml": regime, "correlation": 1, "volatility": regime},
            "timestamp": base_time + timedelta(days=i),
            "market_state": {"SPX_returns": 0.01 if regime == 1 else -0.02},
        })
    return classifications


@pytest.fixture
def high_disagreement_classifications():
    """Generate classifications with persistent high disagreement."""
    classifications = []
    base_time = datetime(2026, 1, 1)
    
    for i in range(15):
        classifications.append({
            "regime": 1 if i % 2 == 0 else 2,
            "confidence": 0.4,
            "disagreement": 0.75,
            "individual_predictions": {"hmm": 1, "ml": 2, "correlation": 3, "volatility": 1},
            "timestamp": base_time + timedelta(days=i),
        })
    return classifications


# ============================================================================
# PERFORMANCE TRACKER TESTS
# ============================================================================

class TestRegimePerformanceTracker:
    """Tests for RegimePerformanceTracker."""

    def test_initialization(self, temp_storage_path):
        """Test tracker initialization."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        assert len(tracker.history) == 0
        assert tracker.storage_path == temp_storage_path

    def test_log_classification(self, temp_storage_path):
        """Test logging a single classification."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        tracker.log_classification(
            regime=1,
            confidence=0.85,
            disagreement=0.2,
            individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 2},
            market_state={"SPX_returns": 0.01},
        )
        
        assert len(tracker.history) == 1
        assert tracker.history[0].regime == 1
        assert tracker.history[0].confidence == 0.85
        assert tracker.history[0].disagreement == 0.2

    def test_log_multiple_classifications(self, temp_storage_path, sample_classifications):
        """Test logging multiple classifications and tracking transitions."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        for c in sample_classifications:
            tracker.log_classification(**c)
        
        assert len(tracker.history) == len(sample_classifications)

    def test_performance_metrics_empty(self, temp_storage_path):
        """Test metrics calculation with no history."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        metrics = tracker.get_performance_metrics(lookback_days=30)
        
        assert metrics.total_classifications == 0
        assert metrics.regime_stability_score == 0.0
        assert metrics.transition_count == 0

    def test_performance_metrics_with_data(self, temp_storage_path, sample_classifications):
        """Test metrics calculation with classification history."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        for c in sample_classifications:
            tracker.log_classification(**c)
        
        metrics = tracker.get_performance_metrics(lookback_days=60)
        
        assert metrics.total_classifications == len(sample_classifications)
        assert 0 <= metrics.regime_stability_score <= 1
        assert metrics.transition_count >= 0
        assert isinstance(metrics.regime_distribution, dict)

    def test_stability_rating(self, temp_storage_path):
        """Test stability rating calculation for different scores."""
        from src.meta_learning.performance_tracker import PerformanceMetrics
        
        now = datetime.now()
        
        # High stability
        m1 = PerformanceMetrics(
            period_start=now - timedelta(days=30), period_end=now,
            total_classifications=30, regime_stability_score=0.9,
            transition_count=1, avg_disagreement=0.15, high_disagreement_days=0,
        )
        assert m1.stability_rating == "High Stability"
        
        # Moderate stability
        m2 = PerformanceMetrics(
            period_start=now - timedelta(days=30), period_end=now,
            total_classifications=30, regime_stability_score=0.65,
            transition_count=5, avg_disagreement=0.3, high_disagreement_days=2,
        )
        assert m2.stability_rating == "Moderate Stability"
        
        # Very low stability
        m3 = PerformanceMetrics(
            period_start=now - timedelta(days=30), period_end=now,
            total_classifications=30, regime_stability_score=0.2,
            transition_count=15, avg_disagreement=0.7, high_disagreement_days=10,
        )
        assert "Very Low" in m3.stability_rating

    def test_metrics_to_dict(self, temp_storage_path):
        """Test metrics serialization."""
        from src.meta_learning.performance_tracker import PerformanceMetrics
        
        now = datetime.now()
        metrics = PerformanceMetrics(
            period_start=now - timedelta(days=30), period_end=now,
            total_classifications=30, regime_stability_score=0.85,
            transition_count=2, avg_disagreement=0.2, high_disagreement_days=1,
            regime_distribution={1: 25, 2: 5},
        )
        d = metrics.to_dict()
        
        assert "stability_rating" in d
        assert "regime_stability_score" in d
        assert d["total_classifications"] == 30
        assert d["regime_distribution"] == {1: 25, 2: 5}

    def test_check_regime_flip_frequency(self, temp_storage_path):
        """Test regime flip detection."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        # Log alternating regimes (excessive flipping)
        now = datetime.now()
        for i in range(10):
            tracker.log_classification(
                regime=1 if i % 2 == 0 else 2,
                confidence=0.6,
                disagreement=0.5,
                individual_predictions={"hmm": 1, "ml": 2, "correlation": 1, "volatility": 2},
                timestamp=now - timedelta(days=4) + timedelta(hours=i * 8),
            )
        
        flip_count, is_excessive = tracker.check_regime_flip_frequency(window_days=5)
        assert flip_count > 3
        assert is_excessive is True

    def test_check_no_flips(self, temp_storage_path):
        """Test flip check with stable regime."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(10):
            tracker.log_classification(
                regime=1,
                confidence=0.9,
                disagreement=0.1,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                timestamp=now - timedelta(days=4) + timedelta(hours=i * 8),
            )
        
        flip_count, is_excessive = tracker.check_regime_flip_frequency(window_days=5)
        assert flip_count == 0
        assert is_excessive is False

    def test_check_persistent_disagreement(self, temp_storage_path):
        """Test persistent disagreement detection."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(15):
            tracker.log_classification(
                regime=1,
                confidence=0.5,
                disagreement=0.75,  # Above default 0.7 threshold
                individual_predictions={"hmm": 1, "ml": 2, "correlation": 3, "volatility": 1},
                timestamp=now - timedelta(days=14) + timedelta(days=i),
            )
        
        days, is_persistent = tracker.check_persistent_disagreement(threshold=0.7, min_days=10)
        assert days >= 10
        assert is_persistent is True

    def test_no_persistent_disagreement(self, temp_storage_path):
        """Test no persistent disagreement with low values."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(10):
            tracker.log_classification(
                regime=1,
                confidence=0.9,
                disagreement=0.2,  # Very low
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                timestamp=now - timedelta(days=9) + timedelta(days=i),
            )
        
        days, is_persistent = tracker.check_persistent_disagreement()
        assert is_persistent is False

    def test_evaluate_prediction_accuracy(self, temp_storage_path):
        """Test prediction accuracy evaluation against market behavior."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        now = datetime.now()
        # Log risk-on predictions with positive SPX returns (should be consistent)
        for i in range(10):
            tracker.log_classification(
                regime=1,  # Risk-On Growth
                confidence=0.85,
                disagreement=0.2,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                market_state={"SPX_returns": 0.01},  # Positive = consistent with risk-on
                timestamp=now - timedelta(days=10) + timedelta(days=i),
            )
        
        accuracy = tracker.evaluate_prediction_accuracy(lookback_days=15)
        assert accuracy > 0.5  # Should be good since regime matches market

    def test_classification_to_dict(self, temp_storage_path):
        """Test RegimeClassification serialization."""
        from src.meta_learning.performance_tracker import RegimeClassification
        
        c = RegimeClassification(
            timestamp=datetime(2026, 1, 15),
            regime=1,
            confidence=0.85,
            disagreement=0.2,
            individual_predictions={"hmm": 1, "ml": 1},
            market_state={"SPX_returns": 0.01},
        )
        d = c.to_dict()
        assert d["regime"] == 1
        assert d["confidence"] == 0.85
        assert "timestamp" in d

    def test_get_classification_history(self, temp_storage_path, sample_classifications):
        """Test getting history as DataFrame."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        for c in sample_classifications:
            tracker.log_classification(**c)
        
        df = tracker.get_classification_history()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_classifications)
        assert "regime" in df.columns
        assert "confidence" in df.columns
        assert "disagreement" in df.columns

    def test_clear_history(self, temp_storage_path):
        """Test clearing classification history."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        tracker.log_classification(
            regime=1, confidence=0.8, disagreement=0.2,
            individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
        )
        assert len(tracker.history) == 1
        
        tracker.clear_history()
        assert len(tracker.history) == 0

    def test_empty_history_edge_cases(self, temp_storage_path):
        """Test edge cases with empty or minimal history."""
        from src.meta_learning.performance_tracker import RegimePerformanceTracker
        
        tracker = RegimePerformanceTracker(storage_path=temp_storage_path)
        
        # All operations should handle empty history gracefully
        flip_count, excessive = tracker.check_regime_flip_frequency()
        assert flip_count == 0
        assert excessive is False
        
        days, persistent = tracker.check_persistent_disagreement()
        assert days == 0
        assert persistent is False
        
        accuracy = tracker.evaluate_prediction_accuracy()
        assert accuracy == 0.5  # Default with no data
        
        df = tracker.get_classification_history()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ============================================================================
# RECALIBRATION TRIGGER TESTS
# ============================================================================

class TestRecalibrationTrigger:
    """Tests for RecalibrationTrigger decision logic."""

    def test_initialization(self):
        """Test trigger initialization with defaults."""
        from src.meta_learning.recalibration import RecalibrationTrigger
        
        trigger = RecalibrationTrigger()
        assert trigger.error_rate_threshold == 0.75
        assert trigger.flip_threshold == 3
        assert trigger.disagreement_persistence_days == 10

    def test_no_recalibration_needed(self):
        """Test that healthy system doesn't trigger recalibration."""
        from src.meta_learning.recalibration import RecalibrationTrigger
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.85,
            recent_flips=1,
            disagreement_days=2,
            avg_confidence=0.8,
        )
        
        assert decision.should_recalibrate is False
        assert len(decision.reasons) == 0
        assert decision.severity == 0.0
        assert "NONE" in decision.urgency_level

    def test_high_error_rate_triggers(self):
        """Test recalibration triggers on low accuracy."""
        from src.meta_learning.recalibration import RecalibrationTrigger, RecalibrationReason
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.50,
            recent_flips=0,
            disagreement_days=0,
            avg_confidence=0.8,
        )
        
        assert decision.should_recalibrate is True
        assert RecalibrationReason.HIGH_ERROR_RATE in decision.reasons
        assert decision.severity > 0

    def test_excessive_flipping_triggers(self):
        """Test recalibration triggers on too many regime flips."""
        from src.meta_learning.recalibration import RecalibrationTrigger, RecalibrationReason
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.85,
            recent_flips=7,
            disagreement_days=0,
            avg_confidence=0.8,
        )
        
        assert decision.should_recalibrate is True
        assert RecalibrationReason.EXCESSIVE_FLIPPING in decision.reasons

    def test_persistent_disagreement_triggers(self):
        """Test recalibration triggers on persistent high disagreement."""
        from src.meta_learning.recalibration import RecalibrationTrigger, RecalibrationReason
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.80,
            recent_flips=1,
            disagreement_days=12,
            avg_confidence=0.7,
        )
        
        assert decision.should_recalibrate is True
        assert RecalibrationReason.PERSISTENT_DISAGREEMENT in decision.reasons

    def test_low_confidence_triggers(self):
        """Test recalibration triggers on low average confidence."""
        from src.meta_learning.recalibration import RecalibrationTrigger, RecalibrationReason
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.80,
            recent_flips=1,
            disagreement_days=2,
            avg_confidence=0.3,
        )
        
        assert decision.should_recalibrate is True
        assert RecalibrationReason.LOW_CONFIDENCE in decision.reasons

    def test_multiple_triggers_simultaneously(self):
        """Test multiple recalibration reasons at once."""
        from src.meta_learning.recalibration import RecalibrationTrigger
        
        trigger = RecalibrationTrigger()
        decision = trigger.evaluate(
            accuracy=0.50,
            recent_flips=8,
            disagreement_days=15,
            avg_confidence=0.3,
            avg_disagreement=0.85,
        )
        
        assert decision.should_recalibrate is True
        assert len(decision.reasons) >= 3
        assert decision.severity >= 0.5

    def test_urgency_levels(self):
        """Test different urgency levels based on severity."""
        from src.meta_learning.recalibration import RecalibrationDecision, RecalibrationReason
        
        now = datetime.now()
        
        # Critical
        d1 = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.HIGH_ERROR_RATE],
            severity=0.9, timestamp=now, recommendations=[],
        )
        assert "CRITICAL" in d1.urgency_level
        
        # High
        d2 = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.EXCESSIVE_FLIPPING],
            severity=0.65, timestamp=now, recommendations=[],
        )
        assert "HIGH" in d2.urgency_level
        
        # Medium
        d3 = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.LOW_CONFIDENCE],
            severity=0.45, timestamp=now, recommendations=[],
        )
        assert "MEDIUM" in d3.urgency_level
        
        # Low
        d4 = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[],
            severity=0.25, timestamp=now, recommendations=[],
        )
        assert "LOW" in d4.urgency_level

    def test_decision_to_dict(self):
        """Test RecalibrationDecision serialization."""
        from src.meta_learning.recalibration import RecalibrationDecision, RecalibrationReason
        
        decision = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.HIGH_ERROR_RATE, RecalibrationReason.EXCESSIVE_FLIPPING],
            severity=0.7,
            timestamp=datetime(2026, 1, 15),
            recommendations=["Retrain classifiers"],
        )
        d = decision.to_dict()
        
        assert d["should_recalibrate"] is True
        assert len(d["reasons"]) == 2
        assert d["severity"] == 0.7
        assert "urgency_level" in d

    def test_check_individual_trigger(self):
        """Test individual trigger checking."""
        from src.meta_learning.recalibration import RecalibrationTrigger
        
        trigger = RecalibrationTrigger()
        
        # Accuracy trigger
        triggered, severity = trigger.check_individual_trigger("accuracy", 0.5)
        assert triggered is True
        assert severity == 0.5
        
        not_triggered, sev = trigger.check_individual_trigger("accuracy", 0.9)
        assert not_triggered is False
        assert sev == 0.0
        
        # Flips trigger
        triggered, severity = trigger.check_individual_trigger("flips", 6)
        assert triggered is True
        
        # Disagreement trigger
        triggered, severity = trigger.check_individual_trigger("disagreement_days", 15)
        assert triggered is True
        
        # Confidence trigger
        triggered, severity = trigger.check_individual_trigger("confidence", 0.3)
        assert triggered is True
        
        # Unknown trigger
        triggered, severity = trigger.check_individual_trigger("unknown", 0.5)
        assert triggered is False

    def test_adjust_thresholds(self):
        """Test dynamic threshold adjustment."""
        from src.meta_learning.recalibration import RecalibrationTrigger
        
        trigger = RecalibrationTrigger()
        original_threshold = trigger.flip_threshold
        
        trigger.adjust_thresholds({"flip_threshold": 5})
        assert trigger.flip_threshold == 5
        assert trigger.flip_threshold != original_threshold

    def test_recalibration_reason_str(self):
        """Test RecalibrationReason string representation."""
        from src.meta_learning.recalibration import RecalibrationReason
        
        for reason in RecalibrationReason:
            desc = str(reason)
            assert len(desc) > 5  # Should have meaningful description


# ============================================================================
# META-LEARNER TESTS
# ============================================================================

class TestMetaLearner:
    """Tests for MetaLearner coordinator."""

    def test_initialization(self, temp_storage_path):
        """Test MetaLearner initialization."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        assert meta.enable_adaptive_weights is True
        assert meta._recalibration_count == 0
        assert meta._last_recalibration is None
        assert sum(meta.adaptive_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_log_classification(self, temp_storage_path):
        """Test logging a classification through MetaLearner."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        meta.log_classification(
            regime=1,
            confidence=0.85,
            disagreement=0.2,
            individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 2},
            market_state={"SPX_returns": 0.01},
        )
        
        assert len(meta.tracker.history) == 1

    def test_high_disagreement_logging(self, temp_storage_path):
        """Test that high disagreement triggers warning logging."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        # This should trigger the high disagreement warning
        meta.log_classification(
            regime=1,
            confidence=0.4,
            disagreement=0.75,  # >0.6 threshold
            individual_predictions={"hmm": 1, "ml": 2, "correlation": 3, "volatility": 1},
        )
        
        assert len(meta.tracker.history) == 1
        assert meta.tracker.history[0].disagreement == 0.75

    def test_check_recalibration_no_data(self, temp_storage_path):
        """Test recalibration check with no classification data."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        decision = meta.check_recalibration_needed()
        
        # With no data, should not trigger recalibration (no evidence of problems)
        assert isinstance(decision.should_recalibrate, bool)

    def test_check_recalibration_with_issues(self, temp_storage_path, high_disagreement_classifications):
        """Test recalibration detection with problematic history."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        for c in high_disagreement_classifications:
            meta.log_classification(**c)
        
        decision = meta.check_recalibration_needed()
        # With alternating regimes and high disagreement, should detect issues
        assert isinstance(decision.should_recalibrate, bool)
        assert isinstance(decision.severity, float)

    def test_execute_recalibration_not_needed(self, temp_storage_path):
        """Test execution when recalibration not needed."""
        from src.meta_learning.meta_learner import MetaLearner
        from src.meta_learning.recalibration import RecalibrationDecision
        
        meta = MetaLearner(storage_path=temp_storage_path)
        decision = RecalibrationDecision(
            should_recalibrate=False,
            reasons=[], severity=0.0,
            timestamp=datetime.now(), recommendations=[],
        )
        
        result = meta.execute_recalibration(decision)
        assert result is True
        assert meta._recalibration_count == 0

    def test_execute_recalibration_needed(self, temp_storage_path):
        """Test execution when recalibration is needed."""
        from src.meta_learning.meta_learner import MetaLearner
        from src.meta_learning.recalibration import RecalibrationDecision, RecalibrationReason
        
        meta = MetaLearner(storage_path=temp_storage_path)
        decision = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.HIGH_ERROR_RATE, RecalibrationReason.EXCESSIVE_FLIPPING],
            severity=0.7,
            timestamp=datetime.now(),
            recommendations=["Retrain classifiers"],
        )
        
        result = meta.execute_recalibration(decision)
        assert result is True
        assert meta._recalibration_count == 1
        assert meta._last_recalibration is not None

    def test_execute_recalibration_persistent_disagreement(self, temp_storage_path):
        """Test execution with persistent disagreement reason."""
        from src.meta_learning.meta_learner import MetaLearner
        from src.meta_learning.recalibration import RecalibrationDecision, RecalibrationReason
        
        meta = MetaLearner(storage_path=temp_storage_path)
        decision = RecalibrationDecision(
            should_recalibrate=True,
            reasons=[RecalibrationReason.PERSISTENT_DISAGREEMENT],
            severity=0.5,
            timestamp=datetime.now(),
            recommendations=["Monitor regime"],
        )
        
        result = meta.execute_recalibration(decision)
        assert result is True
        assert meta._recalibration_count == 1

    def test_get_performance_metrics(self, temp_storage_path):
        """Test performance metrics retrieval through MetaLearner."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        # Log some data
        now = datetime.now()
        for i in range(5):
            meta.log_classification(
                regime=1, confidence=0.8, disagreement=0.15,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                timestamp=now - timedelta(days=5) + timedelta(days=i),
            )
        
        metrics = meta.get_performance_metrics(lookback_days=10)
        assert metrics.total_classifications == 5
        assert metrics.regime_stability_score == 1.0  # No transitions

    def test_get_adaptive_weights(self, temp_storage_path):
        """Test adaptive weight retrieval."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path, enable_adaptive_weights=True)
        weights = meta.get_adaptive_weights()
        
        assert isinstance(weights, dict)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        assert "hmm" in weights
        assert "ml" in weights
        assert "correlation" in weights
        assert "volatility" in weights

    def test_adaptive_weights_disabled(self, temp_storage_path):
        """Test adaptive weights when disabled."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path, enable_adaptive_weights=False)
        weights = meta.get_adaptive_weights()
        
        # Should return default weights
        assert weights == meta.DEFAULT_WEIGHTS

    def test_adaptive_weight_update(self, temp_storage_path):
        """Test that adaptive weights change with sufficient history."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path, enable_adaptive_weights=True)
        initial_weights = meta.get_adaptive_weights().copy()
        
        # Log 10 classifications to trigger weight update (triggers at % 10 == 0)
        now = datetime.now()
        for i in range(10):
            meta.log_classification(
                regime=1, confidence=0.8, disagreement=0.2,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 2},
                timestamp=now - timedelta(hours=10-i),
            )
        
        updated_weights = meta.get_adaptive_weights()
        assert sum(updated_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_generate_uncertainty_signal_no_data(self, temp_storage_path):
        """Test uncertainty signal with no history."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        signal = meta.generate_uncertainty_signal()
        
        assert signal["signal"] == "neutral"
        assert signal["strength"] == 0.0

    def test_generate_uncertainty_signal_low(self, temp_storage_path):
        """Test uncertainty signal with low disagreement."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(10):
            meta.log_classification(
                regime=1, confidence=0.9, disagreement=0.15,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                timestamp=now - timedelta(hours=10-i),
            )
        
        signal = meta.generate_uncertainty_signal()
        assert signal["signal"] == "low_uncertainty"
        assert signal["actionable"] is False

    def test_generate_uncertainty_signal_high(self, temp_storage_path):
        """Test uncertainty signal with high disagreement."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(10):
            meta.log_classification(
                regime=1, confidence=0.4, disagreement=0.75,
                individual_predictions={"hmm": 1, "ml": 2, "correlation": 3, "volatility": 1},
                timestamp=now - timedelta(hours=10-i),
            )
        
        signal = meta.generate_uncertainty_signal()
        assert signal["signal"] == "high_uncertainty"
        assert signal["actionable"] is True
        assert signal["current_disagreement"] > 0.7

    def test_generate_uncertainty_signal_elevated(self, temp_storage_path):
        """Test uncertainty signal with elevated disagreement (0.6-0.7)."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(10):
            meta.log_classification(
                regime=1, confidence=0.5, disagreement=0.65,
                individual_predictions={"hmm": 1, "ml": 2, "correlation": 1, "volatility": 1},
                timestamp=now - timedelta(hours=10-i),
            )
        
        signal = meta.generate_uncertainty_signal()
        assert signal["signal"] == "elevated_uncertainty"
        assert signal["actionable"] is True

    def test_system_health_report(self, temp_storage_path):
        """Test comprehensive system health report generation."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        now = datetime.now()
        for i in range(15):
            meta.log_classification(
                regime=1, confidence=0.85, disagreement=0.2,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                market_state={"SPX_returns": 0.01},
                timestamp=now - timedelta(days=15) + timedelta(days=i),
            )
        
        report = meta.get_system_health_report()
        
        assert "timestamp" in report
        assert "system_status" in report
        assert "performance_30d" in report
        assert "performance_7d" in report
        assert "recalibration" in report
        assert "alerts" in report
        assert "accuracy" in report
        assert "adaptive_weights" in report
        assert report["system_status"] in ["healthy", "needs_attention"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMetaLearningIntegration:
    """Integration tests for the meta-learning pipeline."""

    def test_full_workflow(self, temp_storage_path):
        """Test complete meta-learning workflow."""
        from src.meta_learning.meta_learner import MetaLearner
        
        meta = MetaLearner(storage_path=temp_storage_path)
        
        now = datetime.now()
        
        # Phase 1: Stable risk-on period
        for i in range(20):
            meta.log_classification(
                regime=1, confidence=0.85, disagreement=0.15,
                individual_predictions={"hmm": 1, "ml": 1, "correlation": 1, "volatility": 1},
                market_state={"SPX_returns": 0.005},
                timestamp=now - timedelta(days=30) + timedelta(days=i),
            )
        
        # Phase 2: Rising disagreement (transition forming)
        for i in range(5):
            meta.log_classification(
                regime=1, confidence=0.5, disagreement=0.4 + i * 0.08,
                individual_predictions={"hmm": 1, "ml": 2, "correlation": 1, "volatility": 2},
                market_state={"SPX_returns": -0.003},
                timestamp=now - timedelta(days=10) + timedelta(days=i),
            )
        
        # Phase 3: Crisis detected with high disagreement
        for i in range(5):
            meta.log_classification(
                regime=2, confidence=0.6, disagreement=0.55,
                individual_predictions={"hmm": 2, "ml": 2, "correlation": 1, "volatility": 2},
                market_state={"SPX_returns": -0.03, "VIX_level": 35},
                timestamp=now - timedelta(days=5) + timedelta(days=i),
            )
        
        # Verify metrics
        metrics = meta.get_performance_metrics(lookback_days=35)
        assert metrics.total_classifications == 30
        assert metrics.transition_count >= 1  # At least 1 transition
        
        # Verify uncertainty signal
        signal = meta.generate_uncertainty_signal()
        assert isinstance(signal["signal"], str)
        assert 0 <= signal["strength"] <= 1
        
        # Verify health report
        report = meta.get_system_health_report()
        assert report["performance_30d"]["transition_count"] >= 1
        
        # Verify adaptive weights
        weights = meta.get_adaptive_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
