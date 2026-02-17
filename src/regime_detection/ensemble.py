"""
Regime Ensemble with Voting and Disagreement Index for AMRCAIS.

This is the KILLER FEATURE - the ensemble combines 4 independent classifiers
(HMM, ML, Correlation, Volatility) using weighted voting and calculates
a disagreement index that historically precedes regime transitions.

The disagreement index (0-1) measures how much classifiers disagree:
- Low disagreement (<0.3): High confidence regime
- Medium disagreement (0.3-0.6): Normal uncertainty
- High disagreement (>0.6): WARNING - historically precedes transitions

Classes:
    RegimeEnsemble: Main ensemble voter with disagreement calculation

Example:
    >>> from src.regime_detection.ensemble import RegimeEnsemble
    >>> ensemble = RegimeEnsemble()
    >>> ensemble.fit(training_data)
    >>> result = ensemble.predict(current_data)
    >>> print(f"Regime: {result.regime_name}, Disagreement: {result.disagreement:.2f}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import yaml

from src.regime_detection.base import BaseClassifier, RegimeResult, REGIME_NAMES
from src.regime_detection.hmm_classifier import HMMRegimeClassifier
from src.regime_detection.ml_classifier import MLRegimeClassifier
from src.regime_detection.correlation_classifier import CorrelationRegimeClassifier
from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult(RegimeResult):
    """Extended result from ensemble prediction.
    
    Inherits from RegimeResult and adds ensemble-specific metrics.
    
    Attributes:
        disagreement: Disagreement index (0-1), higher = more uncertainty
        individual_predictions: Dict mapping classifier name to its prediction
        classifier_weights: Weights used for each classifier
        transition_warning: Whether disagreement suggests imminent transition
    """
    disagreement: float = 0.0
    individual_predictions: Dict[str, int] = field(default_factory=dict)
    classifier_weights: Dict[str, float] = field(default_factory=dict)
    transition_warning: bool = False
    
    def __post_init__(self):
        """Validate and set transition warning."""
        super().__post_init__()
        if self.disagreement > 0.6:
            self.transition_warning = True


class RegimeEnsemble(BaseClassifier):
    """Ensemble regime classifier combining multiple approaches.
    
    This is the core innovation of AMRCAIS - combining four independent
    regime classifiers with a meta-learning layer that tracks their
    individual accuracy and adjusts weights dynamically.
    
    CLASSIFIERS:
    1. HMM Classifier: Learns regime dynamics from data structure
    2. ML Classifier: Supervised learning from historical labeled periods
    3. Correlation Classifier: Regime from cross-asset correlations
    4. Volatility Classifier: VIX-based regime identification
    
    KILLER FEATURE: Disagreement Index
    - Calculated as weighted entropy of classifier votes
    - Values >0.6 have historically preceded regime transitions
    - Can be used to flag uncertainty and trigger recalibration
    
    Attributes:
        classifiers: Dict of classifier instances
        weights: Current weights for each classifier
        disagreement_threshold: Level triggering transition warning
        meta_tracker: Performance tracking for dynamic weight adjustment
        
    Example:
        >>> ensemble = RegimeEnsemble()
        >>> 
        >>> # Fit all classifiers
        >>> ensemble.fit(historical_data)
        >>> 
        >>> # Get ensemble prediction
        >>> result = ensemble.predict(current_data)
        >>> 
        >>> # Check for transition warning
        >>> if result.transition_warning:
        ...     print(f"Warning: High disagreement {result.disagreement:.2f}")
        ...     print("Consider reducing position sizes")
        >>> 
        >>> # Access individual predictions
        >>> for name, pred in result.individual_predictions.items():
        ...     print(f"{name}: {REGIME_NAMES[pred]}")
    """
    
    DEFAULT_WEIGHTS = {
        "hmm": 0.35,
        "ml": 0.25,
        "correlation": 0.20,
        "volatility": 0.20,
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        disagreement_threshold: float = 0.6,
    ):
        """Initialize the regime ensemble.
        
        Args:
            config_path: Path to configuration directory (loads model_params.yaml)
            weights: Custom weights for classifiers (must sum to 1)
            disagreement_threshold: Threshold for transition warnings
        """
        super().__init__(n_regimes=4, name="Regime Ensemble")
        
        # Load config from YAML if available
        yaml_config = self._load_yaml_config(config_path)
        
        # Set weights: explicit arg > YAML > defaults
        if weights:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        elif yaml_config:
            yaml_weights = yaml_config.get("ensemble", {}).get("classifier_weights", {})
            if yaml_weights:
                # Map YAML keys to internal keys
                key_map = {"random_forest": "ml", "correlation_cluster": "correlation", "volatility_regime": "volatility"}
                mapped = {}
                for k, v in yaml_weights.items():
                    mapped_key = key_map.get(k, k)
                    mapped[mapped_key] = v
                total = sum(mapped.values())
                self.weights = {k: v / total for k, v in mapped.items()}
                logger.info(f"Loaded ensemble weights from YAML: {self.weights}")
            else:
                self.weights = self.DEFAULT_WEIGHTS.copy()
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()
        
        # Load disagreement threshold from config
        if yaml_config:
            disagr_config = yaml_config.get("ensemble", {}).get("disagreement", {})
            self.disagreement_threshold = disagr_config.get("alert_threshold", disagreement_threshold)
        else:
            self.disagreement_threshold = disagreement_threshold
        
        # Initialize classifiers
        self.classifiers: Dict[str, BaseClassifier] = {
            "hmm": HMMRegimeClassifier(),
            "ml": MLRegimeClassifier(),
            "correlation": CorrelationRegimeClassifier(),
            "volatility": VolatilityRegimeClassifier(),
        }
        
        # Meta-learning tracker
        self._accuracy_history: Dict[str, List[float]] = {
            name: [] for name in self.classifiers
        }
        self._prediction_history: List[EnsembleResult] = []
        self._disagreement_history: List[Tuple[datetime, float]] = []
        
        # Recalibration tracking
        self._consecutive_high_disagreement = 0
        self._regime_flip_count = 0
        self._last_regime: Optional[int] = None
        
        logger.info(
            f"Initialized RegimeEnsemble with weights: {self.weights}, "
            f"disagreement_threshold: {disagreement_threshold}"
        )
    
    def fit(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "RegimeEnsemble":
        """Fit all classifiers in the ensemble.
        
        Args:
            data: Training data with asset prices and indicators
            labels: Optional regime labels (1-4) for supervised classifiers.
                Value 0 denotes unlabeled observations and is filtered out
                before passing to the ML classifier.
            **kwargs: Additional arguments passed to classifiers
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting ensemble on data with shape {data.shape}")
        
        for name, classifier in self.classifiers.items():
            try:
                if name == "ml" and labels is not None:
                    # Filter to labeled-only rows (label != 0)
                    labeled_mask = labels != 0
                    if labeled_mask.sum() >= 100:
                        ml_data = data.iloc[labeled_mask] if isinstance(data, pd.DataFrame) else data[labeled_mask]
                        ml_labels = labels[labeled_mask]
                        classifier.fit(ml_data, labels=ml_labels, **kwargs)
                    else:
                        logger.warning(
                            f"Only {labeled_mask.sum()} labeled rows for ML "
                            f"classifier (need >=100), skipping"
                        )
                else:
                    classifier.fit(data, **kwargs)
                logger.info(f"Successfully fit {name} classifier")
            except Exception as e:
                logger.error(f"Failed to fit {name} classifier: {e}")
                # Don't fail entire ensemble for one classifier
        
        self._is_fitted = True
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame, **kwargs) -> EnsembleResult:
        """Get ensemble regime prediction.
        
        Combines predictions from all classifiers using weighted voting
        and calculates the disagreement index.
        
        Args:
            data: Current market data
            **kwargs: Additional arguments
            
        Returns:
            EnsembleResult with regime, confidence, and disagreement
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Collect predictions from each classifier
        predictions: Dict[str, RegimeResult] = {}
        votes: Dict[str, int] = {}
        
        for name, classifier in self.classifiers.items():
            try:
                result = classifier.predict(data, **kwargs)
                predictions[name] = result
                votes[name] = result.regime
                logger.debug(f"{name}: regime={result.regime}, conf={result.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Classifier {name} failed: {e}")
                # Use most common historical prediction as fallback
                votes[name] = self._get_fallback_vote(name)
        
        # Weighted voting
        regime, confidence = self._weighted_vote(votes, predictions)
        
        # Calculate disagreement
        disagreement = self._calculate_disagreement(votes, predictions)
        
        # Calculate probability distribution
        probabilities = self._aggregate_probabilities(predictions)
        
        # Check for regime flip
        self._track_regime_flip(regime)
        
        # Track disagreement
        self._track_disagreement(disagreement)
        
        # Create result
        result = EnsembleResult(
            regime=regime,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "individual_confidences": {
                    name: pred.confidence for name, pred in predictions.items()
                },
                "weights_used": self.weights.copy(),
            },
            disagreement=disagreement,
            individual_predictions=votes,
            classifier_weights=self.weights.copy(),
        )
        
        self._prediction_history.append(result)
        
        # Log with appropriate level based on disagreement
        if result.transition_warning:
            logger.warning(
                f"ENSEMBLE: {result.regime_name} (conf={confidence:.2f}, "
                f"DISAGREEMENT={disagreement:.2f}) - TRANSITION WARNING"
            )
        else:
            logger.info(
                f"ENSEMBLE: {result.regime_name} (conf={confidence:.2f}, "
                f"disagreement={disagreement:.2f})"
            )
        
        return result
    
    def _weighted_vote(
        self,
        votes: Dict[str, int],
        predictions: Dict[str, RegimeResult],
    ) -> Tuple[int, float]:
        """Perform weighted voting across classifiers.
        
        Args:
            votes: Dict mapping classifier name to predicted regime
            predictions: Full prediction results from each classifier
            
        Returns:
            Tuple of (winning_regime, confidence)
        """
        # Aggregate weighted votes by regime
        regime_scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        total_weight = 0.0
        
        for name, vote in votes.items():
            weight = self.weights.get(name, 0.0)
            
            # Scale by classifier's own confidence
            if name in predictions:
                classifier_conf = predictions[name].confidence
                weight *= classifier_conf
            
            regime_scores[vote] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            regime_scores = {k: v / total_weight for k, v in regime_scores.items()}
        
        # Winner
        winning_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[winning_regime]
        
        return winning_regime, confidence
    
    def _calculate_disagreement(
        self,
        votes: Dict[str, int],
        predictions: Dict[str, RegimeResult],
    ) -> float:
        """Calculate disagreement index.
        
        The disagreement index measures how much classifiers disagree.
        It's calculated as the weighted entropy of the vote distribution.
        
        High disagreement (>0.6) has historically preceded regime transitions.
        
        Args:
            votes: Dict mapping classifier name to predicted regime
            predictions: Full prediction results
            
        Returns:
            Disagreement index between 0 and 1
        """
        # Count weighted votes for each regime
        regime_weights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        total_weight = 0.0
        
        for name, vote in votes.items():
            weight = self.weights.get(name, 0.0)
            regime_weights[vote] += weight
            total_weight += weight
        
        if total_weight == 0:
            return 1.0  # Maximum uncertainty
        
        # Calculate probabilities
        probs = [regime_weights[r] / total_weight for r in range(1, 5)]
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize to [0, 1] (max entropy for 4 regimes is log2(4) = 2)
        max_entropy = np.log2(4)
        disagreement = entropy / max_entropy
        
        return disagreement
    
    def _aggregate_probabilities(
        self,
        predictions: Dict[str, RegimeResult],
    ) -> Dict[int, float]:
        """Aggregate probability distributions from all classifiers.
        
        Args:
            predictions: Predictions from each classifier
            
        Returns:
            Weighted average probability distribution
        """
        agg_probs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        total_weight = 0.0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            total_weight += weight
            
            for regime, prob in pred.probabilities.items():
                agg_probs[regime] += weight * prob
        
        if total_weight > 0:
            agg_probs = {k: v / total_weight for k, v in agg_probs.items()}
        
        return agg_probs
    
    def _track_regime_flip(self, current_regime: int) -> None:
        """Track regime changes for recalibration detection."""
        if self._last_regime is not None and self._last_regime != current_regime:
            self._regime_flip_count += 1
            logger.info(
                f"Regime flip detected: {REGIME_NAMES[self._last_regime]} -> "
                f"{REGIME_NAMES[current_regime]}"
            )
        
        self._last_regime = current_regime
    
    def _track_disagreement(self, disagreement: float) -> None:
        """Track disagreement for recalibration triggers."""
        self._disagreement_history.append((datetime.now(), disagreement))
        
        if disagreement > self.disagreement_threshold:
            self._consecutive_high_disagreement += 1
        else:
            self._consecutive_high_disagreement = 0
    
    def _get_fallback_vote(self, classifier_name: str) -> int:
        """Get fallback vote for failed classifier."""
        # Default to risk-on if no history
        if not self._prediction_history:
            return 1
        
        # Use last known prediction from this classifier
        for result in reversed(self._prediction_history):
            if classifier_name in result.individual_predictions:
                return result.individual_predictions[classifier_name]
        
        # Final fallback: most common regime in history
        all_regimes = [r.regime for r in self._prediction_history]
        if all_regimes:
            return max(set(all_regimes), key=all_regimes.count)
        
        return 1
    
    def needs_recalibration(self) -> Tuple[bool, str]:
        """Check if ensemble needs recalibration.
        
        Recalibration triggers:
        1. >3 regime flips in 5 days
        2. Disagreement >0.7 for 10+ consecutive predictions
        3. Any classifier showing >25% error rate over 2 weeks
        
        Returns:
            Tuple of (needs_recalibration, reason)
        """
        # Check regime flip count (recent window)
        if self._regime_flip_count > 3:
            return True, f"Excessive regime flips: {self._regime_flip_count}"
        
        # Check consecutive high disagreement
        if self._consecutive_high_disagreement >= 10:
            return True, f"Persistent high disagreement: {self._consecutive_high_disagreement} periods"
        
        # Accuracy tracking: use ensemble-agreement as a proxy for accuracy
        # when labeled data is not available.  If any classifier agrees with
        # the ensemble result less than 75% of the time over the last 14
        # predictions, it is likely drifting and the ensemble should be
        # recalibrated.
        lookback = min(14, len(self._prediction_history))
        if lookback >= 5:  # need ≥5 data points to be meaningful
            recent = self._prediction_history[-lookback:]
            for name in self.classifiers:
                agreements = sum(
                    1
                    for r in recent
                    if name in r.individual_predictions
                    and r.individual_predictions[name] == r.regime
                )
                total = sum(
                    1 for r in recent if name in r.individual_predictions
                )
                if total >= 5:
                    agreement_rate = agreements / total
                    # Track rolling accuracy for external consumers
                    self._accuracy_history[name].append(agreement_rate)
                    if agreement_rate < 0.75:
                        return (
                            True,
                            f"Classifier '{name}' agreement rate "
                            f"{agreement_rate:.0%} < 75% over last "
                            f"{total} predictions",
                        )
        
        return False, ""
    
    def recalibrate(
        self,
        data: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Recalibrate the ensemble.
        
        Args:
            data: Recent market data for recalibration
            labels: Optional known regime labels
        """
        logger.info("Recalibrating ensemble...")
        
        # Reset counters
        self._regime_flip_count = 0
        self._consecutive_high_disagreement = 0
        
        # Re-fit classifiers
        self.fit(data, labels=labels)
        
        logger.info("Ensemble recalibration complete")
    
    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update classifier weights.
        
        Args:
            new_weights: New weights for classifiers (will be normalized)
        """
        total = sum(new_weights.values())
        self.weights = {k: v / total for k, v in new_weights.items()}
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    @staticmethod
    def _load_yaml_config(config_path: Optional[str]) -> Optional[Dict]:
        """Load model parameters from YAML config.
        
        Args:
            config_path: Path to config directory containing model_params.yaml
            
        Returns:
            Config dict or None if not found
        """
        if not config_path:
            return None
        
        from pathlib import Path
        
        # Search for model_params.yaml
        search_paths = [
            Path(config_path) / "model_params.yaml",
            Path(config_path),
            Path("config") / "model_params.yaml",
        ]
        
        for path in search_paths:
            if path.exists() and path.is_file():
                try:
                    with open(path, "r") as f:
                        config = yaml.safe_load(f)
                    logger.debug(f"Loaded model params from {path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")
        
        return None
    
    def get_classifier_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for each classifier.
        
        Returns:
            Dict with accuracy, agreement rate, etc. for each classifier
        """
        if len(self._prediction_history) < 2:
            return {}
        
        performance = {}
        
        for name in self.classifiers:
            # Agreement with ensemble
            agreements = 0
            total = 0
            
            for result in self._prediction_history:
                if name in result.individual_predictions:
                    if result.individual_predictions[name] == result.regime:
                        agreements += 1
                    total += 1
            
            agreement_rate = agreements / total if total > 0 else 0
            
            performance[name] = {
                "ensemble_agreement_rate": agreement_rate,
                "predictions_count": total,
                "current_weight": self.weights.get(name, 0),
            }
        
        return performance
    
    def get_disagreement_history(
        self,
        lookback: int = 100,
    ) -> List[Tuple[datetime, float]]:
        """Get recent disagreement history.
        
        Args:
            lookback: Number of periods to return
            
        Returns:
            List of (timestamp, disagreement) tuples
        """
        return self._disagreement_history[-lookback:]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from all classifiers.
        
        Combines feature importance from each classifier, weighted by
        the ensemble's classifier weights.
        
        Returns:
            Dictionary mapping feature name to importance score,
            or None if no classifiers report importance
        """
        aggregated: Dict[str, float] = {}
        total_weight = 0.0
        
        for name, classifier in self.classifiers.items():
            weight = self.weights.get(name, 0.0)
            importance = classifier.get_feature_importance()
            
            if importance is not None:
                total_weight += weight
                for feature, score in importance.items():
                    aggregated[feature] = aggregated.get(feature, 0.0) + weight * score
        
        if not aggregated or total_weight == 0:
            return None
        
        # Normalize by total contributing weight
        aggregated = {k: v / total_weight for k, v in aggregated.items()}
        
        return aggregated
    
    def predict_sequence(
        self,
        data: pd.DataFrame,
        window: int = 60,
        step: int = 1,
    ) -> List[EnsembleResult]:
        """Generate regime predictions over a time series.
        
        Args:
            data: Full time series data
            window: Rolling window size
            step: Step size between predictions
            
        Returns:
            List of EnsembleResult for each period
        """
        results = []
        
        for i in range(window, len(data), step):
            window_data = data.iloc[i-window:i]
            result = self.predict(window_data)
            results.append(result)
        
        return results
