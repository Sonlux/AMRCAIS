"""
Machine Learning (Random Forest) Regime Classifier for AMRCAIS.

This module implements a supervised Random Forest classifier trained on
labeled historical regime data. It uses engineered features including
returns, volatility, correlations, and macro indicators.

Unlike the unsupervised HMM, this classifier requires labeled training data
but can capture more complex non-linear relationships between features and regimes.

Classes:
    MLRegimeClassifier: Random Forest classifier for regime detection

Example:
    >>> classifier = MLRegimeClassifier()
    >>> classifier.fit(features, labels)
    >>> result = classifier.predict(recent_features)
    >>> print(f"Regime: {result.regime}, Confidence: {result.confidence:.2f}")
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.regime_detection.base import BaseClassifier, RegimeResult

logger = logging.getLogger(__name__)


class MLRegimeClassifier(BaseClassifier):
    """Random Forest classifier for market regime detection.
    
    Uses supervised learning to classify regimes based on:
    - Multi-period returns (5d, 20d, 60d)
    - Rolling volatility
    - Cross-asset correlations
    - VIX level and percentile
    - Yield curve features
    
    Requires labeled training data with known regime classifications.
    
    Attributes:
        model: Trained RandomForestClassifier
        scaler: StandardScaler for feature normalization
        feature_names: List of expected feature names
        
    Example:
        >>> # Prepare labeled data
        >>> features = pipeline.prepare_regime_features(data)
        >>> labels = get_regime_labels(features.index)  # Expert labels
        >>> 
        >>> # Train classifier
        >>> ml = MLRegimeClassifier(n_estimators=200)
        >>> ml.fit(features, labels)
        >>> 
        >>> # Predict
        >>> result = ml.predict(recent_features)
    """
    
    # Expected features (from model_params.yaml)
    EXPECTED_FEATURES = [
        "SPX_returns_20d",
        "TLT_returns_20d",
        "VIX_level",
        "VIX_percentile",
        "equity_bond_corr_30d",
        "yield_curve_slope",
    ]
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
        config: Optional[Dict] = None,
    ):
        """Initialize the ML regime classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in leaf nodes
            random_state: Random seed for reproducibility
            config: Optional configuration dictionary
        """
        super().__init__(n_regimes=4, name="ML Classifier", config=config)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_names: List[str] = []
        self._cv_score: Optional[float] = None
    
    def _init_model(self) -> None:
        """Initialize the Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight="balanced",  # Handle class imbalance
            n_jobs=-1,  # Use all cores
        )
        self.scaler = StandardScaler()
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> "MLRegimeClassifier":
        """Train the Random Forest classifier.
        
        Args:
            data: Feature matrix (N observations × M features)
            labels: Regime labels (1-4) for each observation - REQUIRED
            
        Returns:
            self: The fitted classifier
            
        Raises:
            ValueError: If labels not provided or data invalid
        """
        if labels is None:
            raise ValueError(
                "MLRegimeClassifier requires labels for training. "
                "Consider using HMMRegimeClassifier for unsupervised learning."
            )
        
        logger.info(f"Training ML classifier on {len(data)} samples")
        
        # Store feature names
        if isinstance(data, pd.DataFrame):
            self._feature_names = list(data.columns)
        else:
            self._feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        # Validate data
        X = self._validate_data(data, min_samples=100)
        y = np.asarray(labels)
        
        # Validate labels
        if len(X) != len(y):
            raise ValueError(
                f"Data and labels length mismatch: {len(X)} vs {len(y)}"
            )
        
        unique_labels = np.unique(y)
        if not all(1 <= l <= 4 for l in unique_labels):
            raise ValueError(f"Labels must be 1-4, got {unique_labels}")
        
        # Initialize and fit
        self._init_model()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=5, scoring="accuracy"
        )
        self._cv_score = cv_scores.mean()
        
        logger.info(f"Cross-validation accuracy: {self._cv_score:.2%} (+/- {cv_scores.std():.2%})")
        
        # Fit on full data
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self._fit_timestamp = datetime.now()
        self._fit_samples = len(X)
        
        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {class_dist}")
        
        return self
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, pd.Series],
    ) -> RegimeResult:
        """Predict regime from features.
        
        Args:
            data: Feature vector or matrix (last row used for prediction)
            
        Returns:
            RegimeResult with regime, confidence, and probabilities
            
        Raises:
            ValueError: If classifier not fitted or data invalid
        """
        if not self.is_fitted:
            raise ValueError("ML classifier must be fitted before prediction")
        
        # Validate data
        X = self._validate_data(data, min_samples=1)
        
        # Use last row if multiple
        if len(X) > 1:
            X = X[-1:, :]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        regime = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Build probability dict (classes might not include all 1-4)
        prob_dict = {}
        for i, cls in enumerate(self.model.classes_):
            prob_dict[int(cls)] = float(proba[i])
        
        # Fill missing classes with 0
        for r in range(1, 5):
            if r not in prob_dict:
                prob_dict[r] = 0.0
        
        confidence = prob_dict[regime]
        
        result = RegimeResult(
            regime=int(regime),
            confidence=confidence,
            probabilities=prob_dict,
            features_used=self._feature_names,
            metadata={
                "cv_accuracy": self._cv_score,
            },
        )
        
        self._log_classification(result)
        return result
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Get probability distribution over regimes.
        
        Args:
            data: Feature matrix
            
        Returns:
            Array of shape (N, 4) with regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X = self._validate_data(data, min_samples=1)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)
        
        # Ensure we have 4 columns (one per regime)
        full_proba = np.zeros((len(proba), 4))
        for i, cls in enumerate(self.model.classes_):
            full_proba[:, int(cls) - 1] = proba[:, i]
        
        return full_proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from Random Forest.
        
        Returns:
            Dictionary mapping feature name to importance score
        """
        if not self.is_fitted:
            return None
        
        importance = dict(zip(
            self._feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance
    
    def get_classification_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
    ) -> str:
        """Generate classification report.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Classification report string
        """
        from sklearn.metrics import classification_report
        
        X_arr = self._validate_data(X, min_samples=1)
        X_scaled = self.scaler.transform(X_arr)
        y_pred = self.model.predict(X_scaled)
        
        target_names = [self.REGIME_NAMES[i] for i in sorted(np.unique(y))]
        
        return classification_report(y, y_pred, target_names=target_names)


def create_training_labels(
    dates: pd.DatetimeIndex,
    known_regimes: Dict[str, Tuple[str, str, int]],
) -> np.ndarray:
    """Create training labels from known historical regime periods.
    
    Args:
        dates: DatetimeIndex of observations
        known_regimes: Dict mapping period name to (start, end, regime_id)
            Example: {"covid_crash": ("2020-03-01", "2020-04-30", 2)}
            
    Returns:
        Array of regime labels (1-4), with 0 for unknown periods
        
    Example:
        >>> known = {
        ...     "covid_crash": ("2020-03-01", "2020-04-30", 2),
        ...     "post_covid_recovery": ("2020-05-01", "2020-12-31", 1),
        ...     "inflation_2022": ("2022-01-01", "2022-10-31", 3),
        ... }
        >>> labels = create_training_labels(features.index, known)
    """
    labels = np.zeros(len(dates), dtype=int)
    
    for period_name, (start, end, regime) in known_regimes.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        mask = (dates >= start_dt) & (dates <= end_dt)
        labels[mask] = regime
        
        count = mask.sum()
        logger.info(f"Labeled {count} days as Regime {regime} ({period_name})")
    
    # Log unlabeled
    unlabeled = (labels == 0).sum()
    if unlabeled > 0:
        logger.warning(f"{unlabeled} days remain unlabeled (will be excluded from training)")
    
    return labels


# Pre-defined known regime periods for training
KNOWN_REGIME_PERIODS = {
    # Risk-On Growth (Regime 1)
    "bull_2017_2019": ("2017-01-01", "2019-12-31", 1),
    "recovery_2021": ("2021-04-01", "2021-12-31", 1),
    "bull_2023_2024": ("2023-11-01", "2024-12-31", 1),
    
    # Risk-Off Crisis (Regime 2)
    "covid_crash": ("2020-02-20", "2020-03-23", 2),
    "aug_2015_china": ("2015-08-17", "2015-09-30", 2),
    "q4_2018_selloff": ("2018-10-01", "2018-12-24", 2),
    
    # Stagflation (Regime 3)
    "inflation_2022": ("2022-01-01", "2022-10-31", 3),
    "taper_tantrum_2013": ("2013-05-01", "2013-09-30", 3),
    
    # Disinflationary Boom (Regime 4)
    "qe_2012_2013": ("2012-01-01", "2012-12-31", 4),
    "post_covid_rally": ("2020-04-01", "2020-08-31", 4),
    "disinflation_2023": ("2023-06-01", "2023-10-31", 4),
}
