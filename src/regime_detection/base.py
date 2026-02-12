"""
Base Classifier for AMRCAIS Regime Detection.

This module defines the abstract interface that all regime classifiers
must implement, ensuring consistent behavior across HMM, ML, correlation,
and volatility-based classifiers.

Classes:
    BaseClassifier: Abstract base class defining the classifier interface
    RegimeResult: Dataclass containing classification results
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Regime definitions - exported for use across the system
REGIME_NAMES = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}


@dataclass
class RegimeResult:
    """Result of a regime classification.
    
    Attributes:
        regime: The classified regime ID (1-4)
        confidence: Confidence score for the classification (0-1)
        probabilities: Probability distribution across all regimes
        timestamp: Timestamp of the classification
        features_used: List of features used for classification
        metadata: Additional classifier-specific metadata
    """
    regime: int
    confidence: float
    probabilities: Dict[int, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    features_used: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate classification results."""
        if not 1 <= self.regime <= 4:
            raise ValueError(f"Regime must be 1-4, got {self.regime}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def regime_name(self) -> str:
        """Get human-readable regime name."""
        names = {
            1: "Risk-On Growth",
            2: "Risk-Off Crisis",
            3: "Stagflation",
            4: "Disinflationary Boom",
        }
        return names.get(self.regime, f"Unknown Regime {self.regime}")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "regime": self.regime,
            "regime_name": self.regime_name,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "features_used": self.features_used,
            "metadata": self.metadata,
        }


class BaseClassifier(ABC):
    """Abstract base class for all regime classifiers.
    
    Defines the interface that HMM, ML, correlation, and volatility
    classifiers must implement. Provides common functionality for
    logging, configuration, and result handling.
    
    Attributes:
        n_regimes: Number of regimes to classify (default 4)
        name: Human-readable classifier name
        is_fitted: Whether the classifier has been trained
        
    Example:
        >>> class MyClassifier(BaseClassifier):
        ...     def fit(self, data):
        ...         # Training logic
        ...         pass
        ...     def predict(self, data):
        ...         # Prediction logic
        ...         return RegimeResult(regime=1, confidence=0.8)
    """
    
    # Use module-level REGIME_NAMES constant
    REGIME_NAMES = REGIME_NAMES
    
    def __init__(
        self,
        n_regimes: int = 4,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize the classifier.
        
        Args:
            n_regimes: Number of regimes to detect (must be 4 for AMRCAIS)
            name: Classifier name for logging
            config: Optional configuration dictionary
        """
        if n_regimes != 4:
            logger.warning(
                f"AMRCAIS is designed for 4 regimes, got {n_regimes}. "
                "Results may not align with regime definitions."
            )
        
        self.n_regimes = n_regimes
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.is_fitted = False
        
        # Training history
        self._fit_timestamp: Optional[datetime] = None
        self._fit_samples: int = 0
        
        logger.debug(f"Initialized {self.name} with {n_regimes} regimes")
    
    @abstractmethod
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> "BaseClassifier":
        """Train the classifier on historical data.
        
        Args:
            data: Feature matrix (N observations × M features)
            labels: Optional ground truth labels for supervised classifiers
            
        Returns:
            self: The fitted classifier instance
            
        Raises:
            ValueError: If data is invalid or insufficient
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, pd.Series],
    ) -> RegimeResult:
        """Predict the current regime from recent data.
        
        Args:
            data: Recent data for classification (single observation or window)
            
        Returns:
            RegimeResult with regime ID, confidence, and probabilities
            
        Raises:
            ValueError: If classifier is not fitted or data is invalid
        """
        pass
    
    def predict_sequence(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> List[RegimeResult]:
        """Predict regime for a sequence of observations.
        
        Args:
            data: Time series data (N observations × M features)
            
        Returns:
            List of RegimeResult for each observation
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        
        results = []
        
        if isinstance(data, pd.DataFrame):
            for idx in data.index:
                row = data.loc[idx:idx]
                result = self.predict(row)
                result.timestamp = idx if isinstance(idx, datetime) else None
                results.append(result)
        else:
            for i in range(len(data)):
                result = self.predict(data[i:i+1])
                results.append(result)
        
        return results
    
    def get_regime_history(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Get regime classifications as a DataFrame.
        
        Args:
            data: Time series data for classification
            
        Returns:
            DataFrame with Date, Regime, Confidence, and probability columns
        """
        results = self.predict_sequence(data)
        
        records = []
        for i, r in enumerate(results):
            record = {
                "Regime": r.regime,
                "Regime_Name": r.regime_name,
                "Confidence": r.confidence,
            }
            for regime, prob in r.probabilities.items():
                record[f"Prob_Regime_{regime}"] = prob
            
            if isinstance(data, pd.DataFrame):
                record["Date"] = data.index[i]
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)
        
        return df
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available.
        
        Returns:
            Dictionary mapping feature name to importance score,
            or None if not supported by this classifier
        """
        pass
    
    def _validate_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        min_samples: int = 100,
    ) -> np.ndarray:
        """Validate and convert input data.
        
        Args:
            data: Input data (DataFrame or array)
            min_samples: Minimum required samples
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If data is invalid
        """
        if isinstance(data, pd.DataFrame):
            arr = data.values
        elif isinstance(data, pd.Series):
            arr = data.values.reshape(-1, 1)
        else:
            arr = np.asarray(data)
        
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        if len(arr) < min_samples:
            raise ValueError(
                f"Insufficient data: got {len(arr)} samples, "
                f"need at least {min_samples}"
            )
        
        # Check for NaN
        if np.isnan(arr).any():
            nan_count = np.isnan(arr).sum()
            logger.warning(f"Data contains {nan_count} NaN values")
            # Replace with column means
            col_means = np.nanmean(arr, axis=0)
            for i in range(arr.shape[1]):
                arr[np.isnan(arr[:, i]), i] = col_means[i]
        
        return arr
    
    def _log_classification(self, result: RegimeResult) -> None:
        """Log classification result."""
        logger.info(
            f"{self.name}: Regime {result.regime} ({result.regime_name}) "
            f"with confidence {result.confidence:.2f}"
        )
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}(n_regimes={self.n_regimes}, {status})"
