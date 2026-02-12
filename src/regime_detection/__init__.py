"""Regime detection module for AMRCAIS.

Contains classifiers for market regime identification:
- HMM (Hidden Markov Model) classifier
- ML (Random Forest) classifier  
- Correlation-based clustering classifier
- Volatility regime detector
- Ensemble voter with disagreement calculation
"""

from src.regime_detection.base import BaseClassifier
from src.regime_detection.hmm_classifier import HMMRegimeClassifier
from src.regime_detection.ml_classifier import MLRegimeClassifier
from src.regime_detection.correlation_classifier import CorrelationRegimeClassifier
from src.regime_detection.volatility_classifier import VolatilityRegimeClassifier
from src.regime_detection.ensemble import RegimeEnsemble

__all__ = [
    "BaseClassifier",
    "HMMRegimeClassifier",
    "MLRegimeClassifier",
    "CorrelationRegimeClassifier",
    "VolatilityRegimeClassifier",
    "RegimeEnsemble",
]
