"""
AMRCAIS - Adaptive Multi-Regime Cross-Asset Intelligence System

A quantitative finance research framework that integrates regime detection
with dynamic signal interpretation across multiple asset classes.

Core Innovation:
- Detects market regimes using an ensemble of 4+ independent classifiers
- Adapts signal interpretation based on detected regime
- Flags regime uncertainty when classifiers disagree (Disagreement Index > 0.6)

Architecture:
- Layer 1: Market Regime Classification (HMM, ML, Correlation, Volatility)
- Layer 2: Dynamic Signal Interpretation (5 analytical modules)
- Layer 3: Meta-Learning & Adaptation (performance tracking, recalibration)
"""

__version__ = "0.1.0"
__author__ = "AMRCAIS Development Team"

from src.regime_detection.ensemble import RegimeEnsemble
from src.data_pipeline.pipeline import DataPipeline

__all__ = [
    "RegimeEnsemble",
    "DataPipeline",
]
