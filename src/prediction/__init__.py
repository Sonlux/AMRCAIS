"""
Phase 3: Prediction Engine.

Moves AMRCAIS from "what is the regime" to "what will the regime do
to asset prices."  Four sub-modules:

    ReturnForecaster       – Regime-conditional return forecasting
    TailRiskAnalyzer       – Regime-conditional VaR / CVaR with attribution
    PortfolioOptimizer     – Regime-aware mean-variance / Black-Litterman
    AlphaSignalGenerator   – Anomaly-based tradeable signals
"""

from src.prediction.return_forecaster import ReturnForecaster
from src.prediction.tail_risk import TailRiskAnalyzer
from src.prediction.portfolio_optimizer import PortfolioOptimizer
from src.prediction.alpha_signals import AlphaSignalGenerator

__all__ = [
    "ReturnForecaster",
    "TailRiskAnalyzer",
    "PortfolioOptimizer",
    "AlphaSignalGenerator",
]
