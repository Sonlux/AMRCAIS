# AMRCAIS Copilot Instructions

## Project Overview

AMRCAIS (Adaptive Multi-Regime Cross-Asset Intelligence System) is a quantitative finance research project that integrates regime detection with dynamic signal interpretation across asset classes. The **core innovation** is treating market regimes as the foundation for all analysis—signals mean different things in different regimes.

## Architecture (Three-Layer System)

```
Layer 3: Meta-Learning    → Tracks classifier accuracy, triggers recalibration
Layer 2: Analytical Modules → 5 modules with regime-adaptive parameters
Layer 1: Regime Detection  → 4 classifiers + ensemble voter → (regime, confidence, disagreement)
```

**Four Market Regimes:** Risk-On Growth (1), Risk-Off Crisis (2), Stagflation (3), Disinflationary Boom (4)

## Critical Design Principle: Regime-First

**NEVER** write static signal interpretation. Always check regime before interpreting:

```python
# ❌ WRONG
if yield_curve_steepness > 0.5:
    signal = "bullish"

# ✅ CORRECT
regime = regime_classifier.get_current_regime()
if regime == "risk_on_growth":
    signal = "bullish" if yield_curve_steepness > 0.5 else "neutral"
elif regime == "stagflation":
    signal = "bearish" if yield_curve_steepness > 0.5 else "neutral"
```

## Directory Structure

```
src/regime_detection/     # HMM, ML, correlation, volatility classifiers + ensemble
src/modules/              # MacroEventTracker, YieldCurveAnalyzer, OptionsSurfaceMonitor,
                          # FactorExposureAnalyzer, CorrelationAnomalyDetector
src/meta_learning/        # Performance tracking, recalibration logic
src/data_pipeline/        # Fetchers, validators, storage
config/                   # regimes.yaml, data_sources.yaml, model_params.yaml (NO hardcoded params)
```

## Module Interface (All Modules Must Implement)

```python
class AnalyticalModule(ABC):
    def update_regime(self, regime: int, confidence: float): ...
    def analyze(self, data: pd.DataFrame) -> Dict:  # Returns signal, strength, explanation, regime_context
    def get_regime_parameters(self, regime: int) -> Dict: ...
```

## Code Standards

- **Python 3.10+** with full type hints (mandatory)
- **Google-style docstrings** with Args, Returns, Raises, Examples
- **Black** formatter, **pylint** score ≥ 8.5
- **Absolute imports only**: `from src.regime_detection.ensemble import RegimeEnsemble`
- **YAML configs** for all parameters (see `config/regimes.yaml` for regime-specific weights)

## Data Pipeline Rules

- **Sources (priority):** FRED API → yfinance → Alpha Vantage → cached (≤7 days old)
- **Validation required:** No NaN, prices > 0, High ≥ Low, no >20% single-day moves
- **Assets:** SPX, TLT, GLD, DXY, WTI, VIX (2010-present minimum)

## Testing Requirements

- **80% minimum coverage** with pytest
- **Walk-forward validation only**—never use future data
- **Mock external APIs** in tests, never use production keys
- **Backtest against known events:** 2008 crisis, March 2020 COVID, 2022 inflation spike

## Key Patterns

1. **Regime Disagreement Index** (0-1): When >0.6, flag uncertainty—historically precedes transitions
2. **Recalibration triggers:** >3 regime flips in 5 days, disagreement >0.7 for 10+ days, 25% error rate over 2 weeks
3. **Graceful degradation:** If one classifier fails, use remaining classifiers; never crash entire system

## Logging Convention

```python
logger = logging.getLogger(__name__)
# INFO for regime changes, WARNING for data issues, ERROR for classifier failures
logger.info(f"Regime: {name} (confidence={conf:.2f}, disagreement={disagr:.2f})")
```
