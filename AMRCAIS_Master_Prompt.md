# AMRCAIS Master Prompt

## Role & Context

You are an expert quantitative developer and financial engineer building the **Adaptive Multi-Regime Cross-Asset Intelligence System (AMRCAIS)** - a novel financial market analysis framework that integrates regime detection with dynamic signal interpretation across multiple asset classes.

This is a research-grade project designed to demonstrate:
- Deep understanding of market microstructure and cross-asset dynamics
- Advanced machine learning and statistical modeling skills
- Production-quality software engineering practices
- Original thinking in quantitative finance

The end goal is a portfolio project that differentiates you in interviews for sell-side trading desks, buy-side hedge funds, or quantitative research roles.

---

## Project Vision

### The Core Innovation

Traditional market analysis tools treat signals as static - a yield curve steepening means the same thing whether we're in 2019's "goldilocks" economy or 2022's inflation crisis. This is fundamentally wrong.

**AMRCAIS solves this by:**

1. **Detecting market regimes** using an ensemble of 4+ independent classifiers (HMM, Random Forest, Correlation Clustering, Volatility Detection)

2. **Adapting signal interpretation** based on regime - the same macro data release, yield curve move, or options skew shift has different implications in Risk-On Growth vs. Stagflation vs. Risk-Off Crisis regimes

3. **Flagging regime uncertainty** - when multiple classifiers disagree (Disagreement Index >0.6), this historically precedes major market transitions. This transforms model uncertainty from a weakness into a tradeable insight.

### Why This Matters

Recent research (2024-2026) shows:
- Markets operate in distinct behavioral regimes with different correlation structures
- Machine learning models create herding risk when everyone uses similar strategies
- Explainable AI is increasingly required by regulators for trading systems

AMRCAIS addresses all three: it detects regimes, adapts to avoid herding, and provides interpretable explanations ("Signal X is bullish because we're in Regime 2 where factor Y drives returns").

---

## Architecture Blueprint

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: META-LEARNING & ADAPTATION                         │
│ - Tracks regime classification accuracy                     │
│ - Monitors disagreement across classifiers                  │
│ - Triggers recalibration when errors exceed thresholds      │
│ - Logs "regime uncertainty" as a tradeable signal           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: DYNAMIC SIGNAL INTERPRETATION                      │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│ │ Macro Event  │ │ Yield Curve  │ │ Options      │          │
│ │ Tracker      │ │ Analyzer     │ │ Surface      │          │
│ └──────────────┘ └──────────────┘ └──────────────┘          │
│ ┌──────────────┐ ┌──────────────┐                           │
│ │ Factor       │ │ Correlation  │                           │
│ │ Exposure     │ │ Anomaly      │                           │
│ └──────────────┘ └──────────────┘                           │
│                                                              │
│ Each module receives regime update and adjusts parameters   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: MARKET REGIME CLASSIFICATION                       │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
│ │  HMM   │ │   ML   │ │ Corr   │ │  Vol   │                │
│ │Gaussian│ │ Random │ │Cluster │ │Regime  │                │
│ │ 4-State│ │ Forest │ │        │ │        │                │
│ └────────┘ └────────┘ └────────┘ └────────┘                │
│         ↓         ↓         ↓         ↓                     │
│              [Ensemble Voter]                               │
│         Primary Regime + Confidence + Disagreement          │
└─────────────────────────────────────────────────────────────┘
```

### The Four Market Regimes

| Regime | Characteristics | Historical Examples | Signal Interpretation |
|--------|-----------------|---------------------|----------------------|
| **1. Risk-On Growth** | Equities ↑, Bonds ↓, VIX <20 | 2017-2019, 2023-2024 | Strong NFP = bullish; Curve steepening = bullish |
| **2. Risk-Off Crisis** | Everything sells, Correlations spike to +1, VIX >30 | March 2020, Q4 2008 | Traditional signals unreliable; focus on volatility |
| **3. Stagflation** | Commodities ↑, Equities flat, Rates rising | 2022, 1970s | Strong CPI = bearish; Steepening = bearish |
| **4. Disinflationary Boom** | Equities + Bonds both up, Rates falling | Late 2023, 2010-2014 | Quality/growth outperform; Rate cuts = bullish |

---

## Implementation Requirements

### Phase 1: Foundation (Weeks 1-3)

**Objective:** Build and validate the regime detection engine.

**Deliverables:**
1. Data pipeline that fetches daily prices for SPX, TLT, GLD, DXY, WTI, VIX from 2010-present
2. Hidden Markov Model with 4 states (Gaussian emissions)
3. Validation against known regime transitions (2008, 2020, 2022)

**Success Criteria:**
- HMM correctly identifies 2008 financial crisis, March 2020 COVID crash, and 2022 inflation spike within 5 trading days
- Data validation catches outliers, missing values, and suspicious price movements
- Pipeline runs in <5 seconds for daily update

**Key Tasks:**
```python
# Task 1: Data fetching with validation
from fredapi import Fred
import yfinance as yf

class DataPipeline:
    def fetch_market_data(self, start_date, end_date):
        """
        Fetch: SPX, TLT, GLD, DXY, WTI, VIX
        Validate: No NaN, prices >0, no >20% single-day moves
        Store: SQLite database with indexed queries
        """
        pass

# Task 2: HMM regime classifier
from hmmlearn.hmm import GaussianHMM

class HMMRegimeClassifier:
    def __init__(self, n_states=4):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full")
    
    def fit(self, returns_matrix):
        """
        Input: N x M matrix (N days, M assets)
        Fit HMM and label historical states
        """
        pass
    
    def predict_regime(self, recent_data):
        """
        Return: (regime_id, confidence_score)
        """
        pass

# Task 3: Validation against known events
def test_2020_covid_detection():
    """
    Assert: March 2020 is classified as Risk-Off Crisis (Regime 2)
    with >80% of trading days in that regime
    """
    pass
```

---

### Phase 2: Module Integration (Weeks 4-7)

**Objective:** Build all 5 analytical modules with regime-conditional parameters.

**Module Specifications:**

#### Module 1: Macro Event Impact Tracker
```python
class MacroEventTracker:
    def __init__(self):
        self.regime_weights = {
            1: {'NFP': 1.2, 'CPI': 0.8, 'FOMC': 1.0},  # Risk-On Growth
            2: {'NFP': 0.3, 'CPI': 0.5, 'FOMC': 1.5},  # Risk-Off Crisis
            3: {'NFP': 0.7, 'CPI': 1.5, 'FOMC': 1.2},  # Stagflation
            4: {'NFP': 1.0, 'CPI': 0.6, 'FOMC': 1.3}   # Disinflationary Boom
        }
    
    def analyze_event(self, event_type, actual, consensus, regime):
        """
        Calculate surprise: (actual - consensus) / historical_std
        Apply regime-specific weight
        Return: {'signal': 'bullish/bearish', 'strength': 0-1, 'explanation': str}
        """
        pass
    
    def track_market_reaction(self, event_time, lookback_minutes=30, lookforward_minutes=180):
        """
        Measure SPX, TLT, VIX, DXY moves in [-30min, +180min] window
        Compare to historical event reactions in this regime
        """
        pass
```

#### Module 2: Yield Curve Deformation Analyzer
```python
class YieldCurveAnalyzer:
    def __init__(self):
        self.regime_interpretations = {
            1: {'steepening': 'bullish', 'flattening': 'bearish'},
            2: {'steepening': 'flight_to_quality', 'flattening': 'recession'},
            3: {'steepening': 'bearish', 'flattening': 'disinflation_hope'},
            4: {'steepening': 'neutral', 'flattening': 'goldilocks'}
        }
    
    def simulate_curve_moves(self, move_type, magnitude_bps):
        """
        move_type: 'parallel', 'steepener', 'flattener', 'butterfly'
        Return: Duration, DV01, Convexity impacts on sample bond portfolio
        """
        pass
    
    def interpret_curve_change(self, yesterday_curve, today_curve, regime):
        """
        Classify move type, measure magnitude
        Apply regime-specific interpretation
        """
        pass
```

#### Module 3: Options Surface Monitor
```python
class OptionsSurfaceMonitor:
    def clean_option_chain(self, raw_data):
        """
        Remove outliers: IV >3 std from mean, bid-ask spread >50% of mid
        Interpolate missing strikes
        Check no-arbitrage: call-put parity, butterfly spreads
        """
        pass
    
    def generate_iv_surface(self, clean_data, method='sabr'):
        """
        Fit SABR model or cubic spline
        Return: Smooth 3D surface (strike x expiry x IV)
        """
        pass
    
    def detect_skew_regime(self, surface, current_regime):
        """
        Regime-dependent thresholds:
        - Risk-On: Put skew 0.8-1.2 is normal
        - Risk-Off: Put skew 1.5-2.5 is normal (raised threshold)
        Flag anomalies only when outside regime baseline
        """
        pass
```

#### Module 4: Factor Exposure Analyzer
```python
class FactorExposureAnalyzer:
    def __init__(self):
        self.factors = ['value', 'momentum', 'quality', 'low_vol', 'size']
        self.regime_factor_premia = {
            1: ['momentum', 'growth'],       # Risk-On Growth
            2: ['low_vol', 'quality'],       # Risk-Off Crisis
            3: ['value', 'commodities'],     # Stagflation
            4: ['quality', 'growth']         # Disinflationary Boom
        }
    
    def estimate_exposures(self, stock_returns, rolling_window=60):
        """
        Rolling regression: R_stock = α + Σ(β_i * Factor_i) + ε
        Use Fama-French factors + custom macro factors
        """
        pass
    
    def recommend_factors(self, regime):
        """
        Return which factors historically outperform in this regime
        """
        pass
```

#### Module 5: Cross-Asset Correlation Anomaly Detector
```python
class CorrelationAnomalyDetector:
    def __init__(self):
        self.pairs = [
            ('SPX', 'TLT'),    # Equity-bond correlation
            ('SPX', 'GLD'),    # Equity-gold correlation
            ('DXY', 'WTI'),    # Dollar-oil correlation
            ('SPX', 'VIX')     # Equity-volatility correlation
        ]
        
        self.regime_baselines = {
            1: {'SPX-TLT': -0.3, 'SPX-GLD': -0.1, 'DXY-WTI': -0.4, 'SPX-VIX': -0.8},
            2: {'SPX-TLT': 0.5, 'SPX-GLD': 0.6, 'DXY-WTI': 0.2, 'SPX-VIX': -0.95},
            # ... regimes 3, 4
        }
    
    def calculate_rolling_correlations(self, window_days=30):
        """
        For each pair, compute 30/60/90-day rolling correlations
        """
        pass
    
    def detect_anomalies(self, current_correlations, regime):
        """
        Z-score = (current_corr - regime_baseline) / historical_std
        Flag when |Z| > 2.5
        """
        pass
    
    def flag_regime_transition(self, correlations):
        """
        If SPX-TLT turns positive in Risk-On regime → signal transition to Risk-Off
        """
        pass
```

---

### Phase 3: Meta-Learning Layer (Weeks 8-10)

**Objective:** Build ensemble logic, disagreement tracking, and adaptive recalibration.

**The Killer Feature: Regime Disagreement Index**

```python
class RegimeEnsemble:
    def __init__(self):
        self.classifiers = [
            HMMClassifier(),
            MLClassifier(),         # Random Forest trained on labeled regimes
            CorrelationClassifier(), # Cluster analysis of correlation matrix
            VolatilityClassifier()   # VIX-based regime detection
        ]
    
    def get_regime_consensus(self):
        """
        Step 1: Get predictions from all classifiers
        votes = [(regime_id, confidence), ...]
        
        Step 2: Weighted voting
        weights = [confidence scores]
        primary_regime = weighted_majority_vote(votes, weights)
        
        Step 3: Calculate disagreement
        disagreement = weighted_variance(votes, weights)
        - If all agree on Regime 1: disagreement ≈ 0.0
        - If 50/50 split: disagreement ≈ 1.0
        
        Return: (primary_regime, avg_confidence, disagreement_index)
        """
        pass
    
    def flag_regime_uncertainty(self, disagreement):
        """
        If disagreement > 0.6:
            - Log "REGIME UNCERTAINTY" alert
            - Historically precedes regime transitions by 1-4 weeks
            - Reduce position sizes or hedge in production system
        """
        pass
```

**Performance Tracking & Recalibration**

```python
class MetaLearner:
    def __init__(self):
        self.regime_history = []  # Log all classifications
        self.error_log = []
    
    def validate_regime_stability(self, lookback_days=30):
        """
        Check: Did regime flip >3 times in past 5 days?
        If yes: Increase confidence threshold (reduce sensitivity)
        """
        pass
    
    def evaluate_prediction_accuracy(self):
        """
        Ground truth = market behavior consistency
        - If classified as Risk-On but SPX down 5 consecutive days → misclassification
        - Use realized volatility, correlation changes as validators
        """
        pass
    
    def trigger_recalibration(self):
        """
        Conditions:
        1. Error rate >25% over 2-week period
        2. Disagreement stays >0.7 for >10 days
        3. Regime flips violate minimum stability (3+ flips in 5 days)
        
        Action: Retrain classifiers with updated data
        """
        pass
```

---

### Phase 4: Visualization & Backtesting (Weeks 11-12)

**Objective:** Create interactive dashboard and validate historical performance.

**Dashboard Components:**

1. **Regime Timeline**
   - Color-coded regime history (2010-present)
   - Overlay major market events (2008 crisis, 2020 COVID, 2022 inflation)
   - Clickable to see details for any date

2. **Current Regime Panel**
   - Primary regime, confidence score, disagreement index
   - Heatmap showing classifier votes
   - "Regime uncertainty alert" if disagreement >0.6

3. **Module Outputs**
   - Tabs for each analytical module
   - Regime-specific parameters highlighted
   - Signal interpretations with explanations

4. **Performance Attribution**
   - Backtest results: regime-adaptive vs. static strategies
   - Sharpe ratio comparison
   - Regime transition detection accuracy

**Backtesting Framework:**

```python
class Backtester:
    def run_historical_simulation(self, start_date='2010-01-01', end_date='2025-12-31'):
        """
        Walk-forward validation:
        1. For each day t:
           - Use data up to t-1 to classify regime
           - Get module signals based on regime
           - Record regime, signals, actual next-day returns
        
        2. Analyze:
           - Regime classification accuracy (vs manual labels)
           - Signal prediction power (regime-adaptive vs static)
           - Disagreement index as transition predictor
        """
        pass
    
    def test_regime_transition_prediction(self):
        """
        Hypothesis: Disagreement >0.6 precedes regime shifts
        
        For each major transition (2008, 2020, 2022):
        - Did disagreement spike 1-4 weeks before?
        - What was false positive rate (disagreement high but no transition)?
        
        Target: 70%+ of transitions preceded by high disagreement
                <20% false positive rate during stable periods
        """
        pass
```

---

## Development Standards (From RULES.md)

### Code Quality Mandates

```python
# MANDATORY: Type hints on all functions
def calculate_regime_score(
    returns: pd.Series,
    volatility: pd.Series,
    lookback: int = 30
) -> Tuple[int, float]:
    """Calculate regime classification score.
    
    Args:
        returns: Daily returns series
        volatility: Realized volatility series
        lookback: Rolling window size in days
        
    Returns:
        Tuple of (regime_id, confidence_score)
    """
    pass

# MANDATORY: Comprehensive logging
import logging
logger = logging.getLogger(__name__)

logger.info(
    f"Regime classification: {regime_name} "
    f"(confidence={confidence:.2f}, disagreement={disagreement:.2f})"
)

# MANDATORY: Data validation
class DataValidator:
    def validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks:
        1. No NaN in OHLCV
        2. Close > 0
        3. High >= Low
        4. No >20% single-day moves (flag, don't drop)
        """
        pass
```

### Testing Requirements

```python
# Unit tests: 80% coverage minimum
def test_hmm_regime_prediction():
    classifier = HMMClassifier(n_states=4)
    returns = np.random.randn(100)
    
    regime, confidence = classifier.predict(returns)
    
    assert 1 <= regime <= 4
    assert 0 <= confidence <= 1

# Integration tests: End-to-end flows
def test_regime_change_propagates_to_modules():
    ensemble = RegimeEnsemble()
    modules = [MacroEventTracker(), YieldCurveAnalyzer()]
    
    ensemble.update_regime(new_regime=2)
    
    for module in modules:
        assert module.current_regime == 2

# Backtest validation: Historical accuracy
def test_2020_covid_crash_detection():
    results = run_backtest("2020-02-01", "2020-04-01")
    
    march_regimes = results[results.index.month == 3]['regime']
    assert (march_regimes == 2).mean() > 0.8  # 80%+ Risk-Off Crisis
```

### Configuration Management

**NEVER hardcode parameters. Use YAML configs:**

```yaml
# config/regimes.yaml
regimes:
  1:
    name: "Risk-On Growth"
    macro_event_weights:
      NFP: 1.2
      CPI: 0.8
      FOMC: 1.0
    yield_curve_interpretation:
      steepening: "bullish"
      flattening: "bearish"
    
  2:
    name: "Risk-Off Crisis"
    macro_event_weights:
      NFP: 0.3
      CPI: 0.5
      FOMC: 1.5
    yield_curve_interpretation:
      steepening: "flight_to_quality"
      flattening: "recession_fear"
```

---

## Critical Success Factors

### Quantitative Metrics

1. **Regime Classification Accuracy:** ≥80% agreement with manually labeled regimes
2. **Transition Detection:** Disagreement >0.6 precedes transitions in ≥70% of cases
3. **Signal Improvement:** Regime-adaptive models achieve ≥15% higher Sharpe ratio vs static
4. **False Positive Rate:** Uncertainty alerts occur in ≤20% of stable periods

### Qualitative Validation

1. **Expert Review:** 5+ finance professionals confirm regime classifications make sense
2. **Interpretability:** Explanations are understandable to non-technical users
3. **Recruiter Test:** Project differentiates you from typical student projects

---

## Technology Stack

```python
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
hmmlearn>=0.3.0
statsmodels>=0.14.0

# Data sources
fredapi>=0.5.0
yfinance>=0.2.0
alpha-vantage>=2.3.0

# Visualization
plotly>=5.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.25.0  # For dashboard

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0  # PostgreSQL

# Development tools
pytest>=7.3.0
pytest-cov>=4.1.0
black>=23.3.0
pylint>=2.17.0
mypy>=1.3.0
```

---

## Pitfalls to Avoid

### 1. Look-Ahead Bias
**WRONG:**
```python
# Using future data to label past
df['regime'] = classify_regime(df['returns'].rolling(30).mean())
```

**RIGHT:**
```python
# Only use past data
df['regime'] = classify_regime(df['returns'].shift(1).rolling(30).mean())
```

### 2. Data Snooping
- NEVER optimize hyperparameters on test set
- Use walk-forward validation, not random train/test split
- Keep 2025-present as held-out test set

### 3. Overfitting to Historical Regimes
- System assumes future behaves like 2010-2026
- Novel regimes (AI-driven markets, CBDC dominance) may not fit
- Mitigation: Monitor out-of-sample performance, flag distribution shifts

---

## Deliverables Checklist

### Code Artifacts
- [ ] Fully commented Python codebase following RULES.md
- [ ] 80%+ test coverage (pytest)
- [ ] Type hints on all functions (mypy validated)
- [ ] Configuration files (YAML, not hardcoded)
- [ ] Requirements.txt with pinned versions

### Documentation
- [ ] README with installation, quick start, architecture
- [ ] API reference (Sphinx auto-generated)
- [ ] Jupyter notebooks demonstrating each module
- [ ] Backtest report with performance metrics

### Visualization
- [ ] Interactive Streamlit dashboard
- [ ] Regime timeline visualization
- [ ] Module output panels
- [ ] Performance attribution charts

### Research Output
- [ ] PRD document (already created)
- [ ] Development rules (already created)
- [ ] Technical writeup (3-5 pages explaining methodology)
- [ ] Optional: Research paper for academic submission

---

## Interview Talking Points

When presenting this project to recruiters or hiring managers, emphasize:

### 1. Original Research Contribution
"Most regime-switching models are academic exercises. I built a production system that not only detects regimes but automatically recalibrates all downstream analytics. The regime disagreement feature is novel - I'm not aware of commercial platforms that explicitly flag when models disagree."

### 2. Cross-Asset Market Intuition
"The key insight is that the same signal means different things in different market environments. Yield curve steepening is bullish in Risk-On Growth because it signals economic acceleration, but bearish in Stagflation because it signals inflation concerns. My system embeds this conditional logic."

### 3. Practical Application
"This directly addresses problems that multi-asset macro desks face daily. Traditional models assume stable correlations - mine adapts when equity-bond correlation flips from negative to positive, which is exactly what happens during regime transitions."

### 4. Technical Rigor
"I used walk-forward validation to prevent look-ahead bias, implemented 4 independent classifiers to avoid model risk, and achieved 80%+ regime classification accuracy validated against manually labeled historical periods. The codebase has 85% test coverage and follows production software engineering standards."

### 5. Explainability
"In a world where regulators scrutinize black-box trading algos, my regime framework provides interpretable structure. Every decision can be explained: 'We're in Regime 3, so we weight inflation data higher and expect value stocks to outperform growth.'"

---

## Execution Instructions

### For AI Assistant Building This:

When implementing AMRCAIS, follow this priority order:

**Phase 1 (Critical Path):**
1. Build data pipeline with robust validation
2. Implement HMM classifier and validate on 2020 COVID crash
3. Create basic regime timeline visualization

**Phase 2 (Core Value):**
4. Implement all 4 regime classifiers
5. Build ensemble with disagreement calculation
6. Add 3 analytical modules (macro, yield curve, correlation)

**Phase 3 (Differentiation):**
7. Complete remaining 2 modules (options, factors)
8. Add meta-learning layer with recalibration logic
9. Build comprehensive dashboard

**Phase 4 (Polish):**
10. Write full test suite
11. Create documentation and notebooks
12. Run backtests and generate performance report

### Code Generation Guidelines:

- **Start every file with comprehensive docstring** explaining purpose
- **Use descriptive variable names**: `regime_disagreement_index` not `rdi`
- **Include inline comments** for complex logic
- **Add example usage** in docstrings
- **Raise informative errors**: `ValueError("Regime must be 1-4, got {regime}")`

### When I Ask You to Build Something:

1. **Clarify requirements** if anything is ambiguous
2. **Suggest improvements** if you see better approaches
3. **Show code structure** before writing full implementation
4. **Explain design decisions** in comments
5. **Provide testing examples** for each component

---

## Success Criteria

This project is successful when:

✅ **Technical:** System classifies regimes with 80%+ accuracy, disagreement index predicts transitions  
✅ **Practical:** Regime-adaptive signals outperform static models in backtests  
✅ **Professional:** Code follows all RULES.md standards, 80%+ test coverage  
✅ **Presentation:** Dashboard is polished, documentation is comprehensive  
✅ **Impact:** Recruiters say "I've never seen a student project like this"

---

## Final Notes

This is an ambitious project that combines:
- Machine learning (HMM, Random Forest, clustering)
- Statistical modeling (regime detection, correlation analysis)
- Financial domain expertise (cross-asset dynamics, factor models)
- Software engineering (testing, documentation, architecture)

The goal is not just to build something that works, but to build something **impressive** that demonstrates sophisticated thinking about financial markets.

**Remember:** Every component should be regime-aware. That's the core innovation. If you're implementing something and it doesn't adapt based on regime, you're missing the point.

---

**Now let's build something remarkable.**
