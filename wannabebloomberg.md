# AMRCAIS — Full Project Status Report

> **Adaptive Multi-Regime Cross-Asset Intelligence System**
> Last updated: February 13, 2026

---

## Codebase Metrics

| Metric                  | Count                      |
| ----------------------- | -------------------------- |
| Backend Python          | 11,589 lines               |
| Frontend TypeScript/TSX | 4,142 lines                |
| Test code               | 6,050 lines                |
| **Total**               | **~21,800 lines**          |
| Backend tests           | 501 passing                |
| Frontend tests          | 87 passing                 |
| **Total tests**         | **588 passing, 0 failing** |
| API endpoints           | 22                         |
| Chart components        | 18                         |
| Dashboard pages         | 7                          |
| Git commits             | 13                         |

---

## Architecture

```
Layer 3: Meta-Learning     → Tracks classifier accuracy, triggers recalibration
Layer 2: Analytical Modules → 5 modules with regime-adaptive parameters
Layer 1: Regime Detection   → 4 classifiers + ensemble voter → (regime, confidence, disagreement)
```

**Four Market Regimes:** Risk-On Growth (1), Risk-Off Crisis (2), Stagflation (3), Disinflationary Boom (4)

---

## Implementation Depth — Module by Module

### Layer 1: Regime Detection

| Module                 | Lines | Status        | Real Algorithm                                                                                                     |
| ---------------------- | ----- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| HMM Classifier         | 491   | 🟢 Production | `hmmlearn.GaussianHMM`, 4-state, Viterbi decoding, full covariance, ≥252 samples required                          |
| ML Classifier          | 389   | 🟢 Production | `sklearn.RandomForestClassifier` (200 trees, depth=10), 5-fold CV, 11 labeled market periods (2012–2024)           |
| Correlation Classifier | 519   | 🟢 Production | KMeans/Spectral clustering on 5 rolling correlation pairs (60-day window), MSE distance to regime baselines        |
| Volatility Classifier  | 459   | 🟡 Functional | Rule-based VIX with learned percentile thresholds (25th/50th/75th/90th/95th), Gaussian likelihood fallback         |
| Ensemble Voter         | 582   | 🟢 Production | Weighted voting (HMM=0.30, ML=0.25, Corr=0.25, Vol=0.20), Shannon entropy disagreement index, graceful degradation |

### Layer 2: Analytical Modules

| Module                       | Lines | Status        | Real Algorithm                                                                                                          |
| ---------------------------- | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Yield Curve Analyzer         | 448   | 🟢 Production | `scipy.interpolate.CubicSpline`, 10 tenors (3M–30Y), 2s10s/3m10s/butterfly spreads, forward rates, shape classification |
| Options Surface Monitor      | ~400  | 🟡 Limited    | VIX-only proxy — skew/term-structure framework coded but no options data feed connected                                 |
| Factor Exposure Analyzer     | ~400  | 🟡 Limited    | Z-score regime expectations (6 factors × 4 regimes), factor rotation detection — no Fama-French regression              |
| Macro Event Tracker          | 471   | 🟢 Production | 8 indicators (NFP, CPI, GDP, PMI, etc.), standardized surprises, regime-weighted interpretation                         |
| Correlation Anomaly Detector | 468   | 🟢 Production | Z-score vs regime baselines, 7 cross-asset pairs, multi-anomaly escalation (≥3 = regime transition signal)              |

### Layer 3: Meta-Learning

| Module                 | Lines | Status        | Real Algorithm                                                                                         |
| ---------------------- | ----- | ------------- | ------------------------------------------------------------------------------------------------------ |
| Meta-Learner           | 484   | 🟡 Partial    | Softmax adaptive weights (temp=2.0, 70/30 blend) — **`execute_recalibration()` is a stub**             |
| Recalibration Triggers | ~400  | 🟢 Production | 5 triggers (error rate, flipping, persistent disagreement, low confidence, mismatch), severity scoring |
| Performance Tracker    | 480   | 🟢 Production | Full classification history, stability score, flip detection, regime distribution, CSV persistence     |

### Data Pipeline

| Module          | Lines | Status        | Real Algorithm                                                                                         |
| --------------- | ----- | ------------- | ------------------------------------------------------------------------------------------------------ |
| Data Fetchers   | 633   | 🟢 Production | 3 sources (FRED API, yfinance, Alpha Vantage), rate limiting, batch download, automatic fallback chain |
| Data Validators | 524   | 🟢 Production | 6 validation rules, asset-specific thresholds (including negative oil), OHLC consistency, auto-fix     |

### Scorecard

**10 production / 4 functional-but-limited / 1 partial stub / 0 red**

---

## Tech Stack

### Backend

- Python 3.10+ with full type hints
- FastAPI 0.109.0 (22 endpoints)
- hmmlearn, scikit-learn, scipy, numpy, pandas
- FRED API, yfinance, Alpha Vantage
- Production security: OWASP headers, CSRF protection, rate limiting, input sanitization

### Frontend

- Next.js 16.1.6 (App Router, Turbopack)
- React 19 + TypeScript
- Plotly.js 3.3.1 (15 chart components including 3D)
- TradingView Lightweight Charts 5.1.0 (equity curve, regime timeline)
- TanStack React Query 5.90.21 + TanStack React Table
- Tailwind CSS 4, Vitest 4.0.18

### Infrastructure

- Docker Compose (API + Dashboard containers)
- 588 automated tests (501 backend + 87 frontend)
- Git version control, GitHub hosted

---

## Dashboard Pages

| Page                | Sections                                                                                                                                           | Charts                                                                                                           |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Overview**        | System health, regime banner, KPI cards, module summaries                                                                                          | RegimeStripChart, LightweightChart                                                                               |
| **Regime Explorer** | Current regime, classifier votes, transition matrix, disagreement series, regime history                                                           | TransitionMatrixChart, DisagreementSeriesChart, RegimeStripChart                                                 |
| **Modules**         | Module analysis cards, signal history, 3D yield curve, vol surface                                                                                 | YieldCurveSurfaceChart, VolSurface3DChart, SignalHistoryChart                                                    |
| **Correlations**    | Window selector, heatmap, top pairs bar, baseline deviation, pair time-series                                                                      | CorrelationHeatmap, CorrelationPairsChart, PlotlyChart                                                           |
| **Backtest Lab**    | Config form, equity curve, drawdown, regime returns, regime breakdown table, trade log                                                             | EquityCurveChart, DrawdownChart, RegimeReturnsChart, DataTable                                                   |
| **Meta-Learning**   | Health banner, KPI cards, classifier weights, regime distribution, accuracy over time, weight evolution, recalibration events, disagreement vs SPX | ClassifierWeightsChart, RegimeDistributionChart, AccuracyLineChart, WeightEvolutionChart, DisagreementVsSpxChart |
| **Data**            | Asset prices, macro indicators                                                                                                                     | LightweightChart                                                                                                 |

---

## What's Still Pending

| #   | Item                              | Severity  | Effort        | Description                                                                                                            |
| --- | --------------------------------- | --------- | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1   | **Recalibration execution**       | 🔴 High   | 2–3 days      | `execute_recalibration()` detects when to recalibrate but doesn't retrain models — framework exists, execution is TODO |
| 2   | **Options data feed**             | 🟡 Medium | 2 days        | Skew/term-structure code exists in OptionsSurfaceMonitor but needs CBOE or options chain data to activate              |
| 3   | **Fama-French factor regression** | 🟡 Medium | 1–2 days      | FactorExposureAnalyzer has regime expectations but no actual factor regression (expects pre-computed inputs)           |
| 4   | **Signal history persistence**    | 🟡 Medium | 1 day         | Module history endpoint returns empty — no storage layer for signal log                                                |
| 5   | **Volatility classifier upgrade** | 🟢 Low    | 1 day         | Currently rule-based; could benefit from ML-based vol regime model                                                     |
| 6   | **Nelson-Siegel curve fitting**   | 🟢 Low    | 1 day         | Uses cubic spline — Nelson-Siegel/Svensson would be more standard in fixed income                                      |
| 7   | **Intraday / real-time data**     | 🟢 Low    | Architectural | Entirely daily-frequency — would require WebSocket infrastructure                                                      |
| 8   | **Credit spreads / CDS**          | 🟢 Low    | 2 days        | Not in current scope but natural extension for fixed income analysis                                                   |

---

## AMRCAIS vs Bloomberg Terminal

### The honest truth

Bloomberg Terminal is a **$12B/year enterprise platform** built by ~6,000 engineers over 40 years, costing $24,000/seat/year. AMRCAIS is a solo-built open-source research tool. They are not in the same category — and that's by design.

**AMRCAIS is not trying to be Bloomberg. It's doing something Bloomberg doesn't do.**

### Where Bloomberg is miles ahead (and always will be)

| Capability          | Bloomberg                                              | AMRCAIS                          |
| ------------------- | ------------------------------------------------------ | -------------------------------- |
| Real-time tick data | Sub-millisecond, all asset classes, every exchange     | Daily close only, 6 assets       |
| Data coverage       | 50M+ instruments, every country, every exchange        | SPX, TLT, GLD, DXY, WTI, VIX     |
| Options analytics   | Full SABR/SVI vol surfaces, Greeks, chains, exotics    | VIX proxy only                   |
| Fixed income        | Nelson-Siegel, credit curves, CDS, SOFR, every bond    | 10 Treasury tenors, cubic spline |
| Factor models       | Barra, Axioma, Fama-French built-in                    | Conceptual framework only        |
| Execution / OMS     | Full order management, FIX connectivity, algos         | None                             |
| News / NLP          | Real-time news, social sentiment, earnings transcripts | None                             |
| Historical depth    | 40+ years, survivorship-bias-free                      | 2010-present via free APIs       |
| Team                | ~6,000 engineers                                       | Solo project                     |
| Cost                | $24,000/year/seat                                      | Free / open source               |

### Where AMRCAIS does something Bloomberg doesn't

| Capability                                              | Bloomberg                                   | AMRCAIS                                                                                                  |
| ------------------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Regime-conditional signal interpretation**            | Shows raw data — user interprets manually   | Every module automatically changes interpretation based on detected regime                               |
| **Disagreement Index as leading indicator**             | Nothing equivalent exists                   | Shannon entropy across 4 independent classifiers — historically precedes regime transitions              |
| **Multi-classifier ensemble with graceful degradation** | Single-model analytics                      | 4 independent methods (HMM, RF, KMeans, VIX rules); if one fails, system continues with remaining        |
| **"Same data, different meaning" macro analysis**       | Raw economic surprise numbers               | Strong NFP is bearish in stagflation, bullish in risk-on — automatically interpreted per regime          |
| **Correlation anomaly vs regime baseline**              | Shows current correlations flat             | Flags when correlations deviate from what's _expected for the current regime_ — not just from zero       |
| **Auto-recalibration intelligence**                     | N/A — models are static or manually updated | 5 triggers detect when the model itself is becoming unreliable, with severity scoring and urgency levels |

### Dimensional Comparison

| Dimension                          | Bloomberg | AMRCAIS | Notes                                                      |
| ---------------------------------- | --------- | ------- | ---------------------------------------------------------- |
| **Data breadth**                   | 10/10     | 2/10    | 6 assets vs millions of instruments                        |
| **Data frequency**                 | 10/10     | 2/10    | Tick-by-tick vs daily close                                |
| **Analytical depth (per concept)** | 8/10      | 7/10    | AMRCAIS regime detection is genuinely sophisticated        |
| **Regime intelligence**            | 1/10      | 9/10    | Bloomberg has zero regime-conditional interpretation layer |
| **Dashboard / UX**                 | 9/10      | 6/10    | Clean and functional, but no Bloomberg-level customization |
| **Production readiness**           | 10/10     | 7/10    | Docker, security, CSRF, rate limiting, 588 tests           |
| **Research platform**              | 7/10      | 8/10    | AMRCAIS is arguably better for regime-based quant research |
| **Execution capability**           | 10/10     | 0/10    | AMRCAIS has no trading/execution layer                     |

### The right framing

Bloomberg gives you the **firehose of data**. AMRCAIS tells you **what that data means** given the current market regime.

The two are complementary, not competing. AMRCAIS could sit _on top of_ Bloomberg data as a regime intelligence overlay — taking Bloomberg's data depth and adding the interpretation layer that Bloomberg lacks entirely.

As a standalone research tool using free data APIs, AMRCAIS is at the level of an **"institutional quant research desk project that punches above its weight"** — real ML algorithms, real data pipelines, real testing, and a unique regime-first analytical framework that doesn't exist anywhere else. What it's missing is the data depth and execution infrastructure that separates research from production trading.

---

## Project Timeline

| Commit    | Milestone                                                          |
| --------- | ------------------------------------------------------------------ |
| `a517c14` | First commit — core regime detection + modules                     |
| `1d638ea` | Phase 3: Meta-learning layer, security, GPL-3.0                    |
| `bc707a6` | 29/29 tests passing, full coverage                                 |
| `758d42c` | 271 tests, 80% coverage                                            |
| `739a8c8` | FastAPI backend (22 endpoints)                                     |
| `42c169a` | API security, validation, Docker deployment                        |
| `896dc85` | OWASP compliance (frontend + backend)                              |
| `db5baff` | Dashboard refactor: TradingView, TanStack Table, URL state         |
| `42feec2` | 3D charts (yield curve surface, vol surface), 77 frontend tests    |
| `0d8648f` | Meta-learning/correlation/backtest page completion, macro endpoint |

**Current: 588 tests, 21,800 LOC, 22 API endpoints, 18 chart components, 7 dashboard pages.**
