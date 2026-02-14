# Product Requirements Document

## Adaptive Multi-Regime Cross-Asset Intelligence System (AMRCAIS)

**Version:** 2.0  
**Date:** February 13, 2026  
**Status:** Active Development — Phase 0 Complete (~70%)

---

## 1. Executive Summary

The Adaptive Multi-Regime Cross-Asset Intelligence System (AMRCAIS) is a **regime-conditional decision intelligence platform** for financial markets. It integrates regime detection, macro event analysis, factor decomposition, options analytics, and cross-asset correlation monitoring into a single adaptive framework that automatically recalibrates signal interpretation based on the current market state.

### Core Thesis

> **Bloomberg shows you what happened. AMRCAIS tells you what it means, what's about to change, and what to do about it.**

Traditional platforms (Bloomberg, Refinitiv) are data terminals — they aggregate and display data, expecting humans to interpret it. AMRCAIS inverts this: the system interprets data contextually, understanding that a steepening yield curve means something entirely different during stagflation vs a disinflationary boom.

### Key Innovation

The system's core differentiator is its **regime-conditional interpretation layer**:

1. **Multi-classifier ensemble** — 4 independent classifiers (HMM, Random Forest, Correlation Clustering, Volatility Detection) vote on market regime
2. **Dynamic signal adaptation** — Every analytical module adjusts its interpretation based on the detected regime
3. **Regime disagreement signal** — When classifiers disagree (Disagreement Index >0.6), this historically precedes major market transitions, transforming model uncertainty from a weakness into a tradeable insight
4. **Self-improving memory** — Every regime transition observed makes the system smarter (Bloomberg is stateless)

### Strategic Position

AMRCAIS does not compete with Bloomberg on data breadth (50M instruments, tick-by-tick feeds). It competes on **intelligence** — the ability to contextualize, predict, and recommend based on regime awareness. By Phase 3, AMRCAIS will do things Bloomberg literally cannot do. By Phase 5, the compounding knowledge base creates a moat that grows with every market event.

---

## 2. Problem Statement

### 2.1 Current Market Analysis Limitations

Existing financial analysis tools suffer from three critical shortcomings:

- **Static Signal Interpretation:** Traditional models assume that specific signals (e.g., yield curve steepening, CPI surprises) have consistent implications regardless of market context. In reality, the same signal can be bullish in one regime and bearish in another.

- **Siloed Analytics:** Macro event trackers, factor models, volatility surfaces, and correlation monitors operate independently. Traders must manually synthesize insights across tools, creating cognitive overhead and missing regime-dependent relationships.

- **Model Herding Risk:** As machine learning proliferates in finance, similar models trained on similar data generate correlated trading signals, amplifying market instability during regime transitions when models collectively fail.

### 2.2 The Bloomberg Gap

Bloomberg charges $24,000/seat/year for a data terminal. But:

- $24K/year buys **data**, not **intelligence**
- Every fund still needs analysts ($200K–$500K/year each) to interpret the data
- Those analysts still get regime transitions wrong (March 2020, Q4 2018, 2022 inflation)
- Bloomberg's terminal is fundamentally **stateless** — it doesn't learn from past regime transitions

AMRCAIS targets the intelligence gap: if it can detect regime transitions 2–5 days earlier with 70%+ accuracy, the value for a $100M fund is ~$7M/year in alpha from regime intelligence alone.

### 2.3 Research Gaps Addressed

- **Dynamic Regime Detection:** Few systems dynamically adapt their analytical components based on detected regimes. Most treat regime classification as informational rather than operational.
- **Cross-Asset Spillover Analysis:** Multi-asset regime identification capturing correlation structure changes remains underexplored.
- **Explainable AI in Finance:** A regime-based framework provides interpretable structure where decisions can be explained as "We're in Regime 3, where these factors historically drive returns."

---

## 3. Solution Overview

### 3.1 System Architecture

AMRCAIS employs a three-layer architecture with a target-state platform design:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         AMRCAIS Platform                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ WebSocket   │  │ REST API    │  │ Python SDK  │  ← Interfaces   │
│  │ Live Feed   │  │ (FastAPI)   │  │ (pip)       │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│  ┌──────┴────────────────┴────────────────┴──────┐                 │
│  │              Event Bus (Redis/Kafka)           │                 │
│  └──────┬────────────────┬────────────────┬──────┘                 │
│         │                │                │                         │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐                │
│  │  Regime     │  │  Prediction │  │  Portfolio  │                 │
│  │  Engine     │  │  Engine     │  │  Engine     │  ← Core        │
│  │  (4+N clf)  │  │  (transition│  │  (optimizer │                 │
│  │             │  │   + return) │  │   + risk)   │                 │
│  └──────┬──────┘  └──────┴──────┘  └──────┬──────┘                │
│         │                │                │                         │
│  ┌──────┴────────────────┴────────────────┴──────┐                 │
│  │           Knowledge Base (DuckDB)              │                 │
│  │  Regime transitions, anomalies, signals,       │                 │
│  │  model versions, backtest results              │  ← Memory      │
│  └──────┬────────────────────────────────────────┘                 │
│         │                                                           │
│  ┌──────┴──────────────────────────────────────────┐               │
│  │        Data Pipeline                             │               │
│  │  FRED │ yfinance │ Polygon │ CBOE │ AQR │ Alt   │  ← Sources   │
│  └──────────────────────────────────────────────────┘               │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  Dashboard (Next.js) │ Alerts │ Paper Trading │ Research Reports    │
└──────────────────────────────────────────────────────────────────────┘
```

| Layer                              | Description                                                                                                                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Layer 1:** Regime Classification | 4 classifiers (HMM, Random Forest, Correlation Clustering, Volatility Detection) → ensemble voter → (regime, confidence, disagreement)                                           |
| **Layer 2:** Signal Interpretation | 5 analytical modules with regime-adaptive parameters: Macro Event Tracker, Yield Curve Analyzer, Options Surface Monitor, Factor Exposure Analyzer, Correlation Anomaly Detector |
| **Layer 3:** Meta-Learning         | Performance tracking, regime classification accuracy monitoring, disagreement detection, recalibration triggers                                                                  |

### 3.2 Core Market Regimes

The system classifies markets into four primary regimes based on historical analysis (2010–2026):

| Regime                      | Characteristics                                     | Historical Examples  | Signal Interpretation                                      |
| --------------------------- | --------------------------------------------------- | -------------------- | ---------------------------------------------------------- |
| **1. Risk-On Growth**       | Equities ↑, Bonds ↓, VIX <20, positive correlations | 2017-2019, 2023-2024 | Strong NFP = bullish; curve steepening = bullish           |
| **2. Risk-Off Crisis**      | Everything sells, correlations spike to +1, VIX >30 | March 2020, Q4 2008  | Traditional signals unreliable; focus on volatility regime |
| **3. Stagflation**          | Commodities ↑, Equities flat, Rates rising          | 2022, 1970s          | Strong CPI = bearish; curve steepening = bearish           |
| **4. Disinflationary Boom** | Equities and Bonds both ↑, falling rates            | Late 2023, 2010-2014 | Quality/growth factors outperform; rate cuts = bullish     |

---

## 4. Features & Functionality

### 4.1 Analytical Modules (Layer 2)

**Module 1: Macro Event Impact Tracker**

- Monitors scheduled economic releases (NFP, CPI, FOMC) and measures market reactions across equities, FX, rates, and volatility
- Regime Adaptation: In Risk-On Growth, strong NFP triggers equity rallies. In Stagflation, strong NFP is bearish (signals tighter Fed policy)
- Data Sources: FRED API (macro data), Alpha Vantage/Polygon.io (intraday price data)

**Module 2: Yield Curve Deformation Analyzer**

- Simulates parallel shifts, steepeners, flatteners, and butterfly trades. Calculates duration, DV01, and convexity
- Regime Adaptation: Steepening = bullish in Risk-On Growth, bearish in Stagflation
- Target Enhancement: Nelson-Siegel-Svensson parametric model, PCA on yield curve changes (first 3 components ≈ 99.5% of variance)
- Data Sources: Treasury.gov, FRED API

**Module 3: Options Surface Monitor**

- Cleans option chains, interpolates implied volatility, generates smooth volatility surfaces, detects arbitrage opportunities
- Regime Adaptation: In Risk-Off Crisis, put skew steepens dramatically; system raises thresholds for "normal" skew
- Target Enhancement: SABR model calibration, vol-of-vol (VVIX), real options data from CBOE/Polygon.io
- Data Sources: yfinance, CBOE DataShop, Polygon.io options

**Module 4: Factor Exposure Analyzer**

- PCA and rolling regressions to estimate exposures to value, momentum, quality, and volatility factors
- Regime Adaptation: Value outperforms in late-cycle/stagflation; growth outperforms in disinflationary boom
- Target Enhancement: Fama-French/AQR factor data, rolling 60-day OLS, factor crowding detection
- Data Sources: yfinance, Kenneth French Data Library, AQR factor datasets

**Module 5: Cross-Asset Correlation Anomaly Detector**

- Tracks rolling correlations across equities, bonds, gold, USD, oil, and VIX. Flags deviations from regime-specific baselines
- Regime Adaptation: In Risk-On Growth, negative equity-bond correlation is normal; positive → signals regime shift
- Target Enhancement: Granger causality testing, Diebold-Yilmaz spillover index, dynamic contagion network
- Data Sources: yfinance, FRED

### 4.2 The Killer Feature: Regime Disagreement Alert

**Unique Value Proposition:** Most systems provide a single regime classification. AMRCAIS runs multiple classifiers simultaneously and monitors when they disagree.

- **Hidden Markov Model:** Statistical approach based on asset return distributions (GaussianHMM, 4-state)
- **Random Forest Classifier:** 200-tree RF with 5-fold cross-validation on labeled historical regimes
- **Correlation-Based Clustering:** KMeans/Spectral clustering on cross-asset correlation structure
- **Volatility Regime Detector:** VIX levels, realized volatility, term structure analysis

**Actionable Insight:** When Disagreement Index >0.6, the system flags "regime uncertainty" — historically a leading indicator of major market transitions (precedes transitions by 1–4 weeks in ≥70% of cases).

### 4.3 Regime Transition Probability Model (Phase 2)

Move beyond "what regime are we in" to "what's the next regime":

```
Current:  "We are in Risk-On Growth (85% confidence)"
Phase 2:  "We are in Risk-On Growth (85% confidence).
           Probability of transition to Risk-Off in next 30 days: 34% ↑
           Leading indicators: equity-bond correlation rising, VIX term
           structure flattening, HMM transition probability spiking."
```

Leading indicator candidates (ranked by historical predictive power):

1. Disagreement Index trend (rising disagreement → transition)
2. VIX term structure slope (backwardation → risk-off incoming)
3. Credit spread momentum (widening → risk-off)
4. Equity-bond correlation change (decorrelation breaking → regime shift)
5. Yield curve butterfly movement (curvature → regime transition)

### 4.4 Natural Language Regime Narrative (Phase 2)

Auto-generated human-readable market commentary backed by specific module data:

```
"AMRCAIS Daily Briefing — February 13, 2026

The system identifies the current regime as Risk-On Growth with 82%
confidence, down from 89% yesterday. The primary concern is the
equity-bond correlation, which has risen to +0.35 — anomalously high
for this regime (expected: -0.15 to +0.10). Historically, sustained
correlation anomalies of this magnitude have preceded regime transitions
within 12 trading days in 7 of 9 instances.

Recommended positioning shift: Consider reducing equity beta from 1.2x
to 0.9x and adding convexity via VIX call spreads."
```

### 4.5 Regime-Conditional Risk & Portfolio Management (Phase 3)

**Regime-Conditional VaR with Attribution:**

```
Standard VaR:   Portfolio 1-day 99% VaR = -2.3%
AMRCAIS VaR:    Portfolio 1-day 99% VaR:
                  If Risk-On persists (65%):    -1.4%
                  If transition to Risk-Off:    -4.8%   ← THIS is the risk
                  If transition to Stagflation: -3.1%
                Transition probability weighted: -2.3%
```

**Regime-Aware Portfolio Optimizer:**

```
             Risk-On    Transitioning   Risk-Off
SPX          60%   →    40%        →    20%
TLT          15%   →    30%        →    45%
GLD          10%   →    15%        →    25%
Cash         15%   →    15%        →    10%

Rebalance trigger: Regime change OR transition probability > 40%
```

### 4.6 Real-Time Operations (Phase 4)

**WebSocket Infrastructure:**

- Polygon.io/Alpaca WebSocket for live market data
- Redis/Kafka event bus for internal event routing
- 15-minute regime update frequency during market hours
- Server-Sent Events (SSE) for live dashboard updates

**Alert Engine:**

```
Alert Types:
  🔴 REGIME CHANGE: Risk-On → Risk-Off detected (confidence: 78%)
  🟡 TRANSITION WARNING: Disagreement index exceeded 0.6 (current: 0.72)
  🟠 CORRELATION ANOMALY: SPX-TLT correlation spike (+0.45 vs expected -0.15)
  🔵 RECALIBRATION NEEDED: Model accuracy below 70% for 10 consecutive days
  ⚪ MACRO EVENT: CPI surprise +2.3σ — regime interpretation: bearish (stagflation)
```

**Paper Trading:** Alpaca paper trading API, regime-based strategy execution, real-time P&L with regime attribution.

---

## 5. Technical Requirements

### 5.1 Technology Stack

| Component              | Technology                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Language**           | Python 3.10+ (backend), TypeScript (frontend)                                                               |
| **Backend Framework**  | FastAPI 0.109.0 with Uvicorn                                                                                |
| **Frontend Framework** | Next.js 16.1.6 (App Router, Turbopack), React 19                                                            |
| **Data Analysis**      | pandas, numpy, scipy                                                                                        |
| **Machine Learning**   | scikit-learn (Random Forest, KMeans, Spectral), hmmlearn (GaussianHMM), statsmodels                         |
| **Visualization**      | Plotly.js 3.3.1 (15 chart types incl. 3D), TradingView Lightweight Charts 5.1.0                             |
| **Data Grid**          | TanStack React Table                                                                                        |
| **State Management**   | TanStack React Query 5.90.21, URL state persistence                                                         |
| **Data Sources**       | FRED API (free), yfinance, Alpha Vantage, Polygon.io (freemium), CBOE DataShop, Kenneth French Data Library |
| **Database**           | SQLite/DuckDB (local), PostgreSQL (production target)                                                       |
| **Containerization**   | Docker Compose (API + Dashboard containers)                                                                 |
| **Security**           | OWASP hardening, CSRF protection, rate limiting, API key auth                                               |
| **Testing**            | pytest (backend), Vitest 4.0.18 (frontend)                                                                  |

### 5.2 Data Requirements

- **Historical Coverage:** Minimum 15 years (2010–present) to capture multiple regime transitions including 2020 COVID crisis, 2022 inflation shock, 2008 financial crisis (if extending back)
- **Asset Classes:** SPX (equities), TLT/IEF (bonds), GLD (gold), DXY (USD), WTI (oil), VIX (volatility)
- **Frequency:** Daily close prices for regime detection; intraday for event impact analysis
- **Macro Data:** NFP, CPI, Core PCE, FOMC meeting dates and decisions, PMI releases
- **Future Sources (Phase 2+):** CBOE DataShop (options), Kenneth French Data Library (factors), AQR factor datasets, VVIX, CDX indices, TIPS breakevens, Fed Funds futures, MOVE Index, SKEW Index

### 5.3 Performance Requirements

| Requirement                   | Target                               |
| ----------------------------- | ------------------------------------ |
| Regime Classification Latency | < 5 seconds for daily update         |
| Backtesting Speed             | 15 years of data in < 2 minutes      |
| API Response Time             | < 500ms for all endpoints            |
| Dashboard Rendering           | Interactive charts < 3 seconds       |
| Real-Time Updates (Phase 4)   | Every 15 minutes during market hours |

---

## 6. Implementation Roadmap

### Phase 0: Foundation — COMPLETE ✅

**21,800 LOC | 588 tests | 22 API endpoints | 18 chart components**

| Deliverable                                                         | Status |
| ------------------------------------------------------------------- | ------ |
| Data pipeline (FRED, yfinance, Alpha Vantage with fallback chain)   | ✅     |
| 4 regime classifiers + ensemble voter                               | ✅     |
| 5 analytical modules with regime-conditional parameters             | ✅     |
| Meta-learning layer (performance tracking, accuracy, disagreement)  | ✅     |
| FastAPI backend (22 endpoints, OWASP security, CSRF, rate limiting) | ✅     |
| Next.js dashboard (7 pages, 18 charts, 3D surfaces, URL state)      | ✅     |
| Docker Compose deployment                                           | ✅     |
| Backend tests (501) + Frontend tests (87)                           | ✅     |

### Phase 1: Foundation Hardening — Weeks 1–6

**Goal:** Zero stubs, zero placeholders. Everything works with real data.

| Item                                                                            | Timeline  |
| ------------------------------------------------------------------------------- | --------- |
| Recalibration engine (walk-forward retrain, shadow mode, rollback, persistence) | Weeks 1–2 |
| Options data integration (CBOE/Polygon.io, SABR calibration, VVIX)              | Weeks 2–3 |
| Factor model integration (Fama-French/AQR, rolling OLS, crowding detection)     | Weeks 3–4 |
| Signal history persistence (SQLite/DuckDB, queryable across all modules)        | Week 4    |
| Nelson-Siegel yield curve (NSS model, level/slope/curvature factors)            | Weeks 4–5 |
| Volatility classifier upgrade (GARCH, VIX futures term structure, ML-based)     | Weeks 5–6 |

### Phase 2: Intelligence Expansion — Weeks 7–14

**Goal:** Capabilities Bloomberg fundamentally cannot do.

| Item                                | Description                                                                  |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| Regime transition probability model | HMM transition probs + logistic regression on leading indicators             |
| Cross-asset contagion network       | Granger causality, Diebold-Yilmaz spillover, regime-conditional topology     |
| Natural language regime narrative   | Template → LLM-enhanced daily briefings backed by module data                |
| Multi-timeframe regime detection    | Daily/weekly/monthly views, conflicting timeframes = high-conviction signals |
| Macro surprise decay model          | Per-indicator half-lives, cumulative surprise index, stale detection         |

### Phase 3: Prediction Engine — Weeks 15–24

**Goal:** From "what is the regime" to "what will the regime do to asset prices."

| Item                                  | Description                                                        |
| ------------------------------------- | ------------------------------------------------------------------ |
| Regime-conditional return forecasting | Hamilton regime-switching regression, separate α and β per regime  |
| Tail risk attribution                 | Regime-conditional VaR/CVaR with transition scenario decomposition |
| Regime-aware portfolio optimizer      | Mean-variance + Black-Litterman with regime views, txn cost-aware  |
| Anomaly-based alpha signals           | Correlation anomalies → tradeable signals, composite alpha score   |

### Phase 4: Real-Time + Execution — Weeks 25–36

**Goal:** Transform from daily research tool to real-time decision engine.

| Item                          | Description                                                     |
| ----------------------------- | --------------------------------------------------------------- |
| WebSocket data infrastructure | Polygon.io/Alpaca, Redis/Kafka event bus, 15-min regime updates |
| Alert engine                  | Multi-channel (email, Slack, Telegram), configurable thresholds |
| Paper trading integration     | Alpaca paper trading, regime P&L attribution                    |
| API-first + Python SDK        | `pip install amrcais-client`, expanded OpenAPI, gRPC            |

### Phase 5: Network Effects — Weeks 37–52

**Goal:** Defensible advantages that compound over time.

| Item                          | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| Institutional memory          | Indexed regime transitions with outcomes, pattern matching   |
| Multi-user + collaboration    | Role-based access, custom regimes, shared annotations        |
| Alternative data integration  | 7+ new signals (MOVE, SKEW, CDX, copper/gold, HY, TIPS, FFR) |
| Research publication pipeline | Auto-generated case studies and backtest reports             |

---

## 7. Success Metrics & Validation

### 7.1 Quantitative Metrics

| Metric                          | Target                                                        |
| ------------------------------- | ------------------------------------------------------------- |
| Regime Classification Accuracy  | ≥80% vs manually labeled regimes (2010–2026)                  |
| Regime Transition Detection     | Disagreement >0.6 precedes 70%+ of transitions, 1–4 week lead |
| Signal Improvement              | ≥15% higher Sharpe ratio vs static models in backtest         |
| False Positive Rate             | ≤20% uncertainty alerts during stable periods                 |
| Return Forecast R² (Phase 3)    | Positive out-of-sample R² for regime-conditional models       |
| Portfolio Alpha (Phase 3)       | Regime-aware optimizer outperforms static 60/40               |
| Transition Prediction (Phase 2) | ≥70% accuracy with 2–5 day lead time                          |
| Test Coverage                   | ≥80% backend and frontend                                     |

### 7.2 Bloomberg Comparison Matrix (Phase 3+ Target)

| Capability                  | Bloomberg         | AMRCAIS                                                   |
| --------------------------- | ----------------- | --------------------------------------------------------- |
| "What regime are we in?"    | Doesn't exist     | 4-classifier ensemble with confidence + disagreement      |
| "What will happen next?"    | User guesses      | Transition probability model with leading indicators      |
| "What does this data mean?" | Raw numbers       | Regime-conditional interpretation                         |
| "How should I position?"    | Generic analytics | Regime-aware optimal allocation with transition scenarios |
| "What's my real risk?"      | Static VaR        | Regime-conditional VaR with transition decomposition      |
| "Is the model still right?" | N/A               | 5 recalibration triggers with auto-retrain                |
| "Has this happened before?" | Historical charts | Pattern-matched institutional memory                      |
| System learns from history  | No (stateless)    | Yes (every transition improves the model)                 |

### 7.3 Qualitative Validation

- **Expert Review:** 5–10 finance professionals confirm regime classifications align with market memory
- **Interpretability:** Explanations understandable to non-technical users
- **Walk-Forward Only:** All backtests use walk-forward validation — never use future data

---

## 8. Risk Analysis & Mitigation

| Risk                                  | Impact                                                                    | Mitigation                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Overfitting to Historical Regimes** | Novel regimes (AI-driven markets, digital currency dominance) may not fit | Walk-forward validation; out-of-sample monitoring; flag distribution drift                                    |
| **Data Quality Issues**               | Free APIs have gaps, errors, corporate action adjustments                 | Robust validation; cross-reference multiple sources; fallback chain (FRED → yfinance → Alpha Vantage → cache) |
| **Computational Complexity**          | 4+ classifiers × 5 modules may not scale to intraday                      | Vectorized operations, caching, incremental updates; daily sufficient for Phase 0–2                           |
| **Scope Creep**                       | Feature additions delay core deliverable                                  | Phase-gated roadmap with exit criteria; no new phase until previous exits cleanly                             |
| **Regime Novelty**                    | Unprecedented regime type not in training data                            | Monitor classifier confidence; flag universal uncertainty; build regime discovery mechanism                   |
| **API Rate Limits**                   | Throttled during market hours on free tier                                | Rate-limited fetcher with backoff; local caching ≤7 days; batch requests                                      |

---

## 9. Current Implementation Status

### 9.1 Codebase Metrics (February 2026)

| Metric                  | Value                                   |
| ----------------------- | --------------------------------------- |
| Total Lines of Code     | ~21,800                                 |
| Python Backend LOC      | 11,589                                  |
| TypeScript Frontend LOC | 4,142                                   |
| Test Code LOC           | 6,050                                   |
| Backend Tests           | 501 (all passing)                       |
| Frontend Tests          | 87 (all passing)                        |
| API Endpoints           | 22                                      |
| Dashboard Charts        | 18 (incl. 3D yield curve + vol surface) |
| Dashboard Pages         | 7                                       |
| Git Commits             | 13                                      |

### 9.2 Module Readiness

| Module                       | Status        | Notes                                                  |
| ---------------------------- | ------------- | ------------------------------------------------------ |
| HMM Classifier               | 🟢 Production | GaussianHMM 4-state, walk-forward fit                  |
| ML Classifier                | 🟢 Production | 200-tree RF, 5-fold CV                                 |
| Correlation Classifier       | 🟢 Production | KMeans/Spectral clustering                             |
| Volatility Classifier        | 🟡 Functional | Rule-based; needs GARCH + VIX term structure (Phase 1) |
| Ensemble Voter               | 🟢 Production | Weighted voting with confidence calibration            |
| Macro Event Tracker          | 🟢 Production | 6 indicators with regime-conditional weights           |
| Yield Curve Analyzer         | 🟢 Production | Cubic spline; Nelson-Siegel upgrade planned (Phase 1)  |
| Options Surface Monitor      | 🟡 Functional | VIX proxy; real options + SABR planned (Phase 1)       |
| Factor Exposure Analyzer     | 🟡 Functional | PCA-based; Fama-French planned (Phase 1)               |
| Correlation Anomaly Detector | 🟢 Production | Rolling correlations with regime baselines             |
| Performance Tracker          | 🟢 Production | Accuracy tracking + recalibration triggers             |
| Meta-Learner                 | 🟡 Functional | `execute_recalibration()` stub (Phase 1 priority)      |
| Data Pipeline                | 🟢 Production | FRED + yfinance + Alpha Vantage with fallback          |
| FastAPI Backend              | 🟢 Production | 22 endpoints, full security stack                      |
| Next.js Dashboard            | 🟢 Production | 7 pages, 18 charts, URL state, 3D surfaces             |

---

## 10. References

- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." _Econometrica_, 57(2), 357-384.
- Ang, A., & Bekaert, G. (2002). "Regime Switches in Interest Rates." _Journal of Business & Economic Statistics_, 20(2), 163-182.
- Guidolin, M., & Timmermann, A. (2007). "Asset Allocation under Multivariate Regime Switching." _Journal of Economic Dynamics and Control_, 31(11), 3503-3544.
- Diebold, F. X., & Yilmaz, K. (2012). "Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers." _International Journal of Forecasting_, 28(1), 57-66.
- Recent research (2024–2026) on AI in financial markets, ML model risk, and correlation regime changes.

---

## Appendix A: Financial Disclaimer

This software is provided for **educational and research purposes only**. It does NOT constitute financial advice, investment advice, trading advice, or any other sort of advice. The authors and contributors are not financial advisors. Do not make any financial decisions based solely on the output of this software.

Past performance is not indicative of future results. Historical backtests and simulations may not reflect actual market conditions. Markets can remain irrational longer than you can remain solvent.

---

## Appendix B: Document History

| Version | Date              | Changes                                                                                                                                                                                                                                                                                        |
| ------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | February 4, 2026  | Initial release — 4-phase 12-week MVP roadmap                                                                                                                                                                                                                                                  |
| 2.0     | February 13, 2026 | Major update: 5-phase strategic roadmap (52 weeks), updated tech stack (FastAPI + Next.js), current implementation status, Bloomberg comparison matrix, target platform architecture, prediction engine and portfolio optimization specs, real-time & execution plan, network effects strategy |
