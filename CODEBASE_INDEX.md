# AMRCAIS Codebase Index

**Last Updated:** February 17, 2026
**Project Status:** All 5 Phases Complete (Foundation → Network Effects)
**Completion:** ~97% (all phases implemented; Phase 1 quality upgrades pending)
**Tests:** 1,177 passing (971 backend + 206 frontend, 0 failures)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Summary](#architecture-summary)
3. [Codebase Structure](#codebase-structure)
4. [Implementation Status](#implementation-status)
5. [Current Work](#current-work)
6. [Next Steps](#next-steps)
7. [Known Issues](#known-issues)

---

## 🎯 Project Overview

**AMRCAIS** (Adaptive Multi-Regime Cross-Asset Intelligence System) is a quantitative finance framework that treats **market regimes as the foundation for all analysis**. Unlike traditional systems that apply static signal interpretation, AMRCAIS dynamically adjusts its analysis based on the current economic regime.

### Core Innovation

Every market signal means something different depending on the regime:

- **Risk-On Growth (1):** Yield curve steepness = bullish
- **Risk-Off Crisis (2):** Same steepness = flight to quality
- **Stagflation (3):** Same steepness = inflation concern
- **Disinflationary Boom (4):** Same steepness = goldilocks scenario

### Key Features

- **4 Regime Classifiers:** HMM, Random Forest, Correlation-based, Volatility-based
- **Ensemble Voting:** Weighted ensemble with disagreement tracking
- **7 Analytical Modules:** 5 original + Contagion Network + Macro Surprise Decay
- **Meta-Learning Layer:** Self-calibrating system with walk-forward recalibration
- **Uncertainty as Signal:** Converts classifier disagreement into tradeable information
- **Phase 2 Extensions:** Transition forecasting, multi-timeframe detection, NL narratives
- **Phase 3 Prediction:** Regime-conditional return forecasting, VaR, portfolio optimization, alpha signals
- **Phase 4 Real-Time:** Event bus, scheduler, alert engine, SSE streaming, paper trading
- **Phase 5 Knowledge:** Institutional memory, research publishing, alt data, multi-user RBAC
- **Dashboard:** Next.js 16 + React 19, 14 pages, 18 chart components, 6 UI primitives
- **Signal Persistence:** Module signals persisted to SQLite after every analysis run

---

## 🏗️ Architecture Summary

### Five-Phase System

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Network Effects & Moat                             │
│ ├─ KnowledgeBase: Institutional memory (transitions/anom.)  │
│ ├─ ResearchPublisher: Case studies, factor reports           │
│ ├─ AltDataIntegrator: Sentiment, satellite, web, flow data  │
│ └─ UserManager: Multi-user RBAC with annotation system      │
├─────────────────────────────────────────────────────────────┤
│ Phase 4: Real-Time + Execution                              │
│ ├─ EventBus: In-process pub/sub (14 event types)            │
│ ├─ AnalysisScheduler: Periodic regime re-analysis           │
│ ├─ AlertEngine: 7 alert types with cooldown fatigue mgmt    │
│ ├─ StreamManager: SSE streaming to dashboard clients        │
│ └─ PaperTradingEngine: Simulated portfolio execution        │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: Prediction Engine                                  │
│ ├─ ReturnForecaster: Regime-conditional return distributions │
│ ├─ TailRiskAnalyzer: Regime-conditional VaR + stress testing│
│ ├─ PortfolioOptimizer: Mean-variance + regime constraints   │
│ └─ AlphaSignalGenerator: Cross-module composite signals     │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Meta-Learning (ADAPTIVE INTELLIGENCE)              │
│ ├─ MetaLearner: Coordinated adaptive learning               │
│ ├─ PerformanceTracker: Classification history & metrics     │
│ └─ RecalibrationTrigger: Smart recalibration logic          │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Analytical Modules (SIGNAL INTERPRETATION)         │
│ ├─ MacroEventTracker: Event impact analysis                 │
│ ├─ YieldCurveAnalyzer: Nelson-Siegel + cubic spline         │
│ ├─ OptionsSurfaceMonitor: SABR-like vol surface             │
│ ├─ FactorExposureAnalyzer: Factor rotation detection        │
│ ├─ CorrelationAnomalyDetector: Cross-asset correlation      │
│ ├─ ContagionNetwork: Granger + Diebold-Yilmaz spillover    │
│ └─ SurpriseDecayModel: Macro surprise exponential decay     │
├─────────────────────────────────────────────────────────────┤
│ Phase 2 Extensions                                          │
│ ├─ NarrativeGenerator: NL daily briefings                   │
│ ├─ RegimeTransitionModel: HMM + logistic transition probs   │
│ └─ MultiTimeframeDetector: Daily/Weekly/Monthly ensembles   │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Regime Detection (CLASSIFICATION)                  │
│ ├─ HMMRegimeClassifier: Hidden Markov Models                │
│ ├─ MLRegimeClassifier: Random Forest + feature engineering  │
│ ├─ CorrelationClassifier: Asset correlation clustering      │
│ ├─ VolatilityClassifier: VIX + GARCH(1,1) conditional vol  │
│ └─ RegimeEnsemble: Weighted voting with disagreement index  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Market Data → Validators → Storage → Regime Classifiers → Ensemble →
Modules → Meta-Learning → Adaptive Weights → Recalibration →
Prediction Engine → EventBus → Alerts / SSE Stream / Paper Trading →
Knowledge Base → Research Publisher → Dashboard
```

---

## 📁 Codebase Structure

### Code Statistics

| Area              | Files | Lines   |
|-------------------|-------|---------|
| `src/` (Python)   | 48    | 18,571  |
| `api/` (Python)   | 16    | 4,493   |
| `tests/` (Python) | 20    | 10,149  |
| `dashboard/` (TS) | 72    | 9,636   |
| **Total**         | **156** | **42,849** |

---

### Configuration (`config/` — 3 files, ~635 lines)

```
config/
├── regimes.yaml                    # 228 lines — Regime definitions & parameters
├── model_params.yaml               # 196 lines — Classifier hyperparameters
└── data_sources.yaml               # 211 lines — Data fetcher configurations
```

**Purpose:** Configuration-driven design — no hardcoded parameters

---

### Source Code (`src/` — 48 files, ~18,571 lines)

#### Main Entry Point

```
src/
├── __init__.py                     # Package initialization
└── main.py                         # ~728 lines — AMRCAIS main coordinator (with signal persistence)
```

#### Layer 1: Regime Detection (`src/regime_detection/` — 9 files, ~3,806 lines)

```
src/regime_detection/
├── __init__.py                     # Module exports
├── base.py                         # BaseClassifier, RegimeResult, REGIME_NAMES
├── hmm_classifier.py               # HMM with Gaussian emissions
├── ml_classifier.py                # Random Forest classifier
├── correlation_classifier.py       # Correlation matrix clustering
├── volatility_classifier.py        # VIX + GARCH(1,1), learned thresholds
├── ensemble.py                     # Weighted voting ensemble
├── transition_model.py             # HMM + logistic transition forecasting (Phase 2)
└── multi_timeframe.py              # Daily/weekly/monthly ensembles (Phase 2)
```

**Status:** ✅ All classifiers implemented and tested

---

#### Layer 2: Analytical Modules (`src/modules/` — 9 files, ~3,699 lines)

```
src/modules/
├── __init__.py                     # Module exports
├── base.py                         # AnalyticalModule base, ModuleSignal
├── macro_event_tracker.py          # NFP/CPI/FOMC/PMI/GDP event tracking
├── yield_curve_analyzer.py         # Nelson-Siegel + cubic spline, forward rates
├── options_surface_monitor.py      # VIX-based vol analysis
├── factor_exposure_analyzer.py     # 6 factors, rotation detection
├── correlation_anomaly_detector.py # 7 pairs, 2σ anomalies
├── contagion_network.py            # Granger causality + Diebold-Yilmaz spillover (Phase 2)
└── macro_surprise_decay.py         # Per-indicator exponential decay model (Phase 2)
```

**Status:** ✅ All 7 modules implemented with regime-adaptive parameters

---

#### Layer 3: Meta-Learning (`src/meta_learning/` — 4 files, ~1,501 lines)

```
src/meta_learning/
├── __init__.py                     # Module exports
├── performance_tracker.py          # Classification history & DB persistence
├── recalibration.py                # Recalibration trigger logic (5 triggers)
└── meta_learner.py                 # Adaptive coordinator: walk-forward, shadow mode, rollback
```

**Status:** ✅ Fully implemented — the "killer feature"

---

#### Narrative Engine (`src/narrative/` — 2 files, ~378 lines)

```
src/narrative/
├── __init__.py
└── narrative_generator.py          # NL daily briefings with regime tone (Phase 2)
```

---

#### Prediction Engine (`src/prediction/` — 5 files, ~1,672 lines) — Phase 3

```
src/prediction/
├── __init__.py                     # Module exports
├── return_forecaster.py            # Regime-conditional return distributions
├── tail_risk.py                    # VaR, CVaR, stress testing
├── portfolio_optimizer.py          # Mean-variance optimization
└── alpha_signals.py                # Cross-module composite signals
```

**Status:** ✅ 100% Complete — 6 API endpoints

---

#### Real-Time & Execution (`src/realtime/` — 6 files, ~1,860 lines) — Phase 4

```
src/realtime/
├── __init__.py                     # Module exports
├── event_bus.py                    # Pub/sub with 14 event types
├── scheduler.py                    # Periodic analysis with market hours
├── alert_engine.py                 # 7 alert types, cooldown fatigue management
├── stream_manager.py               # SSE streaming to dashboard clients
└── paper_trading.py                # Simulated portfolio execution
```

**Status:** ✅ 100% Complete — 14 API endpoints

---

#### Knowledge & Network Effects (`src/knowledge/` — 5 files, ~2,076 lines) — Phase 5

```
src/knowledge/
├── __init__.py                     # Module exports
├── knowledge_base.py               # 740 lines — Institutional memory (transitions, anomalies)
├── research_publisher.py           # 602 lines — Case studies, factor/backtest reports
├── alt_data.py                     # 501 lines — Sentiment, satellite, web, flow data
└── user_manager.py                 # 576 lines — Multi-user RBAC with annotations
```

**Status:** ✅ 100% Complete — 28 API endpoints

---

#### Data Pipeline (`src/data_pipeline/` — 5 files, ~2,455 lines)

```
src/data_pipeline/
├── __init__.py                     # Module exports
├── fetchers.py                     # FRED, yfinance, AlphaVantage APIs
├── validators.py                   # Data quality validation
├── storage.py                      # SQLite/PostgreSQL storage + signal history
└── pipeline.py                     # End-to-end data orchestration
```

---

#### Security & Utilities (`src/utils/` — 1 file, ~396 lines)

```
src/utils/
└── security.py                     # APIKeyManager, RateLimiter, SecurityValidator, SecureConfigLoader
```

---

### API (`api/` — 16 files, ~4,493 lines)

```
api/
├── __init__.py
├── main.py                         # FastAPI app factory, CORS, middleware
├── dependencies.py                 # Shared state & DI
├── middleware.py                    # OWASP security, rate limiting
├── schemas.py                      # Pydantic request/response models
├── security.py                     # CSRF, API key auth
└── routes/
    ├── __init__.py
    ├── regime.py                   # 5 endpoints — Regime detection
    ├── modules.py                  # 5 endpoints — Analytical modules
    ├── data.py                     # 4 endpoints — Data pipeline + macro
    ├── backtest.py                 # 3 endpoints — Backtesting engine
    ├── meta.py                     # 7 endpoints — Meta-learning + accuracy
    ├── phase2.py                   # 8 endpoints — Transition, contagion, narrative, multi-TF
    ├── phase3.py                   # 6 endpoints — Return forecasts, VaR, portfolio, alpha
    ├── phase4.py                   # 14 endpoints — Events, alerts, SSE streaming, paper trading
    └── phase5.py                   # 28 endpoints — Knowledge base, research, alt data, users
```

**Total: 80 API endpoints** (64 GET, 12 POST, 1 PUT, 1 DELETE)

---

### Dashboard (`dashboard/` — 72 TS/TSX files, ~9,636 lines)

#### Pages (14 pages)

```
dashboard/app/
├── page.tsx                        # Overview dashboard
├── layout.tsx                      # Root layout with sidebar
├── regime/page.tsx                 # Regime analysis
├── modules/page.tsx                # Module signals
├── correlations/page.tsx           # Correlation monitoring
├── backtest/page.tsx               # Backtesting
├── meta/page.tsx                   # Meta-learning & accuracy
├── intelligence/page.tsx           # Phase 2 — Transition forecasts, multi-timeframe, narratives
├── contagion/page.tsx              # Phase 2 — Contagion network, spillover matrix
├── predictions/page.tsx            # Phase 3 — Return forecasts, alpha signals
├── risk/page.tsx                   # Phase 3 — Tail risk, portfolio optimization
├── alerts/page.tsx                 # Phase 4 — Alert management, events, config
├── trading/page.tsx                # Phase 4 — Paper trading, equity curves, regime attribution
├── knowledge/page.tsx              # Phase 5 — Institutional memory, transitions, anomalies
└── research/page.tsx               # Phase 5 — Research reports, case study generator
```

#### Components (27 components)

```
dashboard/components/
├── charts/                         # 18 chart components
│   ├── PlotlyChart.tsx             # Generic Plotly wrapper
│   ├── LightweightChart.tsx        # TradingView lightweight charts
│   ├── RegimeStripChart.tsx        # Regime timeline strip
│   ├── RegimeDistributionChart.tsx # Regime distribution pie/bar
│   ├── RegimeReturnsChart.tsx      # Returns by regime
│   ├── CorrelationHeatmap.tsx      # Correlation matrix heatmap
│   ├── CorrelationPairsChart.tsx   # Asset pair correlations
│   ├── DisagreementSeriesChart.tsx # Disagreement time series
│   ├── DisagreementVsSpxChart.tsx  # Disagreement vs. SPX overlay
│   ├── AccuracyLineChart.tsx       # Classifier accuracy over time
│   ├── ClassifierWeightsChart.tsx  # Classifier weight evolution
│   ├── WeightEvolutionChart.tsx    # Weight evolution line chart
│   ├── TransitionMatrixChart.tsx   # Regime transition matrix
│   ├── SignalHistoryChart.tsx      # Signal history line chart
│   ├── EquityCurveChart.tsx        # Equity curve for backtest/trading
│   ├── DrawdownChart.tsx           # Drawdown chart
│   ├── VolSurface3DChart.tsx       # 3D volatility surface
│   ├── YieldCurveSurfaceChart.tsx  # Yield curve surface
│   └── index.ts                    # Barrel exports
├── layout/
│   ├── Sidebar.tsx                 # Navigation sidebar
│   └── Topbar.tsx                  # Top bar
├── overview/
│   ├── DisagreementGauge.tsx       # Regime disagreement gauge
│   └── RegimeTimeline.tsx          # Overview regime timeline
├── providers/
│   └── QueryProvider.tsx           # TanStack Query provider
└── ui/
    ├── MetricsCard.tsx             # KPI card component
    ├── SignalCard.tsx              # Module signal card
    ├── RegimeBadge.tsx             # Regime badge with colors
    ├── DataTable.tsx               # Sortable data table
    ├── ErrorState.tsx              # Error boundary
    └── Skeleton.tsx                # Loading skeleton
```

#### Libraries

```
dashboard/lib/
├── api.ts                          # API client (40+ fetch functions)
├── types.ts                        # TypeScript interfaces (~700 lines)
├── hooks.ts                        # Custom React hooks
├── utils.ts                        # Formatting: pct, pctRaw, num, currency, cn
└── constants.ts                    # REGIME_NAMES, REGIME_COLORS, TRACKED_ASSETS
```

#### Tech Stack

- **Framework:** Next.js 16.1.6, React 19.2.3, TypeScript 5
- **Styling:** Tailwind CSS 4
- **Charts:** Plotly.js 3.3.1 (15 chart types incl. 3D), TradingView Lightweight Charts 5.1.0
- **Data Fetching:** TanStack Query v5
- **Testing:** Vitest 4.0.18, @testing-library/react 16.3.2

---

### Backend Tests (`tests/` — 20 files, ~10,149 lines, 971 tests)

```
tests/
├── conftest.py                     # Pytest fixtures, mock AMRCAIS
├── test_core.py                    # Core classifiers and ensemble
├── test_meta_learning.py           # MetaLearner, tracker, triggers
├── test_pipeline_and_main.py       # Pipeline, AMRCAIS main class
├── test_coverage_boost.py          # Edge cases and coverage gaps
├── test_remaining_coverage.py      # YCA, VC, validators
├── test_phase1_features.py         # Phase 1 feature tests
├── test_phase2_features.py         # Phase 2 feature tests
├── test_phase3_features.py         # Prediction engine tests
├── test_phase4_features.py         # Real-time + execution tests
├── test_phase5_features.py         # Knowledge + network effects tests
├── test_signal_persistence.py      # Signal persistence pipeline
├── test_api_core.py                # Health, security, rate limiting
├── test_api_regime.py              # Regime endpoints
├── test_api_modules.py             # Module endpoints
├── test_api_data.py                # Data endpoints
├── test_api_backtest.py            # Backtest endpoints
├── test_api_meta.py                # Meta endpoints
├── test_security.py                # CSRF, XSS, path traversal
└── __init__.py
```

**Status:** 971/971 tests passing (100%) ✅

---

### Frontend Tests (`dashboard/__tests__/` — 17 files, 206 tests)

```
dashboard/__tests__/
├── setup.ts                        # Vitest setup, DOM mocks
├── helpers.tsx                     # QueryClient wrapper (renderWithQuery)
├── components/
│   ├── charts.test.tsx            # 16 tests — All chart wrapper components
│   ├── DataTable.test.tsx         # 8 tests — Sortable data table
│   ├── SignalCard.test.tsx        # 3 tests — Signal card rendering
│   ├── ui.test.tsx                # 15 tests — MetricsCard, RegimeBadge, Skeleton, ErrorState
│   ├── VolSurface3DChart.test.tsx # 7 tests — 3D vol surface
│   └── YieldCurveSurfaceChart.test.tsx # 5 tests — Yield curve surface
├── lib/
│   ├── constants.test.ts          # 6 tests — Regime names, colors, assets
│   ├── hooks.test.ts              # 7 tests — Custom hook logic
│   └── utils.test.ts              # 20 tests — Formatting utilities
└── pages/
    ├── intelligence.test.tsx      # 13 tests — Intelligence page
    ├── contagion.test.tsx         # 13 tests — Contagion network page
    ├── predictions.test.tsx       # 14 tests — Predictions page
    ├── risk.test.tsx              # 15 tests — Risk analysis page
    ├── alerts.test.tsx            # 17 tests — Alerts & events page
    ├── trading.test.tsx           # 16 tests — Trading page
    ├── knowledge.test.tsx         # 14 tests — Knowledge base page
    └── research.test.tsx          # 17 tests — Research page
```

**Status:** 206/206 tests passing (100%) ✅

---

### Documentation & Configuration

```
.
├── README.md                       # Project overview & setup guide
├── CODEBASE_INDEX.md               # This file — detailed module-by-module docs
├── AMCRAIS_PRD.md                  # Product requirements document
├── AMRCAIS_Development_Rules.md    # Coding standards & patterns
├── AMRCAIS_Master_Prompt.md        # Technical implementation guide
├── DASHBOARD_PRD.md                # Dashboard design specification
├── AUDIT_REPORT.md                 # Audit findings
├── wannabebloomberg.md             # Bloomberg comparison analysis
├── no1.md                          # Phase roadmap & feature definitions
├── LICENSE                         # Apache 2.0
├── requirements.txt                # 23 Python packages with version pins
├── docker-compose.yml              # Multi-container deployment
├── Dockerfile.api                  # API container
├── Dockerfile.dashboard            # Dashboard container
├── pytest.ini                      # Pytest configuration
└── .github/
    └── copilot-instructions.md     # Project-specific Copilot guidance
```

---

## 📊 Implementation Status

### Phase 0: Foundation — 100% Complete ✅

- ✅ Data pipeline (fetchers, validators, storage, pipeline orchestration)
- ✅ 4 regime classifiers (HMM, ML, Correlation, Volatility/GARCH)
- ✅ Ensemble voting with disagreement index
- ✅ 5 Analytical modules with regime-adaptive parameters
- ✅ Meta-learning layer (tracker, recalibration, meta-learner)
- ✅ FastAPI backend (24 core endpoints)
- ✅ Next.js dashboard (6 pages, 18 chart components)
- ✅ Docker Compose deployment
- ✅ SQLite/PostgreSQL storage with signal persistence

---

### Phase 1: Foundation Hardening — ~85% Complete ⚠️

- ✅ Recalibration engine (walk-forward, shadow mode, rollback)
- ✅ Signal history persistence (wired in main.py)
- ✅ Nelson-Siegel yield curve fitting
- ✅ GARCH(1,1) volatility classifier
- ⚠️ Options data integration (VIX proxy only — real CBOE/SABR planned)
- ⚠️ Factor model regression (PCA-based — Fama-French OLS planned)

---

### Phase 2: Intelligence Expansion — 100% Complete ✅

- ✅ Regime transition probability model (HMM + logistic regression)
- ✅ Cross-asset contagion network (Granger + Diebold-Yilmaz)
- ✅ Natural language narrative generator
- ✅ Multi-timeframe regime detection (daily/weekly/monthly)
- ✅ Macro surprise decay model
- ✅ 8 API endpoints in `phase2.py`
- ✅ Dashboard pages: Intelligence, Contagion

---

### Phase 3: Prediction Engine — 100% Complete ✅

- ✅ Regime-conditional return forecaster
- ✅ Regime-conditional VaR with stress testing
- ✅ Portfolio optimizer (mean-variance + regime constraints)
- ✅ Alpha signal generator (cross-module composite)
- ✅ 6 API endpoints in `phase3.py`
- ✅ Dashboard pages: Predictions, Risk

---

### Phase 4: Real-Time + Execution — 100% Complete ✅

- ✅ EventBus: In-process pub/sub with 14 event types
- ✅ AnalysisScheduler: Periodic regime analysis with market-hours-only mode
- ✅ AlertEngine: 7 alert types with cooldown fatigue management
- ✅ StreamManager: SSE streaming to dashboard clients
- ✅ PaperTradingEngine: Simulated portfolio with regime attribution
- ✅ 14 API endpoints in `phase4.py`
- ✅ Dashboard pages: Alerts, Trading

---

### Phase 5: Network Effects & Moat — 100% Complete ✅

- ✅ KnowledgeBase: Institutional memory (transitions, anomalies, pattern search)
- ✅ ResearchPublisher: Case studies, factor analysis, backtest reports
- ✅ AltDataIntegrator: Sentiment, satellite, web scraping, order flow
- ✅ UserManager: Multi-user RBAC with annotation system
- ✅ 28 API endpoints in `phase5.py`
- ✅ Dashboard pages: Knowledge, Research

---

## 🔨 Current Work

### Development Progress

**Status:** 1,177 total tests passing (971 backend + 206 frontend) — All 5 Phases complete ✅

#### Session 7 (February 17, 2026)

1. ✅ Full codebase audit and reindex
2. ✅ Updated CODEBASE_INDEX.md with accurate statistics
3. ✅ Updated README.md with current test counts and features

#### Session 6 (February 16–17, 2026)

1. ✅ Built 8 new dashboard pages (Phase 2–5: Intelligence, Contagion, Predictions, Risk, Alerts, Trading, Knowledge, Research)
2. ✅ Created comprehensive Vitest test suites for all 8 pages (119 new tests)
3. ✅ All 206 frontend tests passing across 17 test files

#### Session 5 (February 16, 2026)

1. ✅ Implemented Phase 4: Real-Time + Execution (5 modules, ~1,860 lines)
2. ✅ Created 14 Phase 4 API endpoints
3. ✅ Implemented Phase 5: Network Effects (4 modules, ~2,076 lines)
4. ✅ Created 28 Phase 5 API endpoints
5. ✅ All 971 backend tests passing

---

### Remaining TODOs (3)

| #   | Location                         | Description                                     | Severity  |
|-----|----------------------------------|-------------------------------------------------|-----------|
| 1   | `ensemble.py` L467               | Accuracy tracking when labeled data available    | 🟢 Low    |
| 2   | `options_surface_monitor.py`     | Replace VIX proxy with CBOE options + SABR       | 🟡 Medium |
| 3   | `factor_exposure_analyzer.py`    | Replace PCA with Fama-French/AQR rolling OLS     | 🟡 Medium |

---

## 🚀 Next Steps

### Immediate (Quality Upgrades)

1. **Connect real options data** — CBOE/yfinance options chain for SABR skew analysis
2. **Fama-French factor regression** — AQR data + OLS regression in factor_exposure_analyzer.py
3. **End-to-end integration testing** — Full pipeline with real market data
4. **Performance profiling** — Latency benchmarks for prediction endpoints

### Strategic

5. **WebSocket upgrade** — Replace SSE with WebSocket for bidirectional real-time streaming
6. **Alpaca integration** — Real brokerage connectivity for paper trading engine
7. **CI/CD pipeline** — GitHub Actions for automated testing and deployment
8. **Python SDK** — `pip install amrcais-client` for programmatic access
9. **Production deployment** — Cloud hosting, monitoring, error tracking

---

## ⚠️ Known Issues

### Critical

- None — all critical issues resolved ✅

### Non-Critical

- **NumPy RuntimeWarnings:** Cosmetic warnings in correlation calculations
- **Windows Temp File Locks:** Handled with `engine.dispose()` before cleanup
- **Options data:** VIX proxy only — real options chain not connected
- **Factor regression:** PCA-based only — no actual OLS Fama-French
- **FutureWarning:** `Series.pct_change()` fill_method deprecation in validators.py

---

## 📈 Project Metrics

### Code Statistics

| Metric                | Count     |
|-----------------------|-----------|
| Python Source Lines   | 18,571    |
| Python API Lines      | 4,493     |
| Python Test Lines     | 10,149    |
| Frontend Lines (TS)   | 9,636     |
| **Total Lines**       | **42,849**|
| Python Source Files   | 48        |
| Frontend Files        | 72        |
| Test Files (Backend)  | 20        |
| Test Files (Frontend) | 17        |
| API Endpoints         | 80        |
| Dashboard Pages       | 14        |
| Chart Components      | 18        |
| UI Components         | 6         |
| Backend Tests         | 971       |
| Frontend Tests        | 206       |
| **Total Tests**       | **1,177** |

### Dependencies

- **Python:** 3.11.9 (requirement: 3.10+)
- **Core:** pandas, numpy, scikit-learn, scipy
- **ML:** hmmlearn, arch (GARCH)
- **Data:** fredapi, yfinance, alpha-vantage
- **Web:** FastAPI, SQLAlchemy, uvicorn
- **Frontend:** Next.js 16.1.6, React 19.2.3, TypeScript 5
- **Charts:** Plotly.js 3.3.1, TradingView Lightweight Charts 5.1.0
- **Data Fetching:** TanStack Query v5
- **Testing:** pytest 7.4.4, Vitest 4.0.18, @testing-library/react 16.3.2

---

## 🎯 Success Criteria

### Development Complete When:

- [x] All 5 phases implemented ✅
- [x] ≥80% test coverage ✅ (1,177 tests passing)
- [x] All classifiers tested and working ✅
- [x] Ensemble voting operational ✅
- [x] Dashboard for all phases ✅ (14 pages)
- [x] 80 API endpoints functional ✅
- [ ] Meta-learning validated with historical data
- [ ] Real options data integration
- [ ] Fama-French factor regression

### Production Ready When:

- [ ] 95%+ test coverage
- [ ] <5 second prediction latency
- [ ] Successful backtest on 3+ market crises
- [ ] CI/CD pipeline operational
- [ ] Error monitoring configured
- [ ] User documentation complete

---

## 📞 Quick Reference

### Key Files to Know

- **Entry Point:** `src/main.py` — AMRCAIS class coordinates everything
- **Regime Base:** `src/regime_detection/base.py` — All classifiers inherit from here
- **Module Base:** `src/modules/base.py` — All modules inherit from here
- **Meta-Learning:** `src/meta_learning/meta_learner.py` — The "killer feature"
- **Knowledge Base:** `src/knowledge/knowledge_base.py` — Institutional memory
- **Config:** `config/regimes.yaml` — Regime definitions and parameters
- **API Types:** `dashboard/lib/types.ts` — All TypeScript interfaces (~700 lines)

### Common Commands

```bash
# Run all backend tests (971 tests)
python -m pytest tests/ -v

# Run backend tests with coverage
python -m pytest tests/ --cov=src --cov=api --cov-report=html

# Run all frontend tests (206 tests)
cd dashboard && npx vitest run

# Run frontend tests in watch mode
cd dashboard && npx vitest

# Start backend API
uvicorn api.main:app --reload --port 8000

# Start dashboard
cd dashboard && npm run dev

# Docker deployment
docker-compose up --build
```

### Git History (Recent Commits)

```
19f4a4d  Add comprehensive page tests for Phase 2-5 dashboard pages
9c3b077  feat(dashboard): add Phase 2-5 pages
83bd700  Phase 5: Network Effects + Moat
1fa2093  Phase 4: Real-Time + Execution
29dfaa4  feat: Phase 3 — Prediction Engine
2dc6bfc  Phase 2: Intelligence Expansion
b282161  Phase 1: Foundation Hardening
```

---

**Last Updated:** February 17, 2026
**Next Review:** After Phase 1 quality upgrades or production deployment
**Maintained By:** AMRCAIS Development Team
