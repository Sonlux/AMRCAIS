# AMRCAIS — Adaptive Multi-Regime Cross-Asset Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TypeScript-5.0+-blue.svg" alt="TypeScript 5.0+">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 License">
  <img src="https://img.shields.io/badge/Status-All%205%20Phases%20Complete-brightgreen.svg" alt="Status: All 5 Phases Complete">
  <img src="https://img.shields.io/badge/Backend%20Tests-971%20Passing-brightgreen.svg" alt="Backend Tests: 971 Passing">
  <img src="https://img.shields.io/badge/Frontend%20Tests-206%20Passing-brightgreen.svg" alt="Frontend Tests: 206 Passing">
  <img src="https://img.shields.io/badge/API%20Endpoints-80-blue.svg" alt="API Endpoints: 80">
  <img src="https://img.shields.io/badge/Total%20LOC-42.8K-blue.svg" alt="Total LOC: 42.8K">
</p>

**A regime-conditional decision intelligence platform for financial markets.** AMRCAIS integrates regime detection with dynamic signal interpretation across asset classes — because a steepening yield curve means something completely different during stagflation than in a disinflationary boom.

> **Bloomberg shows you what happened. AMRCAIS tells you what it means.**

---

## The Core Innovation

Traditional market analysis tools treat signals as static. AMRCAIS solves this with three innovations:

1. **Regime Detection** — An ensemble of 4 independent classifiers (HMM, Random Forest, Correlation Clustering, Volatility Detection) votes on market regime
2. **Regime-Adaptive Signals** — Every analytical module adjusts interpretation based on regime context. Same macro data release, different implications per regime.
3. **Regime Disagreement Signal** — When classifiers disagree (Disagreement Index >0.6), this historically precedes major market transitions. Model uncertainty becomes a tradeable insight.

### The Four Market Regimes

| Regime                      | Characteristics                       | Examples             |
| --------------------------- | ------------------------------------- | -------------------- |
| **1. Risk-On Growth**       | Equities ↑, Bonds ↓, VIX <20          | 2017-2019, 2023-2024 |
| **2. Risk-Off Crisis**      | Correlations spike to +1, VIX >30     | March 2020, Q4 2008  |
| **3. Stagflation**          | Commodities ↑, Equities flat, Rates ↑ | 2022, 1970s          |
| **4. Disinflationary Boom** | Equities + Bonds both ↑, Rates ↓      | Late 2023, 2010-2014 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: META-LEARNING & ADAPTATION                         │
│ • Tracks regime classification accuracy                     │
│ • Monitors disagreement across classifiers                  │
│ • Triggers recalibration when errors exceed thresholds      │
└─────────────────────────────────────────────────────────────┘
                              ▲
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
└─────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: MARKET REGIME CLASSIFICATION                       │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                │
│ │  HMM   │ │   ML   │ │ Corr   │ │  Vol   │                │
│ │Gaussian│ │ Random │ │Cluster │ │Regime  │                │
│ │ 4-State│ │ Forest │ │        │ │        │                │
│ └────────┘ └────────┘ └────────┘ └────────┘                │
│              [Ensemble Voter]                               │
│      Primary Regime + Confidence + Disagreement             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboard)
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sonlux/AMRCAIS.git
cd AMRCAIS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Set up API keys (optional but recommended)
export FRED_API_KEY="your_fred_api_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

### Run the Backend API

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Run the Dashboard

```bash
cd dashboard
npm install
npm run dev
```

The dashboard will be available at `http://localhost:3000`.

### Docker Deployment

```bash
# Start both API and Dashboard
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:3000
```

### Python Usage

```python
from src.main import AMRCAIS

# Initialize the system
system = AMRCAIS()
system.initialize(lookback_days=365)

# Run analysis
results = system.analyze()

# Access results
print(f"Current Regime: {results['regime']['name']}")
print(f"Confidence: {results['regime']['confidence']:.1%}")
print(f"Disagreement Index: {results['regime']['disagreement']:.2f}")

if results['regime']['transition_warning']:
    print("⚠️ HIGH DISAGREEMENT: Possible regime transition ahead!")
```

---

## Project Structure

```
AMRCAIS/
├── api/                           # FastAPI backend (80 endpoints)
│   ├── main.py                    # App factory, CORS, middleware
│   ├── dependencies.py            # Shared state & DI
│   ├── middleware.py              # OWASP security, rate limiting
│   ├── schemas.py                 # Pydantic request/response models
│   ├── security.py                # CSRF, API key auth
│   └── routes/                    # Endpoint routers
│       ├── regime.py              # Regime detection (5 endpoints)
│       ├── modules.py             # Analytical modules (5 endpoints)
│       ├── data.py                # Data pipeline + macro (4 endpoints)
│       ├── backtest.py            # Backtesting engine (3 endpoints)
│       ├── meta.py                # Meta-learning + accuracy (7 endpoints)
│       ├── phase2.py              # Transition, contagion, narrative (8 endpoints)
│       ├── phase3.py              # Forecasts, VaR, portfolio, alpha (6 endpoints)
│       ├── phase4.py              # Events, alerts, SSE, trading (14 endpoints)
│       └── phase5.py              # Knowledge, research, alt data, users (28 endpoints)
├── src/                           # Core ML engine (~18,500 lines)
│   ├── main.py                    # AMRCAIS orchestrator
│   ├── regime_detection/          # 4 classifiers + ensemble + extensions
│   │   ├── hmm_classifier.py      # GaussianHMM 4-state
│   │   ├── ml_classifier.py       # 200-tree Random Forest
│   │   ├── correlation_classifier.py  # KMeans/Spectral clustering
│   │   ├── volatility_classifier.py   # VIX + GARCH(1,1)
│   │   ├── ensemble.py            # Weighted voting + disagreement
│   │   ├── transition_model.py    # HMM + logistic transition forecasting
│   │   └── multi_timeframe.py     # Daily/weekly/monthly ensembles
│   ├── modules/                   # 7 analytical modules
│   │   ├── macro_event_tracker.py
│   │   ├── yield_curve_analyzer.py    # Nelson-Siegel + cubic spline
│   │   ├── options_surface_monitor.py
│   │   ├── factor_exposure_analyzer.py
│   │   ├── correlation_anomaly_detector.py
│   │   ├── contagion_network.py       # Granger + Diebold-Yilmaz
│   │   └── macro_surprise_decay.py    # Exponential decay model
│   ├── meta_learning/             # Adaptive intelligence layer
│   │   ├── performance_tracker.py # Accuracy tracking
│   │   ├── meta_learner.py        # Walk-forward recalibration
│   │   └── recalibration.py       # 5 trigger conditions
│   ├── prediction/                # Regime-conditional prediction
│   │   ├── return_forecaster.py   # Return distributions
│   │   ├── tail_risk.py           # VaR, CVaR, stress testing
│   │   ├── portfolio_optimizer.py # Mean-variance optimization
│   │   └── alpha_signals.py       # Cross-module composite signals
│   ├── realtime/                  # Real-time + execution
│   │   ├── event_bus.py           # Pub/sub with 14 event types
│   │   ├── scheduler.py           # Periodic analysis
│   │   ├── alert_engine.py        # 7 alert types, cooldown fatigue
│   │   ├── stream_manager.py      # SSE streaming
│   │   └── paper_trading.py       # Simulated portfolio execution
│   ├── knowledge/                 # Institutional memory
│   │   ├── knowledge_base.py      # Transitions, anomalies, patterns
│   │   ├── research_publisher.py  # Case studies, reports
│   │   ├── alt_data.py            # Sentiment, satellite, flow data
│   │   └── user_manager.py        # Multi-user RBAC
│   ├── narrative/                 # NL generation
│   │   └── narrative_generator.py # Daily briefings
│   └── data_pipeline/             # Data fetching & validation
│       ├── fetchers.py            # FRED, yfinance, Alpha Vantage
│       ├── validators.py          # Data quality checks
│       ├── storage.py             # SQLite/PostgreSQL persistence
│       └── pipeline.py            # Orchestrated data flow
├── dashboard/                     # Next.js 16 frontend (~9,600 lines)
│   ├── app/                       # App Router pages (14 pages)
│   │   ├── page.tsx               # Overview dashboard
│   │   ├── regime/page.tsx        # Regime analysis
│   │   ├── modules/page.tsx       # Module signals
│   │   ├── correlations/page.tsx  # Correlation monitoring
│   │   ├── backtest/page.tsx      # Backtesting
│   │   ├── meta/page.tsx          # Meta-learning & accuracy
│   │   ├── intelligence/page.tsx  # Transition forecasts, multi-timeframe
│   │   ├── contagion/page.tsx     # Contagion network, spillover
│   │   ├── predictions/page.tsx   # Return forecasts, alpha signals
│   │   ├── risk/page.tsx          # Tail risk, portfolio optimization
│   │   ├── alerts/page.tsx        # Alert management & events
│   │   ├── trading/page.tsx       # Paper trading & equity curves
│   │   ├── knowledge/page.tsx     # Institutional memory
│   │   └── research/page.tsx      # Research reports & case studies
│   ├── components/                # React components (27 total)
│   │   ├── charts/                # 18 chart components (incl. 3D)
│   │   ├── layout/                # Navigation & layout
│   │   ├── overview/              # Dashboard cards
│   │   └── ui/                    # 6 shared UI primitives
│   ├── lib/                       # Utilities
│   │   ├── api.ts                 # API client (40+ fetch functions)
│   │   ├── hooks.ts               # Custom React hooks
│   │   ├── types.ts               # TypeScript interfaces (~700 lines)
│   │   └── utils.ts               # Formatting: pct, pctRaw, num, currency
│   └── __tests__/                 # Vitest test suite (17 files, 206 tests)
├── config/                        # YAML configuration
│   ├── regimes.yaml               # Regime definitions & weights
│   ├── data_sources.yaml          # API endpoints & keys
│   └── model_params.yaml          # Model hyperparameters
├── tests/                         # Backend test suite (971 tests, 20 files)
├── docker-compose.yml             # Multi-container deployment
├── Dockerfile.api                 # API container
├── Dockerfile.dashboard           # Dashboard container
└── requirements.txt               # Python dependencies
```

---

## API Endpoints

The FastAPI backend exposes **80 endpoints** across 9 route files:

| Category     | Endpoints | Key Routes                                                                                               |
| ------------ | --------- | -------------------------------------------------------------------------------------------------------- |
| **Regime**   | 5         | `/api/regime/current`, `/history`, `/ensemble`, `/disagreement`                                          |
| **Modules**  | 5         | `/api/modules/macro`, `/yield-curve`, `/options`, `/factors`, `/all`                                     |
| **Data**     | 4         | `/api/data/assets`, `/macro/{indicator}`, `/status`                                                      |
| **Backtest** | 3         | `/api/backtest/run`, `/results`                                                                          |
| **Meta**     | 7         | `/api/meta/performance`, `/accuracy`, `/disagreement`, `/recalibration`                                  |
| **Phase 2**  | 8         | `/api/phase2/transition-forecast`, `/contagion/*`, `/narrative`, `/multi-timeframe`, `/surprise-decay/*` |
| **Phase 3**  | 6         | `/api/phase3/return-forecast`, `/tail-risk`, `/portfolio-optimize`, `/alpha-signals`                     |
| **Phase 4**  | 14        | `/api/phase4/events`, `/alerts/*`, `/stream`, `/portfolio/*`, `/trades`, `/rebalance`                    |
| **Phase 5**  | 28        | `/api/phase5/knowledge/*`, `/research/*`, `/alt-data/*`, `/users/*`, `/annotations/*`                    |

Full interactive API docs available at `/docs` when the server is running.

---

## Dashboard

The Next.js dashboard provides **14 pages** with 18 chart components and 6 UI primitives:

| Page             | Key Features                                                        |
| ---------------- | ------------------------------------------------------------------- |
| **Overview**     | Regime gauge, summary cards, disagreement index                     |
| **Regime**       | Timeline, confidence, ensemble heatmap, distribution                |
| **Modules**      | Signal cards per module, regime-adaptive interpretation             |
| **Correlations** | Correlation matrix, anomaly scatter, 3D vol surface                 |
| **Backtest**     | Equity curve, drawdown chart, trade log                             |
| **Meta**         | Accuracy line chart, disagreement vs SPX, weight evolution          |
| **Intelligence** | Transition forecasts, multi-timeframe detection, NL narratives      |
| **Contagion**    | Network density, Granger links, spillover matrix, net spillover     |
| **Predictions**  | Return forecasts per asset, alpha signals, factor contributions     |
| **Risk**         | Tail risk VaR/CVaR, stress scenarios, regime-conditional portfolios |
| **Alerts**       | Alert management, severity filters, event log, config table         |
| **Trading**      | Paper trading positions, equity curve, regime attribution           |
| **Knowledge**    | Institutional memory, transitions, anomalies, pattern search        |
| **Research**     | Research reports, case study generator, regime comparison           |

Chart technologies: Plotly.js 3.3.1 (15 chart types incl. 3D surfaces), TradingView Lightweight Charts 5.1.0 (equity curves, regime timeline).

---

## Configuration

All parameters are in YAML config files — **never hardcode values**:

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
```

---

## Testing

```bash
# Run all backend tests (501 tests)
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov=api --cov-report=html

# Run frontend tests (87 tests)
cd dashboard
npx vitest run

# Run frontend tests in watch mode
npx vitest
```

**Backend Test Suite (501 tests):**

| Category                                | Tests | Coverage                                 |
| --------------------------------------- | ----- | ---------------------------------------- |
| Core system (regime, modules, pipeline) | 29    | Classifiers, ensemble, modules, pipeline |
| API regime endpoints                    | 50+   | All regime routes                        |
| API module endpoints                    | 50+   | All module routes                        |
| API data endpoints                      | 50+   | Data + macro routes                      |
| API backtest endpoints                  | 50+   | Backtest engine                          |
| API meta endpoints                      | 50+   | Accuracy, disagreement, performance      |
| Security & middleware                   | 30+   | OWASP, CSRF, rate limiting               |
| Coverage boost & edge cases             | 100+  | Error paths, edge cases                  |

**Frontend Test Suite (87 tests):**

| Category         | Tests | Coverage                        |
| ---------------- | ----- | ------------------------------- |
| Chart components | 50+   | All 18 chart types              |
| API client       | 15+   | Fetch functions, error handling |
| Utilities        | 20+   | Hooks, helpers, formatters      |

---

## Analytical Modules

| Module                           | Purpose                            | Regime Adaptation                                    |
| -------------------------------- | ---------------------------------- | ---------------------------------------------------- |
| **Macro Event Tracker**          | Monitors NFP, CPI, FOMC impact     | Different event weights per regime                   |
| **Yield Curve Analyzer**         | Nelson-Siegel, DV01, curve shapes  | Steepening bullish in Growth, bearish in Stagflation |
| **Options Surface Monitor**      | IV surfaces, skew analysis         | Adjusted skew thresholds per volatility regime       |
| **Factor Exposure Analyzer**     | Value, Momentum, Quality factors   | Factor recommendations rotate by regime              |
| **Correlation Anomaly Detector** | Cross-asset correlation monitoring | Regime-specific correlation baselines                |
| **Contagion Network**            | Granger causality + spillover      | Regime-dependent transmission channels               |
| **Macro Surprise Decay**         | Per-indicator exponential decay    | Decay rates adjust by regime volatility              |

---

## Data Sources

| Source            | Data                                        | Priority |
| ----------------- | ------------------------------------------- | -------- |
| **FRED API**      | Macroeconomic data (NFP, CPI, yield curves) | Primary  |
| **yfinance**      | Equity & ETF prices (SPX, TLT, GLD, VIX)    | Primary  |
| **Alpha Vantage** | Intraday data, supplementary                | Fallback |

The data pipeline implements automatic fallback: FRED → yfinance → Alpha Vantage → cached data (≤7 days old).

---

## Success Metrics

| Metric                         | Target                                    |
| ------------------------------ | ----------------------------------------- |
| Regime Classification Accuracy | ≥80% vs manual labels                     |
| Transition Detection Lead Time | 1–4 weeks via Disagreement Index          |
| Signal Improvement             | ≥15% higher Sharpe vs static models       |
| False Positive Rate            | ≤20% uncertainty alerts in stable periods |

---

## Project Roadmap

| Phase                              | Status      | Key Deliverable                                                   |
| ---------------------------------- | ----------- | ----------------------------------------------------------------- |
| **Phase 0:** Foundation            | ✅ Complete | Data pipeline, 4 classifiers, 5 modules, 24 endpoints, dashboard  |
| **Phase 1:** Hardening             | ⚠️ ~85%     | Recalibration engine, signal persistence, GARCH, Nelson-Siegel    |
| **Phase 2:** Intelligence          | ✅ Complete | Transition model, contagion network, narrative, multi-timeframe   |
| **Phase 3:** Prediction            | ✅ Complete | Return forecasting, tail risk, portfolio optimizer, alpha signals |
| **Phase 4:** Real-Time + Execution | ✅ Complete | EventBus, alerts, SSE streaming, paper trading                    |
| **Phase 5:** Network Effects       | ✅ Complete | Knowledge base, research publisher, alt data, multi-user RBAC     |

### Remaining Work (Phase 1 Quality Upgrades)

- Replace VIX proxy with real CBOE options data + SABR calibration
- Replace PCA factor analysis with Fama-French/AQR rolling OLS
- End-to-end integration testing with live market data
- CI/CD pipeline, production deployment, Python SDK

See [AMCRAIS_PRD.md](AMCRAIS_PRD.md) for the full product requirements and detailed roadmap.

---

## Documentation

- [Product Requirements (PRD)](AMCRAIS_PRD.md) — Full specifications, roadmap, and Bloomberg comparison
- [Development Rules](AMRCAIS_Development_Rules.md) — Coding standards & best practices
- [Master Prompt](AMRCAIS_Master_Prompt.md) — Technical implementation guide
- [Project Status Report](wannabebloomberg.md) — Current state audit and Bloomberg comparison analysis
- [Codebase Index](CODEBASE_INDEX.md) — Detailed module-by-module documentation

---

## Disclaimer

> **This system is for educational and research purposes only.**
> It does not constitute financial advice. Past performance does not guarantee future results.
> Markets can remain irrational longer than you can remain solvent.

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please review the [Development Rules](AMRCAIS_Development_Rules.md) for coding standards before submitting PRs.

---

## Contact

For questions or collaboration inquiries, please open an issue or reach out through the repository.

---

**Built for quantitative finance research**
