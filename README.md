# AMRCAIS — Adaptive Multi-Regime Cross-Asset Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TypeScript-5.0+-blue.svg" alt="TypeScript 5.0+">
  <img src="https://img.shields.io/badge/License-GPL--3.0-green.svg" alt="GPL-3.0 License">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Backend%20Tests-501%20Passing-brightgreen.svg" alt="Backend Tests: 501 Passing">
  <img src="https://img.shields.io/badge/Frontend%20Tests-87%20Passing-brightgreen.svg" alt="Frontend Tests: 87 Passing">
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
├── api/                           # FastAPI backend (22 endpoints)
│   ├── main.py                    # App factory, CORS, middleware
│   ├── dependencies.py            # Shared state & DI
│   ├── middleware.py              # OWASP security, rate limiting
│   ├── schemas.py                 # Pydantic request/response models
│   ├── security.py                # CSRF, API key auth
│   └── routes/                    # Endpoint routers
│       ├── regime.py              # Regime detection endpoints
│       ├── modules.py             # Analytical module endpoints
│       ├── data.py                # Data pipeline + macro endpoints
│       ├── backtest.py            # Backtesting engine
│       └── meta.py                # Meta-learning + accuracy + disagreement
├── src/                           # Core ML engine
│   ├── main.py                    # AMRCAIS orchestrator
│   ├── regime_detection/          # 4 classifiers + ensemble
│   │   ├── hmm_classifier.py      # GaussianHMM 4-state
│   │   ├── ml_classifier.py       # 200-tree Random Forest
│   │   ├── correlation_classifier.py  # KMeans/Spectral clustering
│   │   ├── volatility_classifier.py   # VIX + realized vol
│   │   └── ensemble.py            # Weighted voting + disagreement
│   ├── modules/                   # 5 analytical modules
│   │   ├── base.py                # AnalyticalModule ABC
│   │   ├── macro_event_tracker.py
│   │   ├── yield_curve_analyzer.py
│   │   ├── options_surface_monitor.py
│   │   ├── factor_exposure_analyzer.py
│   │   └── correlation_anomaly_detector.py
│   ├── meta_learning/             # Layer 3
│   │   ├── performance_tracker.py # Accuracy tracking
│   │   ├── meta_learner.py        # Recalibration logic
│   │   └── recalibration.py       # Trigger evaluation
│   └── data_pipeline/             # Data fetching & validation
│       ├── fetchers.py            # FRED, yfinance, Alpha Vantage
│       ├── validators.py          # Data quality checks
│       ├── storage.py             # SQLite persistence
│       └── pipeline.py            # Orchestrated data flow
├── dashboard/                     # Next.js 16 frontend
│   ├── app/                       # App Router pages
│   │   ├── page.tsx               # Overview dashboard
│   │   ├── regime/page.tsx        # Regime analysis
│   │   ├── modules/page.tsx       # Module signals
│   │   ├── correlations/page.tsx  # Correlation monitoring
│   │   ├── backtest/page.tsx      # Backtesting
│   │   └── meta/page.tsx          # Meta-learning & accuracy
│   ├── components/                # React components
│   │   ├── charts/                # 18 chart components (incl. 3D)
│   │   ├── layout/                # Navigation & layout
│   │   ├── overview/              # Dashboard cards
│   │   └── ui/                    # Shared UI primitives
│   ├── lib/                       # Utilities
│   │   ├── api.ts                 # API client functions
│   │   ├── hooks.ts               # Custom React hooks
│   │   ├── types.ts               # TypeScript types
│   │   └── utils.ts               # Helper utilities
│   └── __tests__/                 # Vitest test suite
├── config/                        # YAML configuration
│   ├── regimes.yaml               # Regime definitions & weights
│   ├── data_sources.yaml          # API endpoints & keys
│   └── model_params.yaml          # Model hyperparameters
├── tests/                         # Backend test suite (501 tests)
├── docker-compose.yml             # Multi-container deployment
├── Dockerfile.api                 # API container
├── Dockerfile.dashboard           # Dashboard container
└── requirements.txt               # Python dependencies
```

---

## API Endpoints

The FastAPI backend exposes 22 endpoints:

| Category     | Endpoint                          | Description                     |
| ------------ | --------------------------------- | ------------------------------- |
| **Regime**   | `GET /api/regime/current`         | Current regime classification   |
|              | `GET /api/regime/history`         | Regime history with transitions |
|              | `GET /api/regime/ensemble`        | Ensemble classifier details     |
|              | `GET /api/regime/disagreement`    | Disagreement index time series  |
| **Modules**  | `GET /api/modules/macro`          | Macro event impact analysis     |
|              | `GET /api/modules/yield-curve`    | Yield curve analysis            |
|              | `GET /api/modules/options`        | Options surface monitor         |
|              | `GET /api/modules/factors`        | Factor exposure analysis        |
|              | `GET /api/modules/correlations`   | Correlation anomaly detection   |
|              | `GET /api/modules/all`            | All module signals combined     |
| **Data**     | `GET /api/data/assets`            | Asset price data                |
|              | `GET /api/data/macro/{indicator}` | Specific macro indicator data   |
|              | `GET /api/data/status`            | Data pipeline health            |
| **Backtest** | `POST /api/backtest/run`          | Run regime-based backtest       |
|              | `GET /api/backtest/results`       | Retrieve backtest results       |
| **Meta**     | `GET /api/meta/performance`       | System performance metrics      |
|              | `GET /api/meta/accuracy`          | Classifier accuracy over time   |
|              | `GET /api/meta/disagreement`      | Disagreement vs SPX analysis    |
|              | `GET /api/meta/recalibration`     | Recalibration status            |

Full interactive API docs available at `/docs` when the server is running.

---

## Dashboard

The Next.js dashboard provides 7 pages with 18 interactive chart components:

| Page             | Charts                                              | Key Features                          |
| ---------------- | --------------------------------------------------- | ------------------------------------- |
| **Overview**     | Regime gauge, summary cards                         | Current regime at a glance            |
| **Regime**       | Timeline, confidence, ensemble heatmap              | Historical regime analysis            |
| **Modules**      | Signal charts per module                            | Regime-adaptive signal interpretation |
| **Correlations** | Correlation matrix, anomaly scatter, 3D vol surface | Cross-asset monitoring                |
| **Backtest**     | Equity curve, drawdown chart, trade log             | Strategy validation                   |
| **Meta**         | Accuracy line chart, disagreement vs SPX            | System self-assessment                |

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
| **Yield Curve Analyzer**         | Duration, DV01, curve shapes       | Steepening bullish in Growth, bearish in Stagflation |
| **Options Surface Monitor**      | IV surfaces, skew analysis         | Adjusted skew thresholds per volatility regime       |
| **Factor Exposure Analyzer**     | Value, Momentum, Quality factors   | Factor recommendations rotate by regime              |
| **Correlation Anomaly Detector** | Cross-asset correlation monitoring | Regime-specific correlation baselines                |

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

| Phase                                   | Status      | Key Deliverable                                                   |
| --------------------------------------- | ----------- | ----------------------------------------------------------------- |
| **Phase 0:** Foundation                 | ✅ Complete | 21,800 LOC, 588 tests, 22 endpoints, 18 charts, Docker deployment |
| **Phase 1:** Hardening (Wk 1–6)         | 🔜 Next     | Zero stubs, real data everywhere, recalibration engine            |
| **Phase 2:** Intelligence (Wk 7–14)     | 📋 Planned  | Transition prediction, contagion network, narrative generation    |
| **Phase 3:** Prediction (Wk 15–24)      | 📋 Planned  | Return forecasting, tail risk attribution, portfolio optimizer    |
| **Phase 4:** Real-Time (Wk 25–36)       | 📋 Planned  | WebSocket, alerts, paper trading, Python SDK                      |
| **Phase 5:** Network Effects (Wk 37–52) | 📋 Planned  | Institutional memory, multi-user, alternative data                |

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

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please review the [Development Rules](AMRCAIS_Development_Rules.md) for coding standards before submitting PRs.

---

## Contact

For questions or collaboration inquiries, please open an issue or reach out through the repository.

---

**Built for quantitative finance research**
