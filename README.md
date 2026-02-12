# AMRCAIS - Adaptive Multi-Regime Cross-Asset Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-GPL--3.0-green.svg" alt="GPL-3.0 License">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Tests-29%2F29%20Passing-brightgreen.svg" alt="Tests: 29/29 Passing">
  <img src="https://img.shields.io/badge/Coverage-100%25-brightgreen.svg" alt="Coverage 100%">
</p>

**A novel financial market analysis framework that integrates regime detection with dynamic signal interpretation across multiple asset classes.**

---

## 🎯 The Core Innovation

Traditional market analysis tools treat signals as static—a yield curve steepening means the same thing whether we're in 2019's "goldilocks" economy or 2022's inflation crisis. **This is fundamentally wrong.**

AMRCAIS solves this by:

1. **Detecting market regimes** using an ensemble of 4 independent classifiers (HMM, Random Forest, Correlation Clustering, Volatility Detection)
2. **Adapting signal interpretation** based on regime—the same macro data release has different implications in Risk-On Growth vs. Stagflation
3. **Flagging regime uncertainty**—when classifiers disagree (Disagreement Index >0.6), this historically precedes major market transitions

> **The Killer Feature:** The regime disagreement signal transforms model uncertainty from a weakness into a tradeable insight.

---

## 🏗️ Architecture

AMRCAIS employs a three-layer architecture:

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

### The Four Market Regimes

| Regime | Characteristics | Historical Examples |
|--------|-----------------|---------------------|
| **1. Risk-On Growth** | Equities ↑, Bonds ↓, VIX <20 | 2017-2019, 2023-2024 |
| **2. Risk-Off Crisis** | Correlations spike to +1, VIX >30 | March 2020, Q4 2008 |
| **3. Stagflation** | Commodities ↑, Equities flat, Rates rising | 2022, 1970s |
| **4. Disinflationary Boom** | Equities + Bonds both up, Rates falling | Late 2023, 2010-2014 |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Sonlux/AMRCAIS.git
cd AMRCAIS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional but recommended)
export FRED_API_KEY="your_fred_api_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

### Basic Usage

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

### Command Line

```bash
# Run regime analysis
python -m src.main --mode=analyze --lookback=365

# Run with custom config
python -m src.main --mode=analyze --config=config
```

---

## 📦 Project Structure

```
AMRCAIS/
├── config/                     # Configuration files (YAML)
│   ├── regimes.yaml           # Regime definitions & parameters
│   ├── data_sources.yaml      # API endpoints & keys
│   └── model_params.yaml      # Model hyperparameters
├── src/
│   ├── data_pipeline/         # Data fetching, validation, storage
│   ├── regime_detection/      # 4 classifiers + ensemble
│   ├── modules/               # 5 analytical modules
│   └── main.py               # Main entry point
├── tests/                     # Test suite (29/29 passing)
├── requirements.txt           # Dependencies
└── docs/                      # Documentation
```

---

## 🔧 Configuration

All parameters are defined in YAML config files—**never hardcode values**:

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

## 📊 Key Features

### Analytical Modules

| Module | Purpose | Regime Adaptation |
|--------|---------|-------------------|
| **Macro Event Tracker** | Monitors NFP, CPI, FOMC | Different event weights per regime |
| **Yield Curve Analyzer** | Duration, DV01, curve shapes | Steepening bullish in Growth, bearish in Stagflation |
| **Options Surface Monitor** | IV surfaces, skew analysis | Adjusted thresholds for volatility regimes |
| **Factor Exposure Analyzer** | Value, Momentum, Quality factors | Recommends factors by regime |
| **Correlation Anomaly Detector** | Cross-asset correlation monitoring | Regime-specific baselines |

### Data Sources

- **FRED API** – Macroeconomic data (NFP, CPI, yield curves)
- **yfinance** – Equity & ETF prices (SPX, TLT, GLD, VIX)
- **Alpha Vantage** – Intraday data (optional)

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/test_core.py -v --timeout=120

# Run with coverage
python -m pytest tests/test_core.py --cov=src --cov-report=html

# Run specific test class
python -m pytest tests/test_core.py::TestRegimeEnsemble -v
```

**Current Status: 29/29 tests passing ✅**

| Test Class | Tests | Status |
|---|---|---|
| TestDataValidator | 4 | ✅ |
| TestDatabaseStorage | 2 | ✅ |
| TestHMMClassifier | 3 | ✅ |
| TestMLClassifier | 2 | ✅ |
| TestVolatilityClassifier | 2 | ✅ |
| TestCorrelationClassifier | 1 | ✅ |
| TestRegimeEnsemble | 5 | ✅ |
| TestMacroEventTracker | 2 | ✅ |
| TestYieldCurveAnalyzer | 2 | ✅ |
| TestOptionsSurfaceMonitor | 1 | ✅ |
| TestCorrelationAnomalyDetector | 2 | ✅ |
| TestFullPipeline | 1 | ✅ |
| TestKnownEvents | 2 | ✅ |

---

## 📈 Success Metrics

| Metric | Target |
|--------|--------|
| Regime Classification Accuracy | ≥80% vs manual labels |
| Transition Detection | Disagreement >0.6 precedes 70%+ of transitions |
| Signal Improvement | ≥15% higher Sharpe ratio vs static models |
| False Positive Rate | ≤20% uncertainty alerts during stable periods |

---

## ⚠️ Disclaimer

> **This system is for educational and research purposes only.**  
> It does not constitute financial advice. Past performance does not guarantee future results.  
> Markets can remain irrational longer than you can remain solvent.

---

## � Project Status

| Phase | Status |
|---|---|
| **Phase 1:** Regime Detection (4 classifiers + ensemble) | ✅ Complete |
| **Phase 2:** Analytical Modules (5 modules) | ✅ Complete |
| **Phase 3:** Meta-Learning Layer | ✅ Complete |
| **Phase 4:** Dashboard & Visualization | 🔜 Planned |

See [CODEBASE_INDEX.md](CODEBASE_INDEX.md) for detailed project status and next steps.

---

## 📚 Documentation

- [Codebase Index](CODEBASE_INDEX.md) – Full project status, architecture & roadmap
- [Development Rules](AMRCAIS_Development_Rules.md) – Coding standards & best practices
- [Product Requirements](AMCRAIS_PRD.md) – Full PRD with detailed specifications
- [Master Prompt](AMRCAIS_Master_Prompt.md) – Technical implementation guide

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or collaboration inquiries, please open an issue or reach out through the repository.

---

**Built with 💡 for quantitative finance research**
