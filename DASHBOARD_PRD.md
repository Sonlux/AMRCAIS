# AMRCAIS Dashboard — Product Requirements Document

**Version:** 1.0  
**Date:** February 12, 2026  
**Status:** Planning  
**Stack:** Next.js 14 (App Router) + FastAPI + TradingView Charts  
**Parent Document:** AMCRAIS_PRD.md (Phase 4: Visualization)

---

## 1. Objective

Build a professional-grade quant dashboard that visualizes AMRCAIS regime detection, module signals, backtest results, and meta-learning performance. The dashboard should look and feel like a Bloomberg terminal or institutional internal tool — not a student project.

**Design Philosophy:** Dark theme, data-dense, zero visual fluff. Every pixel serves an analytical purpose.

---

## 2. Architecture

### 2.1 System Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    BROWSER (Client)                      │
│  Next.js 14 App Router + shadcn/ui + TailwindCSS         │
│  TradingView Lightweight Charts | Plotly.js | AG Grid    │
│  TanStack Query for data fetching + caching              │
└──────────────────────┬───────────────────────────────────┘
                       │ REST (JSON)
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│  Pydantic schemas | CORS | Background tasks              │
│  Singleton AMRCAIS instance | SQLite session              │
├──────────────────────────────────────────────────────────┤
│                 AMRCAIS Python Core                       │
│  src/regime_detection/ | src/modules/ | src/meta_learning/│
│  src/data_pipeline/    | config/                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
d:\AMRCAIS\
├── src/                        # Existing Python core (NO CHANGES)
├── config/                     # Existing configs (NO CHANGES)
├── tests/                      # Existing tests (NO CHANGES)
│
├── api/                        # NEW — FastAPI backend
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, CORS, lifespan events
│   ├── dependencies.py         # AMRCAIS singleton, DB session provider
│   ├── schemas.py              # Pydantic response/request models
│   └── routes/
│       ├── __init__.py
│       ├── regime.py           # /api/regime/*
│       ├── modules.py          # /api/modules/*
│       ├── backtest.py         # /api/backtest/*
│       ├── meta.py             # /api/meta/*
│       └── data.py             # /api/data/*
│
├── dashboard/                  # NEW — Next.js frontend
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── next.config.mjs
│   ├── public/
│   │   └── fonts/              # Inter or JetBrains Mono
│   ├── app/
│   │   ├── layout.tsx          # Root layout: dark theme, sidebar, topbar
│   │   ├── page.tsx            # Overview dashboard (default route)
│   │   ├── globals.css         # Tailwind base + custom theme tokens
│   │   ├── regime/
│   │   │   └── page.tsx        # Regime Explorer
│   │   ├── modules/
│   │   │   └── page.tsx        # Module Deep Dive (tabbed)
│   │   ├── correlations/
│   │   │   └── page.tsx        # Correlation Monitor
│   │   ├── backtest/
│   │   │   └── page.tsx        # Backtest Lab
│   │   └── meta/
│   │       └── page.tsx        # Meta-Learning Analytics
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx     # Navigation sidebar
│   │   │   ├── Topbar.tsx      # Breadcrumb + regime status pill
│   │   │   └── ThemeProvider.tsx
│   │   ├── regime/
│   │   │   ├── RegimeBadge.tsx         # Color-coded regime chip
│   │   │   ├── RegimeTimeline.tsx      # TradingView chart + regime bands
│   │   │   ├── ClassifierVotes.tsx     # 4-classifier vote breakdown
│   │   │   └── TransitionMatrix.tsx    # Regime transition probabilities
│   │   ├── signals/
│   │   │   ├── SignalCard.tsx          # Module signal summary card
│   │   │   ├── SignalStrength.tsx      # Horizontal bar strength indicator
│   │   │   └── SignalHistory.tsx       # Signal over time
│   │   ├── charts/
│   │   │   ├── DisagreementGauge.tsx   # Radial gauge (Plotly.js)
│   │   │   ├── CorrelationHeatmap.tsx  # Annotated heatmap (Plotly.js)
│   │   │   ├── YieldCurve.tsx          # Yield curve line/3D (Plotly.js)
│   │   │   ├── VolSurface.tsx          # Implied vol 3D surface (Plotly.js)
│   │   │   ├── FactorBars.tsx          # Factor exposure bar chart
│   │   │   ├── EquityCurve.tsx         # Backtest P&L (TradingView)
│   │   │   └── AccuracyLine.tsx        # Classifier accuracy over time
│   │   └── ui/
│   │       ├── MetricsCard.tsx         # Key stat card (value + delta)
│   │       ├── DataTable.tsx           # AG Grid wrapper
│   │       └── LoadingState.tsx        # Skeleton + spinner
│   └── lib/
│       ├── api.ts              # Typed fetch client for FastAPI
│       ├── types.ts            # TypeScript interfaces matching Pydantic schemas
│       ├── constants.ts        # Regime colors, names, chart config
│       └── utils.ts            # Formatters (dates, percentages, numbers)
│
├── docker-compose.yml          # Run both services
├── Dockerfile.api              # Python API container
├── Dockerfile.dashboard        # Node.js frontend container
└── requirements.txt            # Updated with FastAPI deps
```

---

## 3. Backend — FastAPI

### 3.1 Dependencies (add to requirements.txt)

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
```

### 3.2 API Endpoints

#### Regime Routes (`/api/regime`)

| Method | Endpoint                               | Description                              | Response                   |
| ------ | -------------------------------------- | ---------------------------------------- | -------------------------- |
| `GET`  | `/api/regime/current`                  | Current regime classification            | `RegimeResponse`           |
| `GET`  | `/api/regime/history?start=&end=`      | Regime history with date range           | `RegimeHistoryResponse`    |
| `GET`  | `/api/regime/classifiers`              | Individual classifier votes + confidence | `ClassifierVotesResponse`  |
| `GET`  | `/api/regime/transitions`              | Regime transition matrix (counts)        | `TransitionMatrixResponse` |
| `GET`  | `/api/regime/disagreement?start=&end=` | Disagreement index time series           | `DisagreementResponse`     |

#### Module Routes (`/api/modules`)

| Method | Endpoint                                  | Description                    | Response                 |
| ------ | ----------------------------------------- | ------------------------------ | ------------------------ |
| `GET`  | `/api/modules/summary`                    | All 5 module signals (current) | `ModuleSummaryResponse`  |
| `GET`  | `/api/modules/{name}/analyze`             | Full analysis for one module   | `ModuleAnalysisResponse` |
| `GET`  | `/api/modules/{name}/history?start=&end=` | Signal history for a module    | `SignalHistoryResponse`  |

`{name}` values: `macro`, `yield_curve`, `options`, `factors`, `correlations`

#### Data Routes (`/api/data`)

| Method | Endpoint                                  | Description                | Response                    |
| ------ | ----------------------------------------- | -------------------------- | --------------------------- |
| `GET`  | `/api/data/assets`                        | Available asset list       | `string[]`                  |
| `GET`  | `/api/data/prices/{asset}?start=&end=`    | OHLCV price data           | `PriceResponse`             |
| `GET`  | `/api/data/macro/{indicator}?start=&end=` | Macro indicator series     | `MacroResponse`             |
| `GET`  | `/api/data/correlations?window=60`        | Current correlation matrix | `CorrelationMatrixResponse` |

#### Backtest Routes (`/api/backtest`)

| Method | Endpoint                     | Description                 | Response                 |
| ------ | ---------------------------- | --------------------------- | ------------------------ |
| `POST` | `/api/backtest/run`          | Execute backtest            | `BacktestResultResponse` |
| `GET`  | `/api/backtest/results`      | List saved backtest results | `BacktestListResponse`   |
| `GET`  | `/api/backtest/results/{id}` | Single backtest detail      | `BacktestResultResponse` |

**`POST /api/backtest/run` request body:**

```json
{
  "start_date": "2020-01-01",
  "end_date": "2024-12-31",
  "strategy": "regime_following",
  "initial_capital": 100000,
  "assets": ["SPX", "TLT", "GLD"]
}
```

#### Meta-Learning Routes (`/api/meta`)

| Method | Endpoint                    | Description                 | Response                |
| ------ | --------------------------- | --------------------------- | ----------------------- |
| `GET`  | `/api/meta/performance`     | Classifier accuracy metrics | `PerformanceResponse`   |
| `GET`  | `/api/meta/weights`         | Current ensemble weights    | `WeightsResponse`       |
| `GET`  | `/api/meta/weights/history` | Weight evolution over time  | `WeightHistoryResponse` |
| `GET`  | `/api/meta/recalibrations`  | Recalibration event log     | `RecalibrationResponse` |
| `GET`  | `/api/meta/health`          | System health summary       | `HealthResponse`        |

#### System Routes

| Method | Endpoint      | Description                   |
| ------ | ------------- | ----------------------------- |
| `GET`  | `/api/health` | Backend health check          |
| `GET`  | `/api/status` | AMRCAIS initialization status |

### 3.3 Pydantic Schemas (key models)

```python
class RegimeResponse(BaseModel):
    regime: int                         # 1-4
    regime_name: str                    # "Risk-On Growth"
    confidence: float                   # 0.0-1.0
    disagreement: float                 # 0.0-1.0
    classifier_votes: Dict[str, int]    # {"hmm": 1, "ml": 2, ...}
    timestamp: datetime

class ModuleSignalResponse(BaseModel):
    module: str                         # "macro", "yield_curve", etc.
    signal: str                         # "bullish", "bearish", "neutral"
    strength: float                     # 0.0-1.0
    explanation: str                    # Human-readable reason
    regime_context: str                 # "Signal interpreted under Risk-On"

class BacktestResultResponse(BaseModel):
    id: str
    start_date: date
    end_date: date
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    regime_returns: Dict[str, float]    # Return per regime
    equity_curve: List[EquityPoint]     # date + value pairs
    regime_timeline: List[RegimePoint]  # date + regime pairs
    benchmark_return: float             # Buy-and-hold comparison
```

### 3.4 AMRCAIS Singleton

The FastAPI app initializes a single AMRCAIS instance on startup via a lifespan event. All route handlers access it through a dependency.

```python
# api/dependencies.py
from contextlib import asynccontextmanager
from src.main import AMRCAIS

_system: AMRCAIS | None = None

@asynccontextmanager
async def lifespan(app):
    global _system
    _system = AMRCAIS(config_path="config")
    _system.initialize()
    yield
    _system = None

def get_system() -> AMRCAIS:
    if _system is None:
        raise RuntimeError("AMRCAIS not initialized")
    return _system
```

---

## 4. Frontend — Next.js

### 4.1 Tech Stack

| Package                         | Version            | Purpose                                                            |
| ------------------------------- | ------------------ | ------------------------------------------------------------------ |
| `next`                          | 14.x               | App Router, SSR/SSG                                                |
| `react`                         | 18.x               | UI framework                                                       |
| `typescript`                    | 5.x                | Type safety                                                        |
| `tailwindcss`                   | 3.x                | Utility-first styling                                              |
| `@radix-ui/*`                   | latest             | Accessible primitives (via shadcn/ui)                              |
| `shadcn/ui`                     | latest             | Pre-built dark components (Cards, Tabs, Badges, Selects, Tooltips) |
| `lightweight-charts`            | 4.x                | TradingView-style price + regime charts                            |
| `plotly.js` + `react-plotly.js` | latest             | Heatmaps, 3D surfaces, gauges                                      |
| `ag-grid-react`                 | latest (community) | Data tables                                                        |
| `@tanstack/react-query`         | 5.x                | API data fetching, caching, auto-refresh                           |
| `lucide-react`                  | latest             | Icons                                                              |
| `date-fns`                      | latest             | Date formatting                                                    |

### 4.2 Design System

#### Color Palette

```
Background:       #0a0a0f (near-black)
Surface:          #12121a (cards, panels)
Surface Elevated: #1a1a28 (hover states, modals)
Border:           #2a2a3a (subtle dividers)
Text Primary:     #e4e4ef (high contrast)
Text Secondary:   #8888a0 (labels, descriptions)
Text Muted:       #555568 (timestamps, IDs)

Regime Colors:
  Risk-On Growth (1):       #22c55e (green-500)
  Risk-Off Crisis (2):      #ef4444 (red-500)
  Stagflation (3):          #f59e0b (amber-500)
  Disinflationary Boom (4): #3b82f6 (blue-500)

Signal Colors:
  Bullish:    #22c55e
  Bearish:    #ef4444
  Neutral:    #6b7280
  Cautious:   #f59e0b

Accents:
  Primary:    #6366f1 (indigo-500)
  Success:    #22c55e
  Warning:    #f59e0b
  Danger:     #ef4444
```

#### Typography

```
Headings:     Inter (600/700 weight)
Body:         Inter (400)
Monospace:    JetBrains Mono (numbers, data, code)
```

All numbers, percentages, and data values use monospace font for alignment.

#### Component Patterns

- **Cards:** `bg-surface rounded-lg border border-border p-4` — no shadows (flat design)
- **Metrics:** Large monospace number + small label below + delta badge (green ↑ or red ↓)
- **Badges:** Pill-shaped, regime-colored, no borders
- **Charts:** No gridlines on dark background, thin axis lines, tooltips on hover
- **Tables:** Zebra striping with `bg-surface` / `bg-surface-elevated` alternation
- **Loading:** Skeleton shimmer (no spinners except on backtest execution)

### 4.3 Layout

```
┌─────────────────────────────────────────────────────────────┐
│ ┌──────┐                                                    │
│ │      │  TOPBAR: Breadcrumb | Regime Badge | Last Updated  │
│ │  S   │────────────────────────────────────────────────────│
│ │  I   │                                                    │
│ │  D   │                                                    │
│ │  E   │              MAIN CONTENT AREA                     │
│ │  B   │                                                    │
│ │  A   │         (page-specific content)                    │
│ │  R   │                                                    │
│ │      │                                                    │
│ │  56px│                                                    │
│ │ wide │                                                    │
│ │      │                                                    │
│ └──────┘                                                    │
└─────────────────────────────────────────────────────────────┘
```

**Sidebar (collapsed by default, expand on hover):**

- Icons: Overview (grid), Regime (layers), Modules (puzzle), Correlations (scatter), Backtest (flask), Meta (brain)
- Active state: left accent bar + highlighted icon
- Expand on hover shows labels

---

## 5. Pages — Detailed Specifications

### 5.1 Overview (`/`)

The command center. One glance tells you the current market state.

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT REGIME                    DISAGREEMENT INDEX        │
│ ┌─────────────────────┐          ┌──────────────────┐      │
│ │  ● Risk-On Growth   │          │    GAUGE: 0.23   │      │
│ │  Confidence: 87%    │          │    [LOW]          │      │
│ │  Since: Jan 15      │          │                   │      │
│ └─────────────────────┘          └──────────────────┘      │
│                                                             │
│ MODULE SIGNALS                                              │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────┐│
│ │Macro     │ │Yield     │ │Options   │ │Factors   │ │Corr││
│ │▲ Bullish │ │► Neutral │ │▲ Bullish │ │▼ Bearish │ │► N ││
│ │str: 0.72 │ │str: 0.31 │ │str: 0.65 │ │str: 0.58 │ │0.2 ││
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────┘│
│                                                             │
│ REGIME TIMELINE (last 90 days)                              │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ SPX price line with colored regime bands beneath        │ │
│ │ ████████ ██████████████████████ ████████████            │ │
│ │ green   amber     green         green                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ KEY METRICS (4 cards)                                       │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐ │
│ │Regime Flips│ │Avg Conf    │ │Meta Acc    │ │Days Curr │ │
│ │    3       │ │   84.2%    │ │  81.5%     │ │   28     │ │
│ │  (30d)     │ │  ↑ 2.1%   │ │  ↓ 0.8%   │ │          │ │
│ └────────────┘ └────────────┘ └────────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Data requirements:**

- `GET /api/regime/current` — regime, confidence, disagreement
- `GET /api/modules/summary` — all 5 module signals
- `GET /api/regime/history?start={90d_ago}` — recent timeline
- `GET /api/data/prices/SPX?start={90d_ago}` — price overlay

**Auto-refresh:** Every 60 seconds via TanStack Query `refetchInterval`.

---

### 5.2 Regime Explorer (`/regime`)

Deep dive into regime classification mechanics.

**Section A: Regime Timeline (full width)**

- TradingView Lightweight Chart with SPX price line
- Colored background bands for each regime period
- Date range selector (presets: 1Y, 3Y, 5Y, 10Y, All)
- Hover tooltip: date, regime, confidence, disagreement

**Section B: Classifier Vote Breakdown (left 60%)**

- Stacked area chart: 4 classifier votes over time
- Color intensity = confidence
- Visual: shows when classifiers agree (solid color) vs. disagree (mixed)

**Section C: Disagreement Index (right 40%)**

- Line chart overlaid with threshold line at 0.6
- Shaded regions above 0.6 = "uncertainty zones"
- Annotations on major spikes (auto-detected)

**Section D: Transition Matrix (bottom)**

- 4×4 heatmap: rows = "from regime", columns = "to regime"
- Cell value = count of transitions
- Color intensity = frequency
- Diagonal should be high (stable regimes)

**Data requirements:**

- `GET /api/regime/history?start=&end=`
- `GET /api/regime/classifiers`
- `GET /api/regime/disagreement?start=&end=`
- `GET /api/regime/transitions`
- `GET /api/data/prices/SPX?start=&end=`

---

### 5.3 Module Deep Dive (`/modules`)

Tabbed interface — one tab per analytical module.

**Tab Navigation:** Horizontal tabs: Macro | Yield Curve | Options | Factors | Correlations

**Each tab contains:**

1. **Current Signal Card** — signal direction, strength bar, regime-adaptive explanation
2. **Primary Chart** — module-specific visualization (see below)
3. **Signal History** — line chart of signal strength over time, colored by direction
4. **Regime Parameters** — table showing how parameters change per regime

**Module-specific primary charts:**

| Module       | Chart Type         | Content                                                                                            |
| ------------ | ------------------ | -------------------------------------------------------------------------------------------------- |
| Macro        | Event timeline     | Scatter plot: events on x-axis, surprise magnitude on y-axis, color = bullish/bearish              |
| Yield Curve  | Multi-line + 3D    | Current curve (line), historical evolution (3D surface), shape badge (Normal/Flat/Inverted/Humped) |
| Options      | 3D surface + gauge | Implied vol surface (strike × expiry × vol), VIX level gauge, skew indicator                       |
| Factors      | Grouped bar chart  | Factor exposures (Value, Momentum, Quality, Vol) per regime, current regime highlighted            |
| Correlations | Annotated heatmap  | 6×6 asset correlation matrix, anomalous cells highlighted (deviation from regime baseline)         |

---

### 5.4 Correlation Monitor (`/correlations`)

Dedicated view for cross-asset correlation analysis.

**Section A: Live Correlation Matrix (left 50%)**

- 6×6 Plotly annotated heatmap (SPX, TLT, GLD, DXY, WTI, VIX)
- Color scale: -1 (blue) → 0 (gray) → +1 (red)
- Cell annotations: correlation value
- Rolling window slider: 20 / 40 / 60 / 120 days

**Section B: Regime Baseline Comparison (right 50%)**

- Same heatmap but showing difference from regime-expected correlations
- Green = within range, Yellow = elevated, Red = anomalous
- Anomaly count badge

**Section C: Correlation Time Series (bottom)**

- Select any pair (e.g., SPX-TLT) from dropdown
- Line chart of rolling correlation
- Regime bands in background
- Threshold lines for "normal" range in current regime

---

### 5.5 Backtest Lab (`/backtest`)

Run and visualize historical backtests.

**Controls Panel (top):**

- Date range picker (Mantine-style calendar)
- Strategy selector: Regime Following, Static Allocation, Risk Parity
- Asset checkboxes: SPX, TLT, GLD, DXY, WTI
- Initial capital input
- **Run Backtest** button (primary accent color)

**Results (shown after execution):**

**Section A: Equity Curve (full width)**

- TradingView area chart: strategy vs. benchmark (buy-and-hold SPX)
- Regime-colored background bands
- Drawdown area chart below (inverted, red fill)

**Section B: Performance Stats (4 cards)**

- Total Return (%) with delta vs benchmark
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate by Regime

**Section C: Regime-Segmented Returns (table)**

| Regime          | Strategy Return | Benchmark Return | Days | Hit Rate |
| --------------- | --------------- | ---------------- | ---- | -------- |
| Risk-On Growth  | +14.2%          | +12.1%           | 523  | 62%      |
| Risk-Off Crisis | -2.1%           | -18.4%           | 87   | 71%      |
| Stagflation     | +1.8%           | -5.2%            | 201  | 55%      |
| Disinfl. Boom   | +9.3%           | +8.7%            | 312  | 58%      |

**Section D: Trade Log (collapsible)**

- AG Grid table: Date, Action, Regime, Asset, Return
- Sortable, filterable, exportable to CSV

---

### 5.6 Meta-Learning Analytics (`/meta`)

Track how well the system learns and adapts.

**Section A: Classifier Accuracy Over Time (full width)**

- Multi-line chart: one line per classifier (HMM, ML, Correlation, Volatility) + ensemble
- Y-axis: rolling accuracy (%)
- X-axis: date
- Threshold line at 75% (recalibration trigger)

**Section B: Ensemble Weight Evolution (left 60%)**

- Stacked area chart: weight allocation across 4 classifiers over time
- Shows how meta-learner shifts trust between classifiers

**Section C: Recalibration Log (right 40%)**

- Vertical timeline of recalibration events
- Each event: date, trigger reason, severity, before/after weights
- Color-coded by severity (info/warning/critical)

**Section D: Disagreement Signal vs. Market (bottom)**

- Dual-axis chart
  - Left Y: Disagreement index (0-1)
  - Right Y: SPX price
- Shaded vertical bands where disagreement > 0.6
- Annotations: "preceded March 2020 crash by 12 days"

---

## 6. Interactions & Behavior

### 6.1 Data Loading

| Pattern            | Implementation                                        |
| ------------------ | ----------------------------------------------------- |
| Initial load       | TanStack Query with `suspense: true` → show skeleton  |
| Background refresh | `refetchInterval: 60000` on Overview page             |
| Stale data         | `staleTime: 30000` — data considered fresh for 30s    |
| Error state        | Red banner with retry button, preserve last good data |
| Empty state        | Illustrated placeholder with actionable message       |

### 6.2 Chart Interactions

| Interaction      | Behavior                                                                          |
| ---------------- | --------------------------------------------------------------------------------- |
| Hover            | Tooltip with date, value, regime context                                          |
| Click data point | If applicable, drill down (e.g., click regime band → navigate to Regime Explorer) |
| Zoom             | Mouse wheel on TradingView charts, drag-to-zoom on Plotly                         |
| Pan              | Click-drag on time axis                                                           |
| Date range       | Synced across all charts on the same page via URL query params                    |
| Export           | Right-click chart → "Download PNG" (Plotly built-in)                              |

### 6.3 URL State

All filter state lives in URL query parameters so pages are shareable and bookmarkable:

```
/regime?start=2022-01-01&end=2023-12-31
/modules?tab=yield_curve
/correlations?pair=SPX-TLT&window=60
/backtest?id=bt_20240115_001
```

---

## 7. Performance Requirements

| Metric                           | Target                       |
| -------------------------------- | ---------------------------- |
| First Contentful Paint           | < 1.5s                       |
| Time to Interactive              | < 3s                         |
| API response (simple query)      | < 200ms                      |
| API response (backtest)          | < 30s (show progress bar)    |
| Chart render (1000 data points)  | < 500ms                      |
| Chart render (10000 data points) | < 2s                         |
| Bundle size (gzipped)            | < 500KB (code-split by page) |

### Optimization Strategies

- **Code splitting:** Each page lazy-loaded via Next.js App Router
- **Chart lazy loading:** Plotly.js loaded only on pages that use it (dynamic import)
- **API response caching:** TanStack Query caches + backend LRU cache for expensive computations
- **Virtualized tables:** AG Grid handles large datasets natively
- **Static generation:** Overview layout statically generated, data hydrated client-side

---

## 8. Deployment

### 8.1 Docker Compose

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - AMRCAIS_ENV=production

  dashboard:
    build:
      context: ./dashboard
      dockerfile: ../Dockerfile.dashboard
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - api
```

### 8.2 Development Mode

```bash
# Terminal 1: API
cd d:\AMRCAIS
uvicorn api.main:app --reload --port 8000

# Terminal 2: Dashboard
cd d:\AMRCAIS\dashboard
npm run dev
```

### 8.3 Production Targets

- **Primary:** Docker Compose on a VPS (DigitalOcean / Hetzner)
- **Alternative:** Railway (auto-deploys from GitHub)
- **Frontend CDN:** Vercel (free tier, auto-deploy Next.js)

---

## 9. Implementation Plan

### Week 1: Backend + Scaffold

| Day | Task                                                                                   |
| --- | -------------------------------------------------------------------------------------- |
| 1   | FastAPI app scaffold, CORS, health check, AMRCAIS singleton lifespan                   |
| 2   | Pydantic schemas for all response models                                               |
| 3   | Regime routes: `/current`, `/history`, `/classifiers`, `/disagreement`                 |
| 4   | Module routes: `/summary`, `/{name}/analyze` + Data routes: `/prices`, `/correlations` |
| 5   | Meta routes + Backtest route (basic). Test all endpoints via Swagger UI                |

**Exit Criteria:** All API endpoints return valid JSON. Swagger UI fully functional at `localhost:8000/docs`.

### Week 2: Frontend Foundation

| Day | Task                                                                        |
| --- | --------------------------------------------------------------------------- |
| 1   | `npx create-next-app`, install shadcn/ui, Tailwind dark theme config, fonts |
| 2   | Root layout: Sidebar, Topbar, ThemeProvider. Navigation between 6 pages     |
| 3   | Overview page: RegimeBadge, DisagreementGauge (Plotly), MetricsCards        |
| 4   | Overview page: 5 SignalCards, mini regime timeline (TradingView)            |
| 5   | API client (`lib/api.ts`), TanStack Query hooks, loading/error states       |

**Exit Criteria:** Overview page fully functional with live data from API. Dark theme polished.

### Week 3: Core Pages

| Day | Task                                                                              |
| --- | --------------------------------------------------------------------------------- |
| 1   | Regime Explorer: TradingView timeline with regime bands, date range selector      |
| 2   | Regime Explorer: Classifier votes chart, disagreement line, transition matrix     |
| 3   | Module Deep Dive: Tab layout, all 5 tabs with signal card + regime params table   |
| 4   | Module Deep Dive: Per-module charts (yield curve line, factor bars, macro events) |
| 5   | Correlation Monitor: Heatmap, baseline comparison, pair time series               |

**Exit Criteria:** Regime, Modules, and Correlations pages fully functional.

### Week 4: Backtest + Meta + Charts

| Day | Task                                                                          |
| --- | ----------------------------------------------------------------------------- |
| 1   | Backtest Lab: Controls panel, run button → call API → show loading            |
| 2   | Backtest Lab: Equity curve (TradingView), stats cards, regime-segmented table |
| 3   | Meta-Learning: Accuracy lines, weight evolution stacked area                  |
| 4   | Meta-Learning: Recalibration timeline, disagreement vs. SPX dual-axis         |
| 5   | Vol surface 3D, yield curve 3D, enhance Plotly chart configs                  |

**Exit Criteria:** All 6 pages fully functional with real data.

### Week 5: Polish + Deploy

| Day | Task                                                                  |
| --- | --------------------------------------------------------------------- |
| 1   | Loading skeletons on every page, error boundaries, empty states       |
| 2   | Responsive adjustments (1280px+ primary, 1024px graceful degrade)     |
| 3   | Chart hover tooltips, cross-chart date syncing, URL state persistence |
| 4   | Dockerfiles (api + dashboard), docker-compose, test full deployment   |
| 5   | README updates, screenshot captures, final bug fixes                  |

**Exit Criteria:** Project runs with `docker compose up`. All pages render correctly. Demo-ready.

---

## 10. Testing Strategy

### Backend Tests

```
tests/
├── test_api_regime.py      # Regime endpoint responses
├── test_api_modules.py     # Module endpoint responses
├── test_api_backtest.py    # Backtest execution + results
├── test_api_meta.py        # Meta-learning endpoints
└── test_api_schemas.py     # Pydantic model validation
```

- Use `httpx.AsyncClient` with FastAPI's `TestClient`
- Mock AMRCAIS instance for unit tests
- One integration test with real AMRCAIS (slow, marked `@pytest.mark.slow`)

### Frontend Tests

- **Component tests:** Vitest + React Testing Library for critical components (RegimeBadge, SignalCard)
- **E2E (optional):** Playwright for smoke test (load each page, verify no crashes)
- **Type safety:** TypeScript strict mode catches most issues at compile time

---

## 11. Out of Scope (V1)

The following are explicitly deferred to future versions:

- Real-time WebSocket push updates (V1 uses polling)
- User authentication / multi-user
- Saved user layouts / preferences
- Mobile responsive design (desktop-first, 1280px minimum)
- PDF report export
- Alert / notification system
- Multiple concurrent backtest comparison
- Custom regime definition (always 4 regimes)

---

## 12. Success Criteria

| Criterion      | Metric                                                                    |
| -------------- | ------------------------------------------------------------------------- |
| Visual quality | Looks comparable to an institutional internal tool, not a student project |
| Data accuracy  | Dashboard numbers match Python backend calculations exactly               |
| Performance    | All pages load in < 3 seconds on localhost                                |
| Completeness   | All 6 pages functional with real AMRCAIS data                             |
| Deployability  | Single `docker compose up` starts entire system                           |
| Code quality   | TypeScript strict, no `any` types, ESLint clean                           |

---

**Document Owner:** AMRCAIS Development Team  
**Last Updated:** February 12, 2026  
**Next Review:** End of Week 1 (Backend complete)
