"""
Pydantic response/request models for the AMRCAIS Dashboard API.

All schemas are designed to be JSON-serializable and match the
TypeScript interfaces on the frontend.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Regime Schemas ───────────────────────────────────────────────


class RegimeResponse(BaseModel):
    """Current regime classification."""

    regime: int = Field(..., ge=1, le=4, description="Regime ID (1-4)")
    regime_name: str = Field(..., description="Human-readable regime name")
    confidence: float = Field(..., ge=0, le=1)
    disagreement: float = Field(..., ge=0, le=1)
    classifier_votes: Dict[str, int] = Field(
        default_factory=dict,
        description="Individual classifier regime predictions",
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across regimes",
    )
    transition_warning: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


class RegimeHistoryPoint(BaseModel):
    """Single point in regime history."""

    date: str
    regime: int
    regime_name: str
    confidence: float
    disagreement: float


class RegimeHistoryResponse(BaseModel):
    """Regime classification history."""

    history: List[RegimeHistoryPoint]
    total_points: int
    start_date: str
    end_date: str


class ClassifierVoteEntry(BaseModel):
    """Single classifier vote snapshot."""

    classifier: str
    regime: int
    confidence: float
    weight: float


class ClassifierVotesResponse(BaseModel):
    """All classifier votes for the current period."""

    votes: List[ClassifierVoteEntry]
    ensemble_regime: int
    ensemble_confidence: float
    weights: Dict[str, float]


class TransitionMatrixResponse(BaseModel):
    """Regime transition count matrix."""

    matrix: List[List[int]] = Field(
        ...,
        description="4x4 matrix: matrix[from_regime][to_regime] = count",
    )
    regime_names: List[str]
    total_transitions: int


class DisagreementPoint(BaseModel):
    """Single disagreement data point."""

    date: str
    disagreement: float
    threshold_exceeded: bool = False


class DisagreementResponse(BaseModel):
    """Disagreement index time series."""

    series: List[DisagreementPoint]
    avg_disagreement: float
    max_disagreement: float
    threshold: float = 0.6


# ─── Module Schemas ───────────────────────────────────────────────


class ModuleSignalResponse(BaseModel):
    """Single module signal."""

    module: str
    signal: str = Field(..., description="bullish | bearish | neutral | cautious")
    strength: float = Field(..., ge=0, le=1)
    confidence: float = Field(0.5, ge=0, le=1)
    explanation: str = ""
    regime_context: str = ""


class ModuleSummaryResponse(BaseModel):
    """All module signals at a glance."""

    signals: List[ModuleSignalResponse]
    current_regime: int
    regime_name: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ModuleAnalysisResponse(BaseModel):
    """Full analysis result for a single module."""

    module: str
    signal: ModuleSignalResponse
    raw_metrics: Dict[str, Any] = Field(default_factory=dict)
    regime_parameters: Dict[str, Any] = Field(default_factory=dict)


class SignalHistoryPoint(BaseModel):
    """Single point in signal history."""

    date: str
    signal: str
    strength: float
    confidence: float = 0.5
    regime: Optional[int] = None



class SignalHistoryResponse(BaseModel):
    """Signal history for a module."""

    module: str
    history: List[SignalHistoryPoint]


# ─── Data Schemas ─────────────────────────────────────────────────


class PricePoint(BaseModel):
    """Single OHLCV data point."""

    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None


class PriceResponse(BaseModel):
    """Asset price data."""

    asset: str
    prices: List[PricePoint]
    total_points: int


class CorrelationMatrixResponse(BaseModel):
    """Cross-asset correlation matrix."""

    assets: List[str]
    matrix: List[List[float]]
    window: int = 60


# ─── Backtest Schemas ─────────────────────────────────────────────


class BacktestRequest(BaseModel):
    """Backtest configuration."""

    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    strategy: str = "regime_following"
    initial_capital: float = 100_000
    assets: List[str] = Field(default_factory=lambda: ["SPX", "TLT", "GLD"])


class EquityPoint(BaseModel):
    """Single equity curve data point."""

    date: str
    value: float
    regime: Optional[int] = None


class RegimeReturnEntry(BaseModel):
    """Return statistics for a single regime."""

    regime: int
    regime_name: str
    strategy_return: float
    benchmark_return: float
    days: int
    hit_rate: float


class DrawdownPoint(BaseModel):
    """Single drawdown observation."""

    date: str
    drawdown: float  # negative pct from peak (e.g. -5.2)


class TradeLogEntry(BaseModel):
    """One allocation change in the backtest trade log."""

    date: str
    action: str  # e.g. "rebalance", "hold"
    regime: int
    regime_name: str
    allocations: Dict[str, float]  # {"SPX": 0.60, "GLD": 0.20, …}
    portfolio_value: float
    daily_return: float


class BacktestResultResponse(BaseModel):
    """Full backtest result."""

    id: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    benchmark_return: float
    equity_curve: List[EquityPoint]
    drawdown_curve: List[DrawdownPoint] = Field(default_factory=list)
    trade_log: List[TradeLogEntry] = Field(default_factory=list)
    regime_returns: List[RegimeReturnEntry]
    strategy: str
    initial_capital: float


# ─── Meta-Learning Schemas ────────────────────────────────────────


class ClassifierAccuracyPoint(BaseModel):
    """Accuracy data point for one classifier."""

    date: str
    accuracy: float
    classifier: str


class PerformanceResponse(BaseModel):
    """Classifier performance metrics."""

    stability_score: float
    stability_rating: str
    transition_count: int
    avg_disagreement: float
    high_disagreement_days: int
    total_classifications: int
    regime_distribution: Dict[str, int] = Field(default_factory=dict)


class WeightsResponse(BaseModel):
    """Current ensemble classifier weights."""

    weights: Dict[str, float]
    is_adaptive: bool = True


class WeightHistoryPoint(BaseModel):
    """Weight snapshot at a point in time."""

    date: str
    weights: Dict[str, float]


class WeightHistoryResponse(BaseModel):
    """Ensemble weight evolution."""

    history: List[WeightHistoryPoint]


class RecalibrationEvent(BaseModel):
    """A single recalibration event."""

    date: str
    trigger_reason: str
    severity: float
    urgency: str
    recommendations: List[str] = Field(default_factory=list)


class RecalibrationResponse(BaseModel):
    """Recalibration event log."""

    events: List[RecalibrationEvent]
    total_recalibrations: int
    last_recalibration: Optional[str] = None


class HealthResponse(BaseModel):
    """System health summary."""

    system_status: str
    needs_recalibration: bool
    urgency: str = "none"
    severity: float = 0.0
    reasons: List[str] = Field(default_factory=list)
    performance_30d: Dict[str, Any] = Field(default_factory=dict)
    performance_7d: Dict[str, Any] = Field(default_factory=dict)
    alerts: Dict[str, Any] = Field(default_factory=dict)
    adaptive_weights: Optional[Dict[str, float]] = None


# ─── System Schemas ───────────────────────────────────────────────


class StatusResponse(BaseModel):
    """AMRCAIS system status."""

    is_initialized: bool
    current_regime: Optional[int] = None
    confidence: float = 0.0
    disagreement: float = 0.0
    modules_loaded: List[str] = Field(default_factory=list)
    uptime_seconds: Optional[float] = None


class HealthCheckResponse(BaseModel):
    """Simple health check."""

    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None


# ─── Macro Data Schema ────────────────────────────────────────────


class MacroDataPoint(BaseModel):
    """Single macro indicator observation."""

    date: str
    value: float


class MacroDataResponse(BaseModel):
    """Time-series of a single macro indicator."""

    indicator: str
    series: List[MacroDataPoint]
    total_points: int


# ─── Classifier Accuracy Schema ───────────────────────────────────


class ClassifierAccuracyResponse(BaseModel):
    """Rolling accuracy series for each classifier + ensemble."""

    classifiers: List[str]
    series: List[ClassifierAccuracyPoint]  # interleaved: each date has N entries
    window: int = 30  # rolling window used


# ─── Surface / 3D Chart Schemas ───────────────────────────────────


class YieldCurveDataResponse(BaseModel):
    """Yield curve snapshot with observed + interpolated points."""

    tenors: List[float] = Field(
        ..., description="Tenors in years (e.g. [0.25, 0.5, 1, 2, …])"
    )
    yields: List[float] = Field(
        ..., description="Yield values corresponding to each tenor"
    )
    curve_shape: str = Field(
        "normal", description="Classified shape: normal, flat, inverted, humped"
    )
    slope_2_10: Optional[float] = None
    slope_3m_10: Optional[float] = None
    curvature: Optional[float] = None
    regime: int = 1
    regime_name: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class VolSurfaceDataResponse(BaseModel):
    """Volatility surface grid for 3D rendering.

    Grid convention: iv_grid[i][j] = IV for moneyness[j] at expiry_days[i].
    """

    moneyness: List[float] = Field(
        ..., description="Moneyness levels (1.0 = ATM)"
    )
    expiry_days: List[int] = Field(
        ..., description="Days to expiration for each row"
    )
    iv_grid: List[List[float]] = Field(
        ..., description="2D IV grid [expiry_idx][moneyness_idx]"
    )
    atm_vol: float = Field(..., description="Current ATM vol (VIX proxy)")
    regime: int = 1
    regime_name: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


# ─── Validation Constants ─────────────────────────────────────────

VALID_STRATEGIES = {"regime_following", "momentum", "mean_reversion"}
VALID_ASSETS = {"SPX", "TLT", "GLD", "DXY", "WTI", "VIX"}


# ─── Phase 2: Transition Forecast Schemas ─────────────────────────


class TransitionForecastResponse(BaseModel):
    """Forward-looking regime transition probabilities."""

    current_regime: int
    horizon_days: int = 30
    hmm_probs: Dict[str, float] = Field(
        default_factory=dict,
        description="HMM-derived transition probabilities",
    )
    indicator_probs: Dict[str, float] = Field(
        default_factory=dict,
        description="Indicator-adjusted probabilities",
    )
    blended_probs: Dict[str, float] = Field(
        default_factory=dict,
        description="Final blended probabilities",
    )
    leading_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Current leading indicator values",
    )
    transition_risk: float = Field(
        0.0, description="Probability of any regime change"
    )
    most_likely_next: int = Field(1, description="Most likely next regime ID")
    most_likely_next_name: str = ""
    confidence: float = 0.0


# ─── Phase 2: Multi-Timeframe Schemas ─────────────────────────────


class TimeframeRegimeResponse(BaseModel):
    """Regime result for a single timeframe."""

    timeframe: str
    regime: int
    regime_name: str
    confidence: float
    disagreement: float
    transition_warning: bool = False
    duration: int = 0


class MultiTimeframeResponse(BaseModel):
    """Aggregated multi-timeframe regime view."""

    daily: TimeframeRegimeResponse
    weekly: TimeframeRegimeResponse
    monthly: TimeframeRegimeResponse
    conflict_detected: bool = False
    highest_conviction: str = "daily"
    trade_signal: str = ""
    agreement_score: float = 1.0


# ─── Phase 2: Contagion Network Schemas ───────────────────────────


class GrangerLinkResponse(BaseModel):
    """Single Granger causality link."""

    cause: str
    effect: str
    f_stat: float
    p_value: float
    lag: int
    significant: bool


class SpilloverResponse(BaseModel):
    """Diebold-Yilmaz spillover decomposition."""

    total_spillover_index: float
    directional_to: Dict[str, float] = Field(default_factory=dict)
    directional_from: Dict[str, float] = Field(default_factory=dict)
    net_spillover: Dict[str, float] = Field(default_factory=dict)
    pairwise: List[List[float]] = Field(default_factory=list)
    assets: List[str] = Field(default_factory=list)


class NetworkGraphResponse(BaseModel):
    """Network graph for visualization."""

    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0


class ContagionAnalysisResponse(BaseModel):
    """Full contagion network analysis."""

    signal: ModuleSignalResponse
    granger_network: List[GrangerLinkResponse] = Field(default_factory=list)
    spillover: SpilloverResponse
    network_graph: NetworkGraphResponse
    contagion_flags: Dict[str, bool] = Field(default_factory=dict)
    n_significant_links: int = 0
    network_density: float = 0.0


# ─── Phase 2: Surprise Decay Schemas ──────────────────────────────


class DecaySurpriseResponse(BaseModel):
    """Single active decaying surprise."""

    indicator: str
    surprise: float
    release_date: str
    half_life_days: float
    initial_weight: float
    regime_at_release: int


class SurpriseIndexResponse(BaseModel):
    """Cumulative surprise index."""

    index: float
    direction: str
    components: Dict[str, float] = Field(default_factory=dict)
    active_surprises: int = 0
    total_historical: int = 0


class DecayCurvePoint(BaseModel):
    """Single point on a decay curve."""

    day: int
    impact: float
    is_stale: bool = False


class DecayCurveResponse(BaseModel):
    """Decay curves for all active surprises."""

    curves: Dict[str, List[DecayCurvePoint]] = Field(default_factory=dict)


# ─── Phase 2: Narrative Schemas ───────────────────────────────────


class NarrativeResponse(BaseModel):
    """Daily narrative briefing."""

    headline: str
    regime_section: str
    signal_section: str
    risk_section: str
    positioning_section: str = ""
    full_text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: Dict[str, str] = Field(default_factory=dict)


# ─── Phase 3: Return Forecast Schemas ─────────────────────────────


class FactorContributionResponse(BaseModel):
    """Per-factor contribution to expected return."""

    factor: str
    contribution: float


class ReturnForecastResponse(BaseModel):
    """Regime-conditional return forecast for one asset."""

    asset: str
    regime: int
    expected_return: float
    volatility: float
    r_squared_regime: float = 0.0
    r_squared_static: float = 0.0
    r_squared_improvement: float = 0.0
    kelly_fraction: float = 0.0
    factor_contributions: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.5


class ReturnForecastsResponse(BaseModel):
    """Return forecasts for all assets."""

    current_regime: int
    regime_name: str = ""
    forecasts: List[ReturnForecastResponse] = Field(default_factory=list)


class RegimeCoefficientsResponse(BaseModel):
    """Fitted coefficients for all regimes for a single asset."""

    asset: str
    regimes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Regime ID → {alpha, betas, residual_vol, r_squared, n_obs}",
    )


# ─── Phase 3: Tail Risk Schemas ──────────────────────────────────


class ScenarioVaRResponse(BaseModel):
    """VaR/CVaR for a single regime-transition scenario."""

    from_regime: int
    to_regime: int
    to_regime_name: str
    probability: float
    var_99: float
    cvar_99: float
    contribution: float
    risk_drivers: Dict[str, float] = Field(default_factory=dict)


class HedgeRecommendationResponse(BaseModel):
    """Single hedging recommendation."""

    scenario: str
    instrument: str
    rationale: str
    urgency: str = "medium"


class TailRiskResponse(BaseModel):
    """Full tail risk analysis."""

    current_regime: int
    weighted_var: float
    weighted_cvar: float
    scenarios: List[ScenarioVaRResponse] = Field(default_factory=list)
    worst_scenario: str = ""
    worst_scenario_var: float = 0.0
    tail_risk_driver: str = ""
    hedge_recommendations: List[HedgeRecommendationResponse] = Field(
        default_factory=list
    )
    portfolio_weights: Dict[str, float] = Field(default_factory=dict)


# ─── Phase 3: Portfolio Optimizer Schemas ─────────────────────────


class RegimeAllocationResponse(BaseModel):
    """Optimal allocation for a single regime scenario."""

    regime: int
    regime_name: str
    weights: Dict[str, float] = Field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0


class PortfolioOptimizationResponse(BaseModel):
    """Full portfolio optimisation result."""

    current_regime: int
    blended_weights: Dict[str, float] = Field(default_factory=dict)
    regime_allocations: List[RegimeAllocationResponse] = Field(
        default_factory=list
    )
    rebalance_trigger: bool = False
    rebalance_reason: str = ""
    transaction_cost_estimate: float = 0.0
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_constraint: float = 0.15


# ─── Phase 3: Alpha Signal Schemas ───────────────────────────────


class AlphaSignalResponse(BaseModel):
    """Single tradeable alpha signal."""

    anomaly_type: str
    direction: str
    rationale: str
    strength: float = 0.0
    confidence: float = 0.5
    holding_period_days: int = 10
    historical_win_rate: float = 0.5
    regime: int = 1


class AlphaSignalsResponse(BaseModel):
    """All alpha signals from anomaly analysis."""

    signals: List[AlphaSignalResponse] = Field(default_factory=list)
    composite_score: float = 0.0
    top_signal: Optional[str] = None
    regime: int = 1
    n_active_anomalies: int = 0
    regime_context: str = ""


# ─── Phase 4: Real-Time & Execution Schemas ──────────────────────


class AlertResponse(BaseModel):
    """Single alert record."""

    alert_id: str
    alert_type: str
    severity: str
    title: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float
    source_event_id: str = ""
    acknowledged: bool = False


class AlertsResponse(BaseModel):
    """List of alerts."""

    alerts: List[AlertResponse] = Field(default_factory=list)
    total: int = 0
    unacknowledged: int = 0


class AlertConfigItem(BaseModel):
    """Configuration for a single alert type."""

    enabled: bool = True
    cooldown_seconds: float = 300.0
    threshold: float = 0.0


class AlertConfigRequest(BaseModel):
    """Request to update alert configuration."""

    alert_type: str = Field(..., description="Alert type to configure")
    enabled: Optional[bool] = None
    cooldown_seconds: Optional[float] = None
    threshold: Optional[float] = None


class AlertConfigResponse(BaseModel):
    """Current alert configuration."""

    configs: Dict[str, AlertConfigItem] = Field(default_factory=dict)


class EventResponse(BaseModel):
    """Single event from the event bus."""

    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    event_id: str
    source: str = ""


class EventsResponse(BaseModel):
    """Event bus history."""

    events: List[EventResponse] = Field(default_factory=list)
    total: int = 0


class PaperOrderResponse(BaseModel):
    """Single paper trade order."""

    order_id: str
    asset: str
    side: str
    quantity: float
    price: float
    status: str
    regime: Optional[int] = None
    timestamp: float
    fill_timestamp: Optional[float] = None
    reason: str = ""
    notional: float = 0.0


class PositionResponse(BaseModel):
    """Single portfolio position."""

    asset: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioSummaryResponse(BaseModel):
    """Paper trading portfolio snapshot."""

    total_equity: float
    cash: float
    positions_value: float
    num_positions: int
    positions: List[PositionResponse] = Field(default_factory=list)
    total_return_pct: float = 0.0
    rebalance_count: int = 0
    last_rebalance: Optional[float] = None
    current_regime: Optional[int] = None


class PerformanceMetricsResponse(BaseModel):
    """Paper trading performance metrics."""

    total_return: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    num_rebalances: int = 0
    win_rate: float = 0.0
    initial_capital: float = 100_000.0
    current_equity: float = 100_000.0
    cash: float = 100_000.0


class RegimeAttributionResponse(BaseModel):
    """P&L breakdown by market regime."""

    regime_pnl: Dict[str, float] = Field(default_factory=dict)
    total_realized: float = 0.0
    unrealized: float = 0.0


class RebalanceRequest(BaseModel):
    """Request to trigger a portfolio rebalance."""

    target_weights: Dict[str, float] = Field(
        ..., description="Asset → target weight (0-1)"
    )
    prices: Dict[str, float] = Field(
        ..., description="Asset → current price"
    )
    regime: Optional[int] = Field(None, description="Current regime ID")
    reason: str = "manual_rebalance"


class RebalanceResponse(BaseModel):
    """Result of a portfolio rebalance."""

    orders: List[PaperOrderResponse] = Field(default_factory=list)
    total_equity: float
    cash: float
    num_orders: int = 0


class EquityCurvePoint(BaseModel):
    """Single equity curve data point."""

    timestamp: float
    equity: float
    cash: float
    positions_value: float
    regime: Optional[int] = None


class EquityCurveResponse(BaseModel):
    """Paper trading equity curve."""

    curve: List[EquityCurvePoint] = Field(default_factory=list)
    total_points: int = 0


class StreamStatusResponse(BaseModel):
    """SSE stream manager status."""

    is_running: bool = False
    connected_clients: int = 0
    total_connections: int = 0
    total_events_broadcast: int = 0
    clients: List[Dict[str, Any]] = Field(default_factory=list)


class Phase4StatusResponse(BaseModel):
    """Overall Phase 4 system status."""

    event_bus: Dict[str, Any] = Field(default_factory=dict)
    scheduler: Dict[str, Any] = Field(default_factory=dict)
    alert_engine: Dict[str, Any] = Field(default_factory=dict)
    stream_manager: Dict[str, Any] = Field(default_factory=dict)
    paper_trading: Dict[str, Any] = Field(default_factory=dict)
