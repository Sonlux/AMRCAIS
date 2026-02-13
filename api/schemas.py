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
    regime: int


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
