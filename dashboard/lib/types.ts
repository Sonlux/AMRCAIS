/** TypeScript interfaces matching the FastAPI Pydantic schemas. */

/* ─── Regime ──────────────────────────────────────────────── */

export interface RegimeResponse {
  regime: number;
  regime_name: string;
  confidence: number;
  disagreement: number;
  classifier_votes: Record<string, number>;
  probabilities: Record<string, number>;
  transition_warning: boolean;
  timestamp: string;
}

export interface RegimeHistoryPoint {
  date: string;
  regime: number;
  regime_name: string;
  confidence: number;
  disagreement: number;
}

export interface RegimeHistoryResponse {
  history: RegimeHistoryPoint[];
  total_points: number;
  start_date: string;
  end_date: string;
}

export interface ClassifierVoteEntry {
  classifier: string;
  regime: number;
  confidence: number;
  weight: number;
}

export interface ClassifierVotesResponse {
  votes: ClassifierVoteEntry[];
  ensemble_regime: number;
  ensemble_confidence: number;
  weights: Record<string, number>;
}

export interface TransitionMatrixResponse {
  matrix: number[][];
  regime_names: string[];
  total_transitions: number;
}

export interface DisagreementPoint {
  date: string;
  disagreement: number;
  threshold_exceeded: boolean;
}

export interface DisagreementResponse {
  series: DisagreementPoint[];
  avg_disagreement: number;
  max_disagreement: number;
  threshold: number;
}

/* ─── Modules ─────────────────────────────────────────────── */

export interface ModuleSignalResponse {
  module: string;
  signal: string;
  strength: number;
  confidence: number;
  explanation: string;
  regime_context: string;
}

export interface ModuleSummaryResponse {
  signals: ModuleSignalResponse[];
  current_regime: number;
  regime_name: string;
  timestamp: string;
}

export interface ModuleAnalysisResponse {
  module: string;
  signal: ModuleSignalResponse;
  raw_metrics: Record<string, unknown>;
  regime_parameters: Record<string, unknown>;
}

export interface SignalHistoryResponse {
  module: string;
  history: { date: string; signal: string; strength: number; regime: number }[];
}

/* ─── Data ────────────────────────────────────────────────── */

export interface PricePoint {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number;
  volume: number | null;
}

export interface PriceResponse {
  asset: string;
  prices: PricePoint[];
  total_points: number;
}

export interface CorrelationMatrixResponse {
  assets: string[];
  matrix: number[][];
  window: number;
}

/* ─── Backtest ────────────────────────────────────────────── */

export interface BacktestRequest {
  start_date: string;
  end_date: string;
  strategy: string;
  initial_capital: number;
  assets: string[];
}

export interface EquityPoint {
  date: string;
  value: number;
  regime: number | null;
}

export interface RegimeReturnEntry {
  regime: number;
  regime_name: string;
  strategy_return: number;
  benchmark_return: number;
  days: number;
  hit_rate: number;
}

export interface BacktestResultResponse {
  id: string;
  start_date: string;
  end_date: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  benchmark_return: number;
  equity_curve: EquityPoint[];
  regime_returns: RegimeReturnEntry[];
  strategy: string;
  initial_capital: number;
}

/* ─── Meta-Learning ───────────────────────────────────────── */

export interface PerformanceResponse {
  stability_score: number;
  stability_rating: string;
  transition_count: number;
  avg_disagreement: number;
  high_disagreement_days: number;
  total_classifications: number;
  regime_distribution: Record<string, number>;
}

export interface WeightsResponse {
  weights: Record<string, number>;
  is_adaptive: boolean;
}

export interface WeightHistoryResponse {
  history: { date: string; weights: Record<string, number> }[];
}

export interface RecalibrationEvent {
  date: string;
  trigger_reason: string;
  severity: number;
  urgency: string;
  recommendations: string[];
}

export interface RecalibrationResponse {
  events: RecalibrationEvent[];
  total_recalibrations: number;
  last_recalibration: string | null;
}

export interface HealthResponse {
  system_status: string;
  needs_recalibration: boolean;
  urgency: string;
  severity: number;
  reasons: string[];
  performance_30d: Record<string, unknown>;
  performance_7d: Record<string, unknown>;
  alerts: Record<string, unknown>;
  adaptive_weights: Record<string, number> | null;
}

/* ─── Surface / 3D Charts ─────────────────────────────────── */

export interface YieldCurveDataResponse {
  tenors: number[];
  yields: number[];
  curve_shape: string;
  slope_2_10: number | null;
  slope_3m_10: number | null;
  curvature: number | null;
  regime: number;
  regime_name: string;
  timestamp: string;
}

export interface VolSurfaceDataResponse {
  moneyness: number[];
  expiry_days: number[];
  iv_grid: number[][];
  atm_vol: number;
  regime: number;
  regime_name: string;
  timestamp: string;
}

/* ─── System ──────────────────────────────────────────────── */

export interface StatusResponse {
  is_initialized: boolean;
  current_regime: number | null;
  confidence: number;
  disagreement: number;
  modules_loaded: string[];
  uptime_seconds: number | null;
}

export interface HealthCheckResponse {
  status: string;
  version: string;
  timestamp: string;
}
