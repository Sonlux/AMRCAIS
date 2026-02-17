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

/* ─── Macro Data ─────────────────────────────────────── */

export interface MacroDataPoint {
  date: string;
  value: number;
}

export interface MacroDataResponse {
  indicator: string;
  series: MacroDataPoint[];
  total_points: number;
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
  drawdown_curve: DrawdownPoint[];
  trade_log: TradeLogEntry[];
  regime_returns: RegimeReturnEntry[];
  strategy: string;
  initial_capital: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
}

export interface TradeLogEntry {
  date: string;
  action: string;
  regime: number;
  regime_name: string;
  allocations: Record<string, number>;
  portfolio_value: number;
  daily_return: number;
}

/* ─── Meta-Learning ───────────────────────────────────────── */
export interface ClassifierAccuracyPoint {
  date: string;
  accuracy: number;
  classifier: string;
}

export interface ClassifierAccuracyResponse {
  classifiers: string[];
  series: ClassifierAccuracyPoint[];
  window: number;
}
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

/* ─── Phase 2: Intelligence Expansion ─────────────────────── */

export interface TimeframeRegimeResponse {
  regime: number;
  regime_name: string;
  confidence: number;
  disagreement: number;
}

export interface MultiTimeframeResponse {
  daily: TimeframeRegimeResponse;
  weekly: TimeframeRegimeResponse;
  monthly: TimeframeRegimeResponse;
  conflict_detected: boolean;
  highest_conviction: string;
  trade_signal: string;
  agreement_score: number;
}

export interface TransitionForecastResponse {
  current_regime: number;
  horizon_days: number;
  hmm_probs: Record<string, number>;
  indicator_probs: Record<string, number>;
  blended_probs: Record<string, number>;
  leading_indicators: Record<string, unknown>;
  transition_risk: string;
  most_likely_next: number;
  most_likely_next_name: string;
  confidence: number;
}

export interface GrangerLinkResponse {
  source: string;
  target: string;
  p_value: number;
  lag: number;
  f_stat: number;
}

export interface SpilloverResponse {
  total_spillover_index: number;
  directional_to: Record<string, number>;
  directional_from: Record<string, number>;
  net_spillover: Record<string, number>;
  pairwise: Record<string, Record<string, number>>;
  assets: string[];
}

export interface NetworkGraphResponse {
  nodes: { id: string; size: number }[];
  edges: { source: string; target: string; weight: number }[];
}

export interface ContagionAnalysisResponse {
  signal: ModuleSignalResponse;
  granger_network: GrangerLinkResponse[];
  spillover: SpilloverResponse;
  network_graph: NetworkGraphResponse;
  contagion_flags: string[];
  n_significant_links: number;
  network_density: number;
}

export interface DecayCurvePoint {
  day: number;
  impact: number;
  is_stale: boolean;
}

export interface DecayCurveResponse {
  curves: Record<string, DecayCurvePoint[]>;
}

export interface DecaySurpriseResponse {
  indicator: string;
  surprise: number;
  release_date: string;
  half_life_days: number;
  initial_weight: number;
  regime_at_release: number;
}

export interface SurpriseIndexResponse {
  index: number;
  direction: string;
  components: Record<string, number>;
  active_surprises: number;
  total_historical: number;
}

export interface NarrativeResponse {
  headline: string;
  regime_section: string;
  signal_section: string;
  risk_section: string;
  positioning_section: string;
  full_text: string;
  data_sources: string[];
}

/* ─── Phase 3: Prediction Engine ──────────────────────────── */

export interface ReturnForecastResponse {
  asset: string;
  regime: number;
  expected_return: number;
  volatility: number;
  r_squared_regime: number;
  r_squared_static: number;
  r_squared_improvement: number;
  kelly_fraction: number;
  factor_contributions: Record<string, number>;
  confidence: number;
}

export interface ReturnForecastsResponse {
  current_regime: number;
  regime_name: string;
  forecasts: ReturnForecastResponse[];
}

export interface RegimeCoefficientsResponse {
  asset: string;
  regimes: Record<string, Record<string, number>>;
}

export interface ScenarioVaRResponse {
  scenario_regime: number;
  scenario_name: string;
  probability: number;
  var_95: number;
  cvar_95: number;
}

export interface HedgeRecommendationResponse {
  hedge_type: string;
  instrument: string;
  rationale: string;
  priority: number;
}

export interface TailRiskResponse {
  current_regime: number;
  weighted_var: number;
  weighted_cvar: number;
  scenarios: ScenarioVaRResponse[];
  worst_scenario: string;
  worst_scenario_var: number;
  tail_risk_driver: string;
  hedge_recommendations: HedgeRecommendationResponse[];
  portfolio_weights: Record<string, number>;
}

export interface RegimeAllocationResponse {
  regime: number;
  regime_name: string;
  probability: number;
  weights: Record<string, number>;
  expected_return: number;
  expected_volatility: number;
}

export interface PortfolioOptimizationResponse {
  current_regime: number;
  blended_weights: Record<string, number>;
  regime_allocations: RegimeAllocationResponse[];
  rebalance_trigger: boolean;
  rebalance_reason: string;
  transaction_cost_estimate: number;
  expected_return: number;
  expected_volatility: number;
  sharpe_ratio: number;
  max_drawdown_constraint: number;
}

export interface AlphaSignalResponse {
  anomaly_type: string;
  direction: string;
  rationale: string;
  strength: number;
  confidence: number;
  holding_period_days: number;
  historical_win_rate: number;
  regime: number;
}

export interface AlphaSignalsResponse {
  signals: AlphaSignalResponse[];
  composite_score: number;
  top_signal: string;
  regime: number;
  n_active_anomalies: number;
  regime_context: string;
}

/* ─── Phase 4: Real-Time + Execution ─────────────────────── */

export interface Phase4StatusResponse {
  event_bus: Record<string, unknown>;
  scheduler: Record<string, unknown>;
  alert_engine: Record<string, unknown>;
  stream_manager: Record<string, unknown>;
  paper_trading: Record<string, unknown>;
}

export interface EventResponse {
  event_type: string;
  data: Record<string, unknown>;
  timestamp: string;
  event_id: string;
  source: string;
}

export interface EventsResponse {
  events: EventResponse[];
  total: number;
}

export interface AlertResponse {
  alert_id: string;
  alert_type: string;
  severity: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  data: Record<string, unknown>;
}

export interface AlertsResponse {
  alerts: AlertResponse[];
  total: number;
  unacknowledged: number;
}

export interface AlertConfigItem {
  enabled: boolean;
  cooldown_seconds: number;
  threshold: number | null;
}

export interface AlertConfigResponse {
  configs: Record<string, AlertConfigItem>;
}

export interface PositionResponse {
  asset: string;
  shares: number;
  avg_price: number;
  current_value: number;
  weight: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface PortfolioSummaryResponse {
  total_equity: number;
  cash: number;
  positions_value: number;
  num_positions: number;
  positions: PositionResponse[];
  total_return_pct: number;
  rebalance_count: number;
  last_rebalance: string | null;
  current_regime: number;
}

export interface PerformanceMetricsResponse {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  avg_daily_return: number;
  volatility: number;
  calmar_ratio: number;
}

export interface EquityCurvePointP4 {
  date: string;
  equity: number;
  regime: number | null;
}

export interface EquityCurveResponseP4 {
  curve: EquityCurvePointP4[];
  total_points: number;
}

export interface RegimeAttributionEntry {
  regime: number;
  regime_name: string;
  pnl: number;
  pnl_pct: number;
  days: number;
}

export interface RegimeAttributionResponse {
  attribution: RegimeAttributionEntry[];
  total_pnl: number;
}

export interface PaperOrderResponse {
  order_id: string;
  timestamp: string;
  asset: string;
  side: string;
  shares: number;
  price: number;
  value: number;
  regime: number;
  reason: string;
}

/* ─── Phase 5: Network Effects + Moat ────────────────────── */

export interface Phase5StatusResponse {
  knowledge_base: Record<string, unknown>;
  alt_data: Record<string, unknown>;
  research_publisher: Record<string, unknown>;
  user_manager: Record<string, unknown>;
}

export interface KnowledgeSummaryResponse {
  total_transitions: number;
  total_anomalies: number;
  total_macro_impacts: number;
  regime_coverage: Record<string, number>;
}

export interface TransitionRecord {
  id: string;
  from_regime: number;
  to_regime: number;
  confidence: number;
  disagreement: number;
  timestamp: string;
  indicators: Record<string, number>;
}

export interface TransitionsResponse {
  transitions: TransitionRecord[];
  total: number;
}

export interface PatternMatchResponse {
  transition: TransitionRecord;
  similarity: number;
  days_ago: number;
  outcome_summary: string;
}

export interface PatternSearchResponse {
  matches: PatternMatchResponse[];
  query_indicators: Record<string, number>;
}

export interface AnomalyRecord {
  id: string;
  anomaly_type: string;
  asset_pair: string;
  regime: number;
  z_score: number;
  expected_value: number;
  actual_value: number;
  timestamp: string;
}

export interface AnomaliesResponse {
  anomalies: AnomalyRecord[];
  total: number;
}

export interface AnomalyStatsResponse {
  stats: Record<string, unknown>;
}

export interface MacroImpactStatsResponse {
  stats: Record<string, unknown>;
}

export interface ResearchReportResponse {
  id: string;
  report_type: string;
  title: string;
  content: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ReportListResponse {
  reports: ResearchReportResponse[];
  total: number;
}

export interface ResearchPublisherSummaryResponse {
  total_reports: number;
  report_types: Record<string, number>;
  last_report: string | null;
}

export interface UserResponse {
  user_id: string;
  name: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
  preferences: Record<string, unknown>;
  api_key?: string;
}

export interface UserListResponse {
  users: UserResponse[];
  total: number;
}

export interface UserManagerSummaryResponse {
  total_users: number;
  active_users: number;
  roles: Record<string, number>;
}

export interface AnnotationResponse {
  id: string;
  user_id: string;
  content: string;
  context: Record<string, unknown>;
  created_at: string;
}

export interface AnnotationsResponse {
  annotations: AnnotationResponse[];
  total: number;
}
