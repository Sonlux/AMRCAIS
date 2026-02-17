/**
 * Typed fetch client for the AMRCAIS FastAPI backend.
 *
 * Security features:
 * - Automatic CSRF token management for POST requests
 * - Input sanitization before sending to API
 * - No sensitive data stored in browser (no localStorage for secrets)
 * - Typed responses matching Pydantic schemas
 * - Errors thrown as plain Error objects for TanStack Query
 */

import { API_BASE_URL } from "./constants";
import type {
  BacktestRequest,
  BacktestResultResponse,
  ClassifierAccuracyResponse,
  ClassifierVotesResponse,
  CorrelationMatrixResponse,
  DisagreementResponse,
  HealthCheckResponse,
  HealthResponse,
  MacroDataResponse,
  ModuleAnalysisResponse,
  ModuleSummaryResponse,
  PerformanceResponse,
  PriceResponse,
  RecalibrationResponse,
  RegimeHistoryResponse,
  RegimeResponse,
  SignalHistoryResponse,
  StatusResponse,
  TransitionMatrixResponse,
  VolSurfaceDataResponse,
  WeightHistoryResponse,
  WeightsResponse,
  YieldCurveDataResponse,
  // Phase 2
  TransitionForecastResponse,
  MultiTimeframeResponse,
  ContagionAnalysisResponse,
  SpilloverResponse,
  SurpriseIndexResponse,
  DecayCurveResponse,
  DecaySurpriseResponse,
  NarrativeResponse,
  // Phase 3
  ReturnForecastResponse,
  ReturnForecastsResponse,
  TailRiskResponse,
  PortfolioOptimizationResponse,
  AlphaSignalsResponse,
  // Phase 4
  Phase4StatusResponse,
  EventsResponse,
  AlertsResponse,
  AlertConfigResponse,
  PortfolioSummaryResponse,
  PerformanceMetricsResponse,
  EquityCurveResponseP4,
  RegimeAttributionResponse,
  PaperOrderResponse,
  // Phase 5
  Phase5StatusResponse,
  KnowledgeSummaryResponse,
  TransitionsResponse,
  PatternSearchResponse,
  AnomaliesResponse,
  AnomalyStatsResponse,
  ResearchPublisherSummaryResponse,
  ReportListResponse,
  ResearchReportResponse,
  UserManagerSummaryResponse,
  UserListResponse,
  AnnotationsResponse,
} from "./types";

/* ─── CSRF Token Management ────────────────────────────────── */

let _csrfToken: string | null = null;

/**
 * Fetch a fresh CSRF token from the API.
 * Tokens are short-lived and should be refreshed periodically.
 */
async function fetchCsrfToken(): Promise<string> {
  const res = await fetch(`${API_BASE_URL}/api/csrf-token`, {
    credentials: "include",
  });
  if (!res.ok) throw new Error("Failed to obtain CSRF token");
  const data = await res.json();
  const token: string = data.csrf_token;
  _csrfToken = token;
  return token;
}

/**
 * Get a cached CSRF token or fetch a new one.
 */
async function getCsrfToken(): Promise<string> {
  if (_csrfToken) return _csrfToken;
  return fetchCsrfToken();
}

/* ─── Input Sanitization ───────────────────────────────────── */

/**
 * Sanitize a string to prevent XSS in query parameters.
 * Strips HTML tags, null bytes, and limits length.
 */
function sanitize(value: string, maxLen = 200): string {
  return value
    .replace(/[<>]/g, "") // Strip angle brackets (basic XSS)
    .replace(/\x00/g, "") // Null bytes
    .replace(/javascript:/gi, "") // JS protocol
    .trim()
    .slice(0, maxLen);
}

/**
 * Validate a date string matches YYYY-MM-DD format.
 */
function isValidDate(d: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(d) && !isNaN(Date.parse(d));
}

/* ─── Generic fetcher ─────────────────────────────────────── */

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    credentials: "include",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  // Obtain CSRF token before state-changing request
  const csrfToken = await getCsrfToken();

  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": csrfToken,
    },
    body: JSON.stringify(body),
  });

  // If CSRF token was rejected, refresh and retry once
  if (res.status === 403) {
    _csrfToken = null;
    const freshToken = await getCsrfToken();
    const retry = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": freshToken,
      },
      body: JSON.stringify(body),
    });
    if (!retry.ok) {
      const text = await retry.text().catch(() => "");
      throw new Error(`API ${retry.status}: ${text || retry.statusText}`);
    }
    return retry.json() as Promise<T>;
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/* ─── System ──────────────────────────────────────────────── */

export const fetchHealth = () => get<HealthCheckResponse>("/api/health");
export const fetchStatus = () => get<StatusResponse>("/api/status");

/* ─── Regime ──────────────────────────────────────────────── */

export const fetchCurrentRegime = () =>
  get<RegimeResponse>("/api/regime/current");

export const fetchRegimeHistory = (start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start && isValidDate(start)) params.set("start", sanitize(start));
  if (end && isValidDate(end)) params.set("end", sanitize(end));
  const qs = params.toString();
  return get<RegimeHistoryResponse>(`/api/regime/history${qs ? `?${qs}` : ""}`);
};

export const fetchClassifierVotes = () =>
  get<ClassifierVotesResponse>("/api/regime/classifiers");

export const fetchTransitionMatrix = () =>
  get<TransitionMatrixResponse>("/api/regime/transitions");

export const fetchDisagreement = (start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start && isValidDate(start)) params.set("start", sanitize(start));
  if (end && isValidDate(end)) params.set("end", sanitize(end));
  const qs = params.toString();
  return get<DisagreementResponse>(
    `/api/regime/disagreement${qs ? `?${qs}` : ""}`,
  );
};

/* ─── Modules ─────────────────────────────────────────────── */

export const fetchModuleSummary = () =>
  get<ModuleSummaryResponse>("/api/modules/summary");

export const fetchModuleAnalysis = (name: string) =>
  get<ModuleAnalysisResponse>(
    `/api/modules/${encodeURIComponent(sanitize(name, 50))}/analyze`,
  );

export const fetchModuleHistory = (name: string) =>
  get<SignalHistoryResponse>(
    `/api/modules/${encodeURIComponent(sanitize(name, 50))}/history`,
  );

/* ─── Surface / 3D ────────────────────────────────────────── */

export const fetchYieldCurveData = () =>
  get<YieldCurveDataResponse>("/api/modules/yield_curve/curve");

export const fetchVolSurface = () =>
  get<VolSurfaceDataResponse>("/api/modules/options/surface");

/* ─── Data ────────────────────────────────────────────────── */

export const fetchAssets = () => get<string[]>("/api/data/assets");

export const fetchPrices = (asset: string, start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start && isValidDate(start)) params.set("start", sanitize(start));
  if (end && isValidDate(end)) params.set("end", sanitize(end));
  const qs = params.toString();
  return get<PriceResponse>(
    `/api/data/prices/${encodeURIComponent(sanitize(asset, 10))}${qs ? `?${qs}` : ""}`,
  );
};

export const fetchCorrelations = (window = 60) =>
  get<CorrelationMatrixResponse>(`/api/data/correlations?window=${window}`);

export const fetchMacroData = (indicator: string) =>
  get<MacroDataResponse>(
    `/api/data/macro/${encodeURIComponent(sanitize(indicator, 30))}`,
  );

/* ─── Backtest ────────────────────────────────────────────── */

export const runBacktest = (req: BacktestRequest) =>
  post<BacktestResultResponse>("/api/backtest/run", req);

export const fetchBacktestResults = () =>
  get<BacktestResultResponse[]>("/api/backtest/results");

export const fetchBacktestResult = (id: string) =>
  get<BacktestResultResponse>(
    `/api/backtest/results/${encodeURIComponent(sanitize(id, 50))}`,
  );

/* ─── Meta-Learning ───────────────────────────────────────── */

export const fetchPerformance = () =>
  get<PerformanceResponse>("/api/meta/performance");

export const fetchClassifierAccuracy = (window = 30) =>
  get<ClassifierAccuracyResponse>(`/api/meta/accuracy?window=${window}`);

export const fetchMetaDisagreement = () =>
  get<DisagreementResponse>("/api/meta/disagreement");

export const fetchWeights = () => get<WeightsResponse>("/api/meta/weights");

export const fetchWeightHistory = () =>
  get<WeightHistoryResponse>("/api/meta/weights/history");

export const fetchRecalibrations = () =>
  get<RecalibrationResponse>("/api/meta/recalibrations");

export const fetchSystemHealth = () => get<HealthResponse>("/api/meta/health");

/* ─── Phase 2: Intelligence Expansion ─────────────────────── */

export const fetchTransitionForecast = (horizon = 30) =>
  get<TransitionForecastResponse>(
    `/api/phase2/transition-forecast?horizon=${horizon}`,
  );

export const fetchMultiTimeframe = () =>
  get<MultiTimeframeResponse>("/api/phase2/multi-timeframe");

export const fetchContagionAnalysis = () =>
  get<ContagionAnalysisResponse>("/api/phase2/contagion/analyze");

export const fetchSpillover = () =>
  get<SpilloverResponse>("/api/phase2/contagion/spillover");

export const fetchSurpriseIndex = () =>
  get<SurpriseIndexResponse>("/api/phase2/surprise-decay/index");

export const fetchDecayCurves = (daysForward = 30) =>
  get<DecayCurveResponse>(
    `/api/phase2/surprise-decay/curves?days_forward=${daysForward}`,
  );

export const fetchActiveSurprises = () =>
  get<DecaySurpriseResponse[]>("/api/phase2/surprise-decay/active");

export const fetchNarrative = () =>
  get<NarrativeResponse>("/api/phase2/narrative");

/* ─── Phase 3: Prediction Engine ──────────────────────────── */

export const fetchReturnForecasts = () =>
  get<ReturnForecastsResponse>("/api/phase3/return-forecast");

export const fetchReturnForecast = (asset: string) =>
  get<ReturnForecastResponse>(
    `/api/phase3/return-forecast/${encodeURIComponent(sanitize(asset, 10))}`,
  );

export const fetchTailRisk = () =>
  get<TailRiskResponse>("/api/phase3/tail-risk");

export const fetchPortfolioOptimize = () =>
  get<PortfolioOptimizationResponse>("/api/phase3/portfolio-optimize");

export const fetchAlphaSignals = () =>
  get<AlphaSignalsResponse>("/api/phase3/alpha-signals");

/* ─── Phase 4: Real-Time + Execution ─────────────────────── */

export const fetchPhase4Status = () =>
  get<Phase4StatusResponse>("/api/phase4/status");

export const fetchEvents = (limit = 50, eventType?: string) => {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (eventType) params.set("event_type", sanitize(eventType, 50));
  return get<EventsResponse>(`/api/phase4/events?${params.toString()}`);
};

export const fetchAlerts = (
  limit = 50,
  severity?: string,
  unacknowledgedOnly = false,
) => {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (severity) params.set("severity", sanitize(severity, 20));
  if (unacknowledgedOnly) params.set("unacknowledged_only", "true");
  return get<AlertsResponse>(`/api/phase4/alerts?${params.toString()}`);
};

export const acknowledgeAlert = (alertId: string) =>
  post<{ status: string; alert_id: string }>(
    `/api/phase4/alerts/acknowledge?alert_id=${encodeURIComponent(sanitize(alertId, 100))}`,
    {},
  );

export const fetchAlertConfig = () =>
  get<AlertConfigResponse>("/api/phase4/alerts/config");

export const fetchPortfolio = () =>
  get<PortfolioSummaryResponse>("/api/phase4/portfolio");

export const fetchPortfolioMetrics = () =>
  get<PerformanceMetricsResponse>("/api/phase4/portfolio/metrics");

export const fetchPortfolioEquity = (limit = 500) =>
  get<EquityCurveResponseP4>(`/api/phase4/portfolio/equity?limit=${limit}`);

export const fetchRegimeAttribution = () =>
  get<RegimeAttributionResponse>("/api/phase4/portfolio/attribution");

export const fetchTrades = (limit = 50, asset?: string) => {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  if (asset) params.set("asset", sanitize(asset, 10));
  return get<{ orders: PaperOrderResponse[]; total: number }>(
    `/api/phase4/trades?${params.toString()}`,
  );
};

/* ─── Phase 5: Network Effects + Moat ────────────────────── */

export const fetchPhase5Status = () =>
  get<Phase5StatusResponse>("/api/phase5/status");

export const fetchKnowledgeSummary = () =>
  get<KnowledgeSummaryResponse>("/api/phase5/knowledge/summary");

export const fetchTransitions = (limit = 50) =>
  get<TransitionsResponse>(`/api/phase5/transitions?limit=${limit}`);

export const searchPatterns = (indicators: Record<string, number>) =>
  post<PatternSearchResponse>("/api/phase5/transitions/search", {
    current_indicators: indicators,
    top_k: 10,
    min_similarity: 0.3,
  });

export const fetchAnomalies = (limit = 50) =>
  get<AnomaliesResponse>(`/api/phase5/anomalies?limit=${limit}`);

export const fetchAnomalyStats = () =>
  get<AnomalyStatsResponse>("/api/phase5/anomalies/stats");

export const fetchResearchSummary = () =>
  get<ResearchPublisherSummaryResponse>("/api/phase5/research/summary");

export const fetchResearchReports = (limit = 20) =>
  get<ReportListResponse>(`/api/phase5/research/reports?limit=${limit}`);

export const generateCaseStudy = (fromRegime: number, toRegime: number) =>
  post<ResearchReportResponse>("/api/phase5/research/case-study", {
    from_regime: fromRegime,
    to_regime: toRegime,
    limit: 10,
  });

export const fetchUsersSummary = () =>
  get<UserManagerSummaryResponse>("/api/phase5/users/summary");

export const fetchUsers = () => get<UserListResponse>("/api/phase5/users");

export const fetchAnnotations = (limit = 50) =>
  get<AnnotationsResponse>(`/api/phase5/annotations?limit=${limit}`);
