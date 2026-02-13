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
  ClassifierVotesResponse,
  CorrelationMatrixResponse,
  DisagreementResponse,
  HealthCheckResponse,
  HealthResponse,
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
  WeightHistoryResponse,
  WeightsResponse,
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
  _csrfToken = data.csrf_token;
  return _csrfToken;
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
  get<ModuleAnalysisResponse>(`/api/modules/${encodeURIComponent(sanitize(name, 50))}/analyze`);

export const fetchModuleHistory = (name: string) =>
  get<SignalHistoryResponse>(`/api/modules/${encodeURIComponent(sanitize(name, 50))}/history`);

/* ─── Data ────────────────────────────────────────────────── */

export const fetchAssets = () => get<string[]>("/api/data/assets");

export const fetchPrices = (asset: string, start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start && isValidDate(start)) params.set("start", sanitize(start));
  if (end && isValidDate(end)) params.set("end", sanitize(end));
  const qs = params.toString();
  return get<PriceResponse>(`/api/data/prices/${encodeURIComponent(sanitize(asset, 10))}${qs ? `?${qs}` : ""}`);
};

export const fetchCorrelations = (window = 60) =>
  get<CorrelationMatrixResponse>(`/api/data/correlations?window=${window}`);

/* ─── Backtest ────────────────────────────────────────────── */

export const runBacktest = (req: BacktestRequest) =>
  post<BacktestResultResponse>("/api/backtest/run", req);

export const fetchBacktestResults = () =>
  get<BacktestResultResponse[]>("/api/backtest/results");

export const fetchBacktestResult = (id: string) =>
  get<BacktestResultResponse>(`/api/backtest/results/${encodeURIComponent(sanitize(id, 50))}`);

/* ─── Meta-Learning ───────────────────────────────────────── */

export const fetchPerformance = () =>
  get<PerformanceResponse>("/api/meta/performance");

export const fetchWeights = () => get<WeightsResponse>("/api/meta/weights");

export const fetchWeightHistory = () =>
  get<WeightHistoryResponse>("/api/meta/weights/history");

export const fetchRecalibrations = () =>
  get<RecalibrationResponse>("/api/meta/recalibrations");

export const fetchSystemHealth = () => get<HealthResponse>("/api/meta/health");
