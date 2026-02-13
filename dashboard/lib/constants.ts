/** AMRCAIS dashboard constants — regime colors, names, chart config. */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* ─── Regimes ─────────────────────────────────────────────── */

export const REGIME_NAMES: Record<number, string> = {
  1: "Risk-On Growth",
  2: "Risk-Off Crisis",
  3: "Stagflation",
  4: "Disinflationary Boom",
};

export const REGIME_COLORS: Record<number, string> = {
  1: "#22c55e",
  2: "#ef4444",
  3: "#f59e0b",
  4: "#3b82f6",
};

export const REGIME_BG_COLORS: Record<number, string> = {
  1: "rgba(34,197,94,0.12)",
  2: "rgba(239,68,68,0.12)",
  3: "rgba(245,158,11,0.12)",
  4: "rgba(59,130,246,0.12)",
};

/* ─── Signals ─────────────────────────────────────────────── */

export const SIGNAL_COLORS: Record<string, string> = {
  bullish: "#22c55e",
  bearish: "#ef4444",
  neutral: "#6b7280",
  cautious: "#f59e0b",
};

export const SIGNAL_ICONS: Record<string, string> = {
  bullish: "▲",
  bearish: "▼",
  neutral: "►",
  cautious: "◆",
};

/* ─── Assets ──────────────────────────────────────────────── */

export const TRACKED_ASSETS = [
  "SPX",
  "TLT",
  "GLD",
  "DXY",
  "WTI",
  "VIX",
] as const;

/* ─── Module names ────────────────────────────────────────── */

export const MODULE_NAMES: Record<string, string> = {
  macro: "Macro Events",
  yield_curve: "Yield Curve",
  options: "Options Surface",
  factors: "Factor Exposure",
  correlations: "Correlations",
};

/* ─── Chart defaults ──────────────────────────────────────── */

export const PLOTLY_DARK_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#8888a0", family: "var(--font-geist-sans), system-ui" },
  margin: { t: 30, r: 20, b: 40, l: 50 },
  xaxis: { gridcolor: "#1a1a28", zerolinecolor: "#2a2a3a" },
  yaxis: { gridcolor: "#1a1a28", zerolinecolor: "#2a2a3a" },
};

export const PLOTLY_CONFIG: Record<string, unknown> = {
  displayModeBar: false,
  responsive: true,
};

/* ─── Refresh intervals ──────────────────────────────────── */

export const REFETCH_INTERVAL = 60_000; // 60 s
export const STALE_TIME = 30_000; // 30 s
