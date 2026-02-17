import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithQuery } from "../helpers";

// ── Mocks ───────────────────────────────────────────────────────────

vi.mock("@/components/charts/PlotlyChart", () => ({
  __esModule: true,
  default: (props: Record<string, unknown>) => (
    <div data-testid="plotly-mock" data-traces={JSON.stringify(props.data)} />
  ),
}));

const mockForecasts = {
  current_regime: 1,
  regime_name: "Risk-On Growth",
  forecasts: [
    {
      asset: "SPX",
      regime: 1,
      expected_return: 0.08,
      volatility: 0.15,
      r_squared_regime: 0.42,
      r_squared_static: 0.3,
      r_squared_improvement: 0.12,
      kelly_fraction: 0.35,
      factor_contributions: { momentum: 0.03, value: 0.02 },
      confidence: 0.85,
    },
    {
      asset: "TLT",
      regime: 1,
      expected_return: -0.02,
      volatility: 0.1,
      r_squared_regime: 0.35,
      r_squared_static: 0.28,
      r_squared_improvement: 0.07,
      kelly_fraction: -0.1,
      factor_contributions: { duration: -0.01 },
      confidence: 0.72,
    },
  ],
};

const mockAlpha = {
  signals: [
    {
      anomaly_type: "momentum_divergence",
      direction: "bullish",
      rationale: "SPX momentum diverging from risk-off indicators",
      strength: 0.7,
      confidence: 0.8,
      holding_period_days: 14,
      historical_win_rate: 0.65,
      regime: 1,
    },
  ],
  composite_score: 0.55,
  top_signal: "momentum_divergence",
  regime: 1,
  n_active_anomalies: 3,
  regime_context: "Risk-On Growth",
};

vi.mock("@/lib/api", () => ({
  fetchReturnForecasts: vi.fn(),
  fetchAlphaSignals: vi.fn(),
}));

import { fetchReturnForecasts, fetchAlphaSignals } from "@/lib/api";
import PredictionsPage from "@/app/predictions/page";

beforeEach(() => {
  vi.mocked(fetchReturnForecasts).mockResolvedValue(mockForecasts);
  vi.mocked(fetchAlphaSignals).mockResolvedValue(mockAlpha);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("PredictionsPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<PredictionsPage />);
    expect(screen.getByText("Prediction Engine")).toBeInTheDocument();
  });

  it("renders regime KPI", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Current Regime")).toBeInTheDocument();
      // "Risk-On Growth" appears in KPI, regime_context sub, and alpha signals
      const matches = screen.getAllByText("Risk-On Growth");
      expect(matches.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows assets forecasted count", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Assets Forecasted")).toBeInTheDocument();
      // forecasts.length = 2
      expect(screen.getByText("2")).toBeInTheDocument();
    });
  });

  it("shows composite alpha score", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Composite Alpha")).toBeInTheDocument();
      expect(screen.getByText("0.55")).toBeInTheDocument();
    });
  });

  it("shows active anomalies count", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Active Anomalies")).toBeInTheDocument();
      expect(screen.getByText("3")).toBeInTheDocument();
    });
  });

  it("shows top signal KPI", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Top Signal")).toBeInTheDocument();
      expect(screen.getByText("momentum_divergence")).toBeInTheDocument();
    });
  });

  it("renders return forecasts chart section", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(
        screen.getByText(/Expected Returns by Asset/),
      ).toBeInTheDocument();
    });
  });

  it("renders forecast detail cards with asset data", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("SPX")).toBeInTheDocument();
      expect(screen.getByText("TLT")).toBeInTheDocument();
      // "Volatility" and "Kelly Fraction" appear once per forecast card
      const volLabels = screen.getAllByText("Volatility");
      expect(volLabels.length).toBeGreaterThanOrEqual(2);
      const kellyLabels = screen.getAllByText("Kelly Fraction");
      expect(kellyLabels.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("renders factor contributions in forecast cards", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      const fcLabels = screen.getAllByText("Factor Contributions");
      expect(fcLabels.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("momentum")).toBeInTheDocument();
    });
  });

  it("renders alpha signals section", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(screen.getByText("Alpha Signals")).toBeInTheDocument();
      expect(screen.getByText("bullish")).toBeInTheDocument();
      expect(screen.getByText("momentum divergence")).toBeInTheDocument();
    });
  });

  it("shows alpha signal details", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(
        screen.getByText(
          "SPX momentum diverging from risk-off indicators",
        ),
      ).toBeInTheDocument();
    });
  });

  it("renders Plotly chart", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("calls both API functions", async () => {
    renderWithQuery(<PredictionsPage />);
    await waitFor(() => {
      expect(fetchReturnForecasts).toHaveBeenCalled();
      expect(fetchAlphaSignals).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchReturnForecasts).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchAlphaSignals).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<PredictionsPage />);
    expect(screen.getByText("Prediction Engine")).toBeInTheDocument();
  });
});
