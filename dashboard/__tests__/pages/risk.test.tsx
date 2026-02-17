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

const mockTailRisk = {
  current_regime: 1,
  weighted_var: -3.5,
  weighted_cvar: -5.2,
  scenarios: [
    {
      scenario_regime: 1,
      scenario_name: "Risk-On Growth",
      probability: 0.6,
      var_95: -2.0,
      cvar_95: -3.5,
    },
    {
      scenario_regime: 2,
      scenario_name: "Risk-Off Crisis",
      probability: 0.25,
      var_95: -8.0,
      cvar_95: -12.0,
    },
  ],
  worst_scenario: "Risk-Off Crisis",
  worst_scenario_var: -8.0,
  tail_risk_driver: "VIX spike",
  hedge_recommendations: [
    {
      hedge_type: "put_spread",
      instrument: "SPY 400/380 Put Spread",
      rationale: "Hedge against tail risk in equity drawdown",
      priority: 1,
    },
    {
      hedge_type: "long_vol",
      instrument: "VIX calls",
      rationale: "Direct volatility exposure",
      priority: 2,
    },
  ],
  portfolio_weights: { SPX: 0.4, TLT: 0.3, GLD: 0.2, DXY: 0.1 },
};

const mockPortfolio = {
  current_regime: 1,
  blended_weights: { SPX: 0.35, TLT: 0.25, GLD: 0.2, DXY: 0.1, WTI: 0.1 },
  regime_allocations: [
    {
      regime: 1,
      regime_name: "Risk-On Growth",
      probability: 0.6,
      weights: { SPX: 0.5, TLT: 0.15, GLD: 0.15, DXY: 0.1, WTI: 0.1 },
      expected_return: 0.08,
      expected_volatility: 0.14,
    },
    {
      regime: 2,
      regime_name: "Risk-Off Crisis",
      probability: 0.25,
      weights: { SPX: 0.1, TLT: 0.4, GLD: 0.3, DXY: 0.15, WTI: 0.05 },
      expected_return: -0.02,
      expected_volatility: 0.2,
    },
  ],
  rebalance_trigger: true,
  rebalance_reason: "Regime drift detected — weights deviated >5%",
  transaction_cost_estimate: 0.2,
  expected_return: 6.0,
  expected_volatility: 12.0,
  sharpe_ratio: 1.25,
  max_drawdown_constraint: -15.0,
};

vi.mock("@/lib/api", () => ({
  fetchTailRisk: vi.fn(),
  fetchPortfolioOptimize: vi.fn(),
}));

import { fetchTailRisk, fetchPortfolioOptimize } from "@/lib/api";
import RiskPage from "@/app/risk/page";

beforeEach(() => {
  vi.mocked(fetchTailRisk).mockResolvedValue(mockTailRisk);
  vi.mocked(fetchPortfolioOptimize).mockResolvedValue(mockPortfolio);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("RiskPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<RiskPage />);
    expect(
      screen.getByRole("heading", { name: /Risk & Portfolio/i }),
    ).toBeInTheDocument();
  });

  it("renders VaR KPI", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Weighted VaR (95%)")).toBeInTheDocument();
      expect(screen.getByText("-3.5%")).toBeInTheDocument();
    });
  });

  it("renders CVaR KPI", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Weighted CVaR (95%)")).toBeInTheDocument();
      expect(screen.getByText("-5.2%")).toBeInTheDocument();
    });
  });

  it("shows worst scenario", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Worst Scenario")).toBeInTheDocument();
      // "Risk-Off Crisis" appears in both KPI and scenario chart
      const matches = screen.getAllByText("Risk-Off Crisis");
      expect(matches.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows tail risk driver", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Tail Risk Driver")).toBeInTheDocument();
      expect(screen.getByText("VIX spike")).toBeInTheDocument();
    });
  });

  it("shows expected Sharpe", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Expected Sharpe")).toBeInTheDocument();
      expect(screen.getByText("1.25")).toBeInTheDocument();
    });
  });

  it("renders scenario VaR chart", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(
        screen.getByText(/Scenario Analysis — VaR by Regime/),
      ).toBeInTheDocument();
    });
  });

  it("renders optimal portfolio allocation chart", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Optimal Portfolio Allocation"),
      ).toBeInTheDocument();
    });
  });

  it("renders portfolio stats cards", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      // "Expected Return" appears in KPI card and in regime allocations table
      const returnLabels = screen.getAllByText("Expected Return");
      expect(returnLabels.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Expected Volatility")).toBeInTheDocument();
      expect(screen.getByText("Max DD Constraint")).toBeInTheDocument();
      expect(screen.getByText("Transaction Cost")).toBeInTheDocument();
    });
  });

  it("shows rebalance trigger warning", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Rebalance Triggered")).toBeInTheDocument();
      expect(
        screen.getByText(/Regime drift detected/),
      ).toBeInTheDocument();
    });
  });

  it("renders regime allocations table", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Regime-Conditional Allocations"),
      ).toBeInTheDocument();
      expect(screen.getByText("Probability")).toBeInTheDocument();
    });
  });

  it("renders hedge recommendations", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(screen.getByText("Hedge Recommendations")).toBeInTheDocument();
      expect(
        screen.getByText(/put spread — SPY 400\/380 Put Spread/i),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Direct volatility exposure/),
      ).toBeInTheDocument();
    });
  });

  it("renders Plotly charts", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("calls both API functions", async () => {
    renderWithQuery(<RiskPage />);
    await waitFor(() => {
      expect(fetchTailRisk).toHaveBeenCalled();
      expect(fetchPortfolioOptimize).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchTailRisk).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchPortfolioOptimize).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<RiskPage />);
    expect(
      screen.getByRole("heading", { name: /Risk & Portfolio/i }),
    ).toBeInTheDocument();
  });
});
