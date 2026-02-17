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

const mockPortfolio = {
  total_equity: 105000,
  cash: 15000,
  positions_value: 90000,
  num_positions: 4,
  positions: [
    {
      asset: "SPX",
      shares: 10,
      avg_price: 4500,
      current_value: 46000,
      weight: 0.44,
      unrealized_pnl: 1000,
      unrealized_pnl_pct: 0.022,
    },
    {
      asset: "TLT",
      shares: 100,
      avg_price: 95,
      current_value: 9800,
      weight: 0.09,
      unrealized_pnl: 300,
      unrealized_pnl_pct: 0.032,
    },
  ],
  total_return_pct: 5.0,
  rebalance_count: 3,
  last_rebalance: "2026-02-10",
  current_regime: 1,
};

const mockMetrics = {
  total_return: 0.05,
  sharpe_ratio: 1.3,
  max_drawdown: -8.0,
  win_rate: 0.62,
  total_trades: 25,
  avg_daily_return: 0.001,
  volatility: 12.0,
  calmar_ratio: 0.65,
};

const mockEquity = {
  curve: [
    { date: "2026-01-01", equity: 100000, regime: 1 },
    { date: "2026-01-15", equity: 102000, regime: 1 },
    { date: "2026-02-01", equity: 103500, regime: 2 },
    { date: "2026-02-15", equity: 105000, regime: 1 },
  ],
  total_points: 4,
};

const mockAttribution = {
  attribution: [
    { regime: 1, regime_name: "Risk-On Growth", pnl: 4000, pnl_pct: 0.04, days: 30 },
    { regime: 2, regime_name: "Risk-Off Crisis", pnl: -1500, pnl_pct: -0.015, days: 10 },
  ],
  total_pnl: 2500,
};

const mockTrades = {
  orders: [
    {
      order_id: "ord-001",
      timestamp: "2026-02-12T14:30:00Z",
      asset: "SPX",
      side: "buy",
      shares: 5,
      price: 4520.5,
      value: 22602.5,
      regime: 1,
      reason: "Risk-On rebalance",
    },
    {
      order_id: "ord-002",
      timestamp: "2026-02-10T10:00:00Z",
      asset: "TLT",
      side: "sell",
      shares: 20,
      price: 96.0,
      value: 1920,
      regime: 2,
      reason: "Regime shift to Risk-Off",
    },
  ],
  total: 2,
};

vi.mock("@/lib/api", () => ({
  fetchPortfolio: vi.fn(),
  fetchPortfolioMetrics: vi.fn(),
  fetchPortfolioEquity: vi.fn(),
  fetchRegimeAttribution: vi.fn(),
  fetchTrades: vi.fn(),
}));

import {
  fetchPortfolio,
  fetchPortfolioMetrics,
  fetchPortfolioEquity,
  fetchRegimeAttribution,
  fetchTrades,
} from "@/lib/api";
import TradingPage from "@/app/trading/page";

beforeEach(() => {
  vi.mocked(fetchPortfolio).mockResolvedValue(mockPortfolio);
  vi.mocked(fetchPortfolioMetrics).mockResolvedValue(mockMetrics);
  vi.mocked(fetchPortfolioEquity).mockResolvedValue(mockEquity);
  vi.mocked(fetchRegimeAttribution).mockResolvedValue(mockAttribution);
  vi.mocked(fetchTrades).mockResolvedValue(mockTrades);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("TradingPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<TradingPage />);
    expect(screen.getByText("Paper Trading")).toBeInTheDocument();
  });

  it("renders equity KPI", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Total Equity")).toBeInTheDocument();
      expect(screen.getByText("$105,000")).toBeInTheDocument();
    });
  });

  it("shows total return", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Total Return")).toBeInTheDocument();
      expect(screen.getByText("5.0%")).toBeInTheDocument();
    });
  });

  it("shows positions count with cash", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Positions")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
    });
  });

  it("shows rebalance count", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Rebalances")).toBeInTheDocument();
      expect(screen.getByText("3")).toBeInTheDocument();
    });
  });

  it("shows Sharpe ratio from metrics", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument();
      expect(screen.getByText("1.30")).toBeInTheDocument();
    });
  });

  it("renders performance metrics row", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Max Drawdown")).toBeInTheDocument();
      expect(screen.getByText("Win Rate")).toBeInTheDocument();
      expect(screen.getByText("Volatility (Ann.)")).toBeInTheDocument();
      expect(screen.getByText("Calmar Ratio")).toBeInTheDocument();
    });
  });

  it("renders equity curve section", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("Equity Curve")).toBeInTheDocument();
    });
  });

  it("renders positions table", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      // "Asset" appears in both positions and trade log tables
      const assetHeaders = screen.getAllByText("Asset");
      expect(assetHeaders.length).toBeGreaterThanOrEqual(2);
      const sharesHeaders = screen.getAllByText("Shares");
      expect(sharesHeaders.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Weight")).toBeInTheDocument();
    });
  });

  it("shows position data in table", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      // SPX and TLT appear in both positions and trades
      const spxCells = screen.getAllByText("SPX");
      expect(spxCells.length).toBeGreaterThanOrEqual(1);
      const tltCells = screen.getAllByText("TLT");
      expect(tltCells.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders regime attribution chart", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText(/P&L by Regime/i)).toBeInTheDocument();
    });
  });

  it("renders trade history table", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText(/Trade History/)).toBeInTheDocument();
      expect(screen.getByText("Side")).toBeInTheDocument();
      expect(screen.getByText("Price")).toBeInTheDocument();
      expect(screen.getByText("Reason")).toBeInTheDocument();
    });
  });

  it("shows trade data including side badges", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(screen.getByText("buy")).toBeInTheDocument();
      expect(screen.getByText("sell")).toBeInTheDocument();
      expect(screen.getByText("Risk-On rebalance")).toBeInTheDocument();
    });
  });

  it("renders Plotly charts", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("calls all API functions", async () => {
    renderWithQuery(<TradingPage />);
    await waitFor(() => {
      expect(fetchPortfolio).toHaveBeenCalled();
      expect(fetchPortfolioMetrics).toHaveBeenCalled();
      expect(fetchPortfolioEquity).toHaveBeenCalled();
      expect(fetchRegimeAttribution).toHaveBeenCalled();
      expect(fetchTrades).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchPortfolio).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchPortfolioMetrics).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchPortfolioEquity).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchRegimeAttribution).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchTrades).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<TradingPage />);
    expect(screen.getByText("Paper Trading")).toBeInTheDocument();
  });
});
