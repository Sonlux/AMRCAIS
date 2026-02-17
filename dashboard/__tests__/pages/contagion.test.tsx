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

const mockContagion = {
  signal: { signal: "elevated", strength: 0.7 },
  granger_network: [
    { source: "SPX", target: "TLT", p_value: 0.01, lag: 3, f_stat: 8.5 },
    { source: "GLD", target: "DXY", p_value: 0.03, lag: 1, f_stat: 5.2 },
  ],
  spillover: {
    total_spillover_index: 55.3,
    directional_to: { SPX: 20, TLT: 15 },
    directional_from: { SPX: 10, TLT: 12 },
    net_spillover: { SPX: 10, TLT: 3, GLD: -5 },
    pairwise: { SPX: { TLT: 0.4, GLD: 0.2 }, TLT: { SPX: 0.3, GLD: 0.1 } },
    assets: ["SPX", "TLT", "GLD"],
  },
  network_graph: {
    nodes: [
      { id: "SPX", size: 10 },
      { id: "TLT", size: 8 },
    ],
    edges: [{ source: "SPX", target: "TLT", weight: 0.5 }],
  },
  contagion_flags: ["High SPX→TLT spillover", "Granger link cluster detected"],
  n_significant_links: 4,
  network_density: 0.65,
};

const mockSpillover = {
  total_spillover_index: 55.3,
  directional_to: { SPX: 20, TLT: 15, GLD: 10 },
  directional_from: { SPX: 10, TLT: 12, GLD: 8 },
  net_spillover: { SPX: 10, TLT: 3, GLD: -5 },
  pairwise: {
    SPX: { SPX: 0, TLT: 0.4, GLD: 0.2 },
    TLT: { SPX: 0.3, TLT: 0, GLD: 0.1 },
    GLD: { SPX: 0.15, TLT: 0.1, GLD: 0 },
  },
  assets: ["SPX", "TLT", "GLD"],
};

vi.mock("@/lib/api", () => ({
  fetchContagionAnalysis: vi.fn(),
  fetchSpillover: vi.fn(),
}));

import { fetchContagionAnalysis, fetchSpillover } from "@/lib/api";
import ContagionPage from "@/app/contagion/page";

beforeEach(() => {
  vi.mocked(fetchContagionAnalysis).mockResolvedValue(mockContagion);
  vi.mocked(fetchSpillover).mockResolvedValue(mockSpillover);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("ContagionPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<ContagionPage />);
    expect(screen.getByText("Contagion Network")).toBeInTheDocument();
  });

  it("renders KPI cards with contagion data", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Network Density")).toBeInTheDocument();
      expect(screen.getByText("65.0%")).toBeInTheDocument();
    });
  });

  it("shows significant links count", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Significant Links")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
    });
  });

  it("shows spillover index KPI", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Spillover Index")).toBeInTheDocument();
      expect(screen.getByText("55.30")).toBeInTheDocument();
    });
  });

  it("shows contagion flags count", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Contagion Flags")).toBeInTheDocument();
      expect(screen.getByText("2")).toBeInTheDocument();
    });
  });

  it("renders active contagion flag messages", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(
        screen.getByText(/High SPX→TLT spillover/),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Granger link cluster detected/),
      ).toBeInTheDocument();
    });
  });

  it("renders Granger causality network chart", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Granger Causality Network"),
      ).toBeInTheDocument();
    });
  });

  it("renders spillover matrix heatmap", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Spillover Matrix")).toBeInTheDocument();
    });
  });

  it("renders net spillover bar chart", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(
        screen.getByText(/Net Spillover \(Transmitter vs Receiver\)/),
      ).toBeInTheDocument();
    });
  });

  it("renders Granger links table with data", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(screen.getByText("Source")).toBeInTheDocument();
      expect(screen.getByText("Target")).toBeInTheDocument();
      expect(screen.getByText("p-value")).toBeInTheDocument();
      expect(screen.getByText("SPX")).toBeInTheDocument();
      expect(screen.getByText("TLT")).toBeInTheDocument();
    });
  });

  it("renders Plotly charts", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("calls both API functions", async () => {
    renderWithQuery(<ContagionPage />);
    await waitFor(() => {
      expect(fetchContagionAnalysis).toHaveBeenCalled();
      expect(fetchSpillover).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchContagionAnalysis).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchSpillover).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<ContagionPage />);
    expect(screen.getByText("Contagion Network")).toBeInTheDocument();
  });
});
