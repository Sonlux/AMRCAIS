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

const mockSummary = {
  total_transitions: 42,
  total_anomalies: 18,
  total_macro_impacts: 7,
  regime_coverage: { "1": 120, "2": 45, "3": 15, "4": 30 },
};

const mockTransitions = {
  transitions: [
    {
      id: "tr-001",
      from_regime: 1,
      to_regime: 2,
      confidence: 0.85,
      disagreement: 0.2,
      timestamp: "2026-02-10T12:00:00Z",
      indicators: { vix: 28, spread: -0.5 },
    },
    {
      id: "tr-002",
      from_regime: 2,
      to_regime: 1,
      confidence: 0.78,
      disagreement: 0.35,
      timestamp: "2026-01-20T08:00:00Z",
      indicators: { vix: 18, spread: 0.8 },
    },
  ],
  total: 2,
};

const mockAnomalies = {
  anomalies: [
    {
      id: "anom-001",
      anomaly_type: "correlation_break",
      asset_pair: "SPX-TLT",
      regime: 1,
      z_score: 2.8,
      expected_value: -0.3,
      actual_value: 0.15,
      timestamp: "2026-02-12T14:00:00Z",
    },
    {
      id: "anom-002",
      anomaly_type: "mean_reversion",
      asset_pair: "GLD-DXY",
      regime: 2,
      z_score: -1.5,
      expected_value: -0.6,
      actual_value: -0.2,
      timestamp: "2026-02-11T10:00:00Z",
    },
  ],
  total: 2,
};

vi.mock("@/lib/api", () => ({
  fetchKnowledgeSummary: vi.fn(),
  fetchTransitions: vi.fn(),
  fetchAnomalies: vi.fn(),
}));

import {
  fetchKnowledgeSummary,
  fetchTransitions,
  fetchAnomalies,
} from "@/lib/api";
import KnowledgePage from "@/app/knowledge/page";

beforeEach(() => {
  vi.mocked(fetchKnowledgeSummary).mockResolvedValue(mockSummary);
  vi.mocked(fetchTransitions).mockResolvedValue(mockTransitions);
  vi.mocked(fetchAnomalies).mockResolvedValue(mockAnomalies);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("KnowledgePage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<KnowledgePage />);
    expect(screen.getByText("Knowledge Base")).toBeInTheDocument();
  });

  it("renders total transitions KPI", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("Total Transitions")).toBeInTheDocument();
      expect(screen.getByText("42")).toBeInTheDocument();
    });
  });

  it("renders total anomalies KPI", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("Total Anomalies")).toBeInTheDocument();
      expect(screen.getByText("18")).toBeInTheDocument();
    });
  });

  it("renders macro impacts KPI", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("Macro Impacts")).toBeInTheDocument();
      expect(screen.getByText("7")).toBeInTheDocument();
    });
  });

  it("renders regime coverage KPI", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("Regime Coverage")).toBeInTheDocument();
      expect(screen.getByText("4")).toBeInTheDocument();
    });
  });

  it("renders knowledge coverage chart", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(
        screen.getByText("Knowledge Coverage by Regime"),
      ).toBeInTheDocument();
    });
  });

  it("renders transitions section with count", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText(/Recent Transitions/)).toBeInTheDocument();
    });
  });

  it("renders transition cards with confidence and disagreement", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      // Text is rendered as "Confidence: 85.0%" inside combined spans
      expect(screen.getByText(/Confidence: 85\.0%/)).toBeInTheDocument();
      expect(screen.getByText(/Disagreement: 20\.0%/)).toBeInTheDocument();
    });
  });

  it("renders anomalies table section", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText(/Correlation Anomalies/)).toBeInTheDocument();
    });
  });

  it("renders anomaly data in table", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("correlation_break")).toBeInTheDocument();
      expect(screen.getByText("SPX-TLT")).toBeInTheDocument();
      expect(screen.getByText("GLD-DXY")).toBeInTheDocument();
      expect(screen.getByText("2.80")).toBeInTheDocument();
    });
  });

  it("renders anomaly table columns", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(screen.getByText("Type")).toBeInTheDocument();
      expect(screen.getByText("Pair")).toBeInTheDocument();
      expect(screen.getByText("Z-Score")).toBeInTheDocument();
      expect(screen.getByText("Expected")).toBeInTheDocument();
      expect(screen.getByText("Actual")).toBeInTheDocument();
    });
  });

  it("renders Plotly chart for coverage", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("calls all API functions", async () => {
    renderWithQuery(<KnowledgePage />);
    await waitFor(() => {
      expect(fetchKnowledgeSummary).toHaveBeenCalled();
      expect(fetchTransitions).toHaveBeenCalled();
      expect(fetchAnomalies).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchKnowledgeSummary).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchTransitions).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchAnomalies).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<KnowledgePage />);
    expect(screen.getByText("Knowledge Base")).toBeInTheDocument();
  });
});
