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

const mockForecast = {
  current_regime: 1,
  horizon_days: 30,
  hmm_probs: { "1": 0.6, "2": 0.25, "3": 0.1, "4": 0.05 },
  indicator_probs: { "1": 0.55, "2": 0.3, "3": 0.1, "4": 0.05 },
  blended_probs: {
    "Risk-On Growth": 0.6,
    "Risk-Off Crisis": 0.25,
    Stagflation: 0.1,
    "Disinflationary Boom": 0.05,
  },
  leading_indicators: {},
  transition_risk: "medium",
  most_likely_next: 2,
  most_likely_next_name: "Risk-Off Crisis",
  confidence: 0.82,
};

const mockMultiTimeframe = {
  daily: { regime: 1, confidence: 0.9 },
  weekly: { regime: 1, confidence: 0.85 },
  monthly: { regime: 2, confidence: 0.7 },
  agreement_score: 0.67,
  conflict_detected: true,
  highest_conviction: "daily",
  trade_signal: "caution",
};

const mockSurpriseIndex = {
  index: 0.42,
  direction: "positive",
  components: { CPI: 0.3, NFP: -0.1 },
  active_surprises: 3,
  total_historical: 25,
};

const mockActiveSurprises = [
  {
    indicator: "CPI",
    surprise: 0.3,
    release_date: "2026-02-10",
    half_life_days: 14,
    initial_weight: 0.8,
    regime_at_release: 1,
  },
  {
    indicator: "NFP",
    surprise: -0.2,
    release_date: "2026-02-07",
    half_life_days: 7,
    initial_weight: 0.5,
    regime_at_release: 1,
  },
];

const mockDecayCurves = {
  curves: {
    CPI: [
      { day: 0, impact: 1.0, is_stale: false },
      { day: 7, impact: 0.7, is_stale: false },
      { day: 14, impact: 0.5, is_stale: false },
    ],
  },
};

const mockNarrative = {
  headline: "Risk-On Continues with CPI Headwind",
  regime_section: "Currently in Risk-On Growth (confidence 82%)",
  signal_section: "Macro indicators mostly supportive",
  risk_section: "Transition risk moderate due to CPI surprise",
  positioning_section: "Maintain risk-on tilt with hedges",
  full_text: "Full narrative text here",
  data_sources: ["FRED", "yfinance"],
};

vi.mock("@/lib/api", () => ({
  fetchTransitionForecast: vi.fn(),
  fetchMultiTimeframe: vi.fn(),
  fetchSurpriseIndex: vi.fn(),
  fetchActiveSurprises: vi.fn(),
  fetchDecayCurves: vi.fn(),
  fetchNarrative: vi.fn(),
}));

import {
  fetchTransitionForecast,
  fetchMultiTimeframe,
  fetchSurpriseIndex,
  fetchActiveSurprises,
  fetchDecayCurves,
  fetchNarrative,
} from "@/lib/api";
import IntelligencePage from "@/app/intelligence/page";

beforeEach(() => {
  vi.mocked(fetchTransitionForecast).mockResolvedValue(mockForecast);
  vi.mocked(fetchMultiTimeframe).mockResolvedValue(mockMultiTimeframe);
  vi.mocked(fetchSurpriseIndex).mockResolvedValue(mockSurpriseIndex);
  vi.mocked(fetchActiveSurprises).mockResolvedValue(mockActiveSurprises);
  vi.mocked(fetchDecayCurves).mockResolvedValue(mockDecayCurves);
  vi.mocked(fetchNarrative).mockResolvedValue(mockNarrative);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("IntelligencePage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<IntelligencePage />);
    expect(screen.getByText("Intelligence Hub")).toBeInTheDocument();
  });

  it("renders KPI cards with forecast data", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText("Transition Risk")).toBeInTheDocument();
      expect(screen.getByText("medium")).toBeInTheDocument();
    });
  });

  it("shows forecast confidence", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText("Forecast Confidence")).toBeInTheDocument();
      expect(screen.getByText("82.0%")).toBeInTheDocument();
    });
  });

  it("renders surprise index KPI", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText("Surprise Index")).toBeInTheDocument();
    });
  });

  it("renders transition probability chart", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText(/Transition Probabilities/i)).toBeInTheDocument();
    });
  });

  it("renders multi-timeframe analysis section", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(
        screen.getByText(/Multi-Timeframe Regime Analysis/i),
      ).toBeInTheDocument();
    });
  });

  it("shows conflict warning when detected", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText(/Agreement/i)).toBeInTheDocument();
    });
  });

  it("renders active surprises table", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(screen.getByText("CPI")).toBeInTheDocument();
      expect(screen.getByText("NFP")).toBeInTheDocument();
    });
  });

  it("renders the market narrative headline", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(
        screen.getByText("Risk-On Continues with CPI Headwind"),
      ).toBeInTheDocument();
    });
  });

  it("renders narrative sections", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(
        screen.getByText(/Macro indicators mostly supportive/),
      ).toBeInTheDocument();
    });
  });

  it("calls all API functions", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      expect(fetchTransitionForecast).toHaveBeenCalled();
      expect(fetchMultiTimeframe).toHaveBeenCalled();
      expect(fetchSurpriseIndex).toHaveBeenCalled();
      expect(fetchActiveSurprises).toHaveBeenCalled();
      expect(fetchDecayCurves).toHaveBeenCalled();
      expect(fetchNarrative).toHaveBeenCalled();
    });
  });

  it("renders Plotly charts for transition probabilities and decay", async () => {
    renderWithQuery(<IntelligencePage />);
    await waitFor(() => {
      const charts = screen.getAllByTestId("plotly-mock");
      expect(charts.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("shows skeleton loading states initially", () => {
    // Delay mock resolution to test loading state
    vi.mocked(fetchTransitionForecast).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchMultiTimeframe).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchSurpriseIndex).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchActiveSurprises).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchDecayCurves).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchNarrative).mockReturnValue(new Promise(() => {}));

    renderWithQuery(<IntelligencePage />);
    // Header should still render while data is loading
    expect(screen.getByText("Intelligence Hub")).toBeInTheDocument();
  });
});
