import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithQuery } from "../helpers";

// ── Mocks ───────────────────────────────────────────────────────────

const mockSummary = {
  total_reports: 12,
  report_types: { case_study: 5, factor_analysis: 4, regime_report: 3 },
  last_report: "2026-02-15T10:00:00Z",
};

const mockReports = {
  reports: [
    {
      id: "rpt-001",
      report_type: "case_study",
      title: "Risk-On to Risk-Off Transition Analysis",
      content: "Detailed analysis of the Q1 regime shift...",
      created_at: "2026-02-15T10:00:00Z",
      metadata: { from_regime: 1, to_regime: 2 },
    },
    {
      id: "rpt-002",
      report_type: "factor_analysis",
      title: "Momentum Factor Deep Dive",
      content: "Factor exposure analysis across regimes...",
      created_at: "2026-02-10T08:00:00Z",
      metadata: { regime: 1 },
    },
  ],
  total: 2,
};

const mockCaseStudy = {
  id: "rpt-003",
  report_type: "case_study",
  title: "Generated Case Study",
  content: "Auto-generated content...",
  created_at: "2026-02-16T12:00:00Z",
  metadata: {},
};

vi.mock("@/lib/api", () => ({
  fetchResearchSummary: vi.fn(),
  fetchResearchReports: vi.fn(),
  generateCaseStudy: vi.fn(),
}));

import {
  fetchResearchSummary,
  fetchResearchReports,
  generateCaseStudy,
} from "@/lib/api";
import ResearchPage from "@/app/research/page";

beforeEach(() => {
  vi.mocked(fetchResearchSummary).mockResolvedValue(mockSummary);
  vi.mocked(fetchResearchReports).mockResolvedValue(mockReports);
  vi.mocked(generateCaseStudy).mockResolvedValue(mockCaseStudy);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("ResearchPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<ResearchPage />);
    expect(screen.getByText("Research Publisher")).toBeInTheDocument();
  });

  it("renders total reports KPI", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Total Reports")).toBeInTheDocument();
      expect(screen.getByText("12")).toBeInTheDocument();
    });
  });

  it("renders report types count KPI", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Report Types")).toBeInTheDocument();
      // "3" appears as report types count and as regime_report count
      const threes = screen.getAllByText("3");
      expect(threes.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders last report KPI", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Last Report")).toBeInTheDocument();
    });
  });

  it("renders publisher status KPI", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Publisher Status")).toBeInTheDocument();
      expect(screen.getByText("Active")).toBeInTheDocument();
    });
  });

  it("renders report type breakdown", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Reports by Type")).toBeInTheDocument();
      // "case_study" appears in breakdown and in report list badges
      const caseStudies = screen.getAllByText("case_study");
      expect(caseStudies.length).toBeGreaterThanOrEqual(1);
      const factorAnalysis = screen.getAllByText("factor_analysis");
      expect(factorAnalysis.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("regime_report")).toBeInTheDocument();
    });
  });

  it("renders case study generator section", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText("Generate Case Study")).toBeInTheDocument();
      expect(screen.getByText("From Regime")).toBeInTheDocument();
      expect(screen.getByText("To Regime")).toBeInTheDocument();
    });
  });

  it("renders generate button", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Generate" }),
      ).toBeInTheDocument();
    });
  });

  it("disables generate button when regimes are same", async () => {
    renderWithQuery(<ResearchPage />);
    // Default: fromRegime=1, toRegime=2 → enabled initially
    // We need to set them to the same value
    const user = userEvent.setup();
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Generate" }),
      ).toBeInTheDocument();
    });
    // Change toRegime to 1 (same as fromRegime)
    const selects = screen.getAllByRole("combobox");
    // Second select is toRegime
    await user.selectOptions(selects[1], "1");
    expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
    expect(
      screen.getByText("Source and target regimes must differ"),
    ).toBeInTheDocument();
  });

  it("calls generateCaseStudy on Generate click", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => {
      expect(generateCaseStudy).toHaveBeenCalledWith(1, 2);
    });
  });

  it("shows success message after generation", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: "Generate" }));
    await waitFor(() => {
      expect(
        screen.getByText("Case study generated successfully."),
      ).toBeInTheDocument();
    });
  });

  it("renders report list", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(screen.getByText(/Research Reports/)).toBeInTheDocument();
      expect(
        screen.getByText("Risk-On to Risk-Off Transition Analysis"),
      ).toBeInTheDocument();
      expect(screen.getByText("Momentum Factor Deep Dive")).toBeInTheDocument();
    });
  });

  it("shows report type badges", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      // report_type badges inside the report list
      const badges = screen.getAllByText("case_study");
      expect(badges.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("expands report on click to show content", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Risk-On to Risk-Off Transition Analysis"),
      ).toBeInTheDocument();
    });
    await user.click(
      screen.getByText("Risk-On to Risk-Off Transition Analysis"),
    );
    await waitFor(() => {
      expect(
        screen.getByText("Detailed analysis of the Q1 regime shift..."),
      ).toBeInTheDocument();
    });
  });

  it("collapses report on second click", async () => {
    const user = userEvent.setup();
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Risk-On to Risk-Off Transition Analysis"),
      ).toBeInTheDocument();
    });
    // Expand
    await user.click(
      screen.getByText("Risk-On to Risk-Off Transition Analysis"),
    );
    await waitFor(() => {
      expect(
        screen.getByText("Detailed analysis of the Q1 regime shift..."),
      ).toBeInTheDocument();
    });
    // Collapse
    await user.click(
      screen.getByText("Risk-On to Risk-Off Transition Analysis"),
    );
    await waitFor(() => {
      expect(
        screen.queryByText("Detailed analysis of the Q1 regime shift..."),
      ).not.toBeInTheDocument();
    });
  });

  it("calls both query API functions", async () => {
    renderWithQuery(<ResearchPage />);
    await waitFor(() => {
      expect(fetchResearchSummary).toHaveBeenCalled();
      expect(fetchResearchReports).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchResearchSummary).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchResearchReports).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<ResearchPage />);
    expect(screen.getByText("Research Publisher")).toBeInTheDocument();
  });
});
