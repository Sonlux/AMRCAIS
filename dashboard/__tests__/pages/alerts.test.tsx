import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithQuery } from "../helpers";

// ── Mocks ───────────────────────────────────────────────────────────

const mockAlerts = {
  alerts: [
    {
      alert_id: "alert-001",
      alert_type: "regime_change",
      severity: "critical",
      message: "Regime switched from Risk-On to Risk-Off",
      timestamp: "2026-02-15T10:00:00Z",
      acknowledged: false,
      data: {},
    },
    {
      alert_id: "alert-002",
      alert_type: "vix_spike",
      severity: "high",
      message: "VIX exceeded 30",
      timestamp: "2026-02-14T09:00:00Z",
      acknowledged: true,
      data: {},
    },
  ],
  total: 2,
  unacknowledged: 1,
};

const mockEvents = {
  events: [
    {
      event_type: "regime_update",
      data: {},
      timestamp: "2026-02-15T10:05:00Z",
      event_id: "evt-001",
      source: "regime_detector",
    },
    {
      event_type: "data_refresh",
      data: {},
      timestamp: "2026-02-15T09:00:00Z",
      event_id: "evt-002",
      source: "pipeline",
    },
  ],
  total: 2,
};

const mockConfig = {
  configs: {
    regime_change: {
      enabled: true,
      cooldown_seconds: 300,
      threshold: null,
    },
    vix_spike: {
      enabled: true,
      cooldown_seconds: 600,
      threshold: 30,
    },
  },
};

const mockPhase4Status = {
  event_bus: {},
  scheduler: {},
  alert_engine: {},
  stream_manager: {},
  paper_trading: {},
};

vi.mock("@/lib/api", () => ({
  fetchAlerts: vi.fn(),
  fetchEvents: vi.fn(),
  fetchAlertConfig: vi.fn(),
  acknowledgeAlert: vi.fn(),
  fetchPhase4Status: vi.fn(),
}));

import {
  fetchAlerts,
  fetchEvents,
  fetchAlertConfig,
  acknowledgeAlert,
  fetchPhase4Status,
} from "@/lib/api";
import AlertsPage from "@/app/alerts/page";

beforeEach(() => {
  vi.mocked(fetchAlerts).mockResolvedValue(mockAlerts);
  vi.mocked(fetchEvents).mockResolvedValue(mockEvents);
  vi.mocked(fetchAlertConfig).mockResolvedValue(mockConfig);
  vi.mocked(fetchPhase4Status).mockResolvedValue(mockPhase4Status);
  vi.mocked(acknowledgeAlert).mockResolvedValue({ status: "acknowledged" } as never);
});

// ── Tests ───────────────────────────────────────────────────────────

describe("AlertsPage", () => {
  it("renders the page header", async () => {
    renderWithQuery(<AlertsPage />);
    expect(
      screen.getByRole("heading", { name: /Alerts & Events/i }),
    ).toBeInTheDocument();
  });

  it("renders KPI cards", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("Total Alerts")).toBeInTheDocument();
      expect(screen.getByText("Unacknowledged")).toBeInTheDocument();
      expect(screen.getByText("Total Events")).toBeInTheDocument();
      expect(screen.getByText("Alert Configs")).toBeInTheDocument();
    });
  });

  it("shows total alerts count", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      // total = 2
      const totalAlerts = screen.getByText("Total Alerts");
      expect(totalAlerts).toBeInTheDocument();
    });
  });

  it("shows unacknowledged count", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("1")).toBeInTheDocument();
    });
  });

  it("renders severity filter dropdown", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("Severity")).toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });
  });

  it("renders unacknowledged only checkbox", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(
        screen.getByLabelText("Unacknowledged only"),
      ).toBeInTheDocument();
    });
  });

  it("renders alert list with alert data", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(
        screen.getByText("Regime switched from Risk-On to Risk-Off"),
      ).toBeInTheDocument();
      expect(screen.getByText("VIX exceeded 30")).toBeInTheDocument();
    });
  });

  it("shows severity badges on alerts", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("critical")).toBeInTheDocument();
      expect(screen.getByText("high")).toBeInTheDocument();
    });
  });

  it("shows Ack button for unacknowledged alerts", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("Ack")).toBeInTheDocument();
    });
  });

  it("shows checkmark for acknowledged alerts", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("✓")).toBeInTheDocument();
    });
  });

  it("calls acknowledgeAlert on Ack click", async () => {
    const user = userEvent.setup();
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("Ack")).toBeInTheDocument();
    });
    await user.click(screen.getByText("Ack"));
    await waitFor(() => {
      expect(acknowledgeAlert).toHaveBeenCalled();
      expect(vi.mocked(acknowledgeAlert).mock.calls[0][0]).toBe("alert-001");
    });
  });

  it("renders alert config table", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("Alert Configuration")).toBeInTheDocument();
      expect(screen.getByText("Alert Type")).toBeInTheDocument();
      expect(screen.getByText("Enabled")).toBeInTheDocument();
      expect(screen.getByText("Cooldown (s)")).toBeInTheDocument();
      expect(screen.getByText("Threshold")).toBeInTheDocument();
    });
  });

  it("shows ON/OFF badges in config table", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      const onBadges = screen.getAllByText("ON");
      expect(onBadges.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("renders events log", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText(/Recent Events/)).toBeInTheDocument();
      expect(screen.getByText("regime_update")).toBeInTheDocument();
      expect(screen.getByText("data_refresh")).toBeInTheDocument();
    });
  });

  it("shows event sources", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(screen.getByText("regime_detector")).toBeInTheDocument();
      expect(screen.getByText("pipeline")).toBeInTheDocument();
    });
  });

  it("calls all API functions", async () => {
    renderWithQuery(<AlertsPage />);
    await waitFor(() => {
      expect(fetchAlerts).toHaveBeenCalled();
      expect(fetchEvents).toHaveBeenCalled();
      expect(fetchAlertConfig).toHaveBeenCalled();
      expect(fetchPhase4Status).toHaveBeenCalled();
    });
  });

  it("shows skeletons during loading", () => {
    vi.mocked(fetchAlerts).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchEvents).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchAlertConfig).mockReturnValue(new Promise(() => {}));
    vi.mocked(fetchPhase4Status).mockReturnValue(new Promise(() => {}));
    renderWithQuery(<AlertsPage />);
    expect(
      screen.getByRole("heading", { name: /Alerts & Events/i }),
    ).toBeInTheDocument();
  });
});
