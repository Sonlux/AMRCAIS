import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";

vi.mock("@/components/charts/PlotlyChart", () => ({
  __esModule: true,
  default: (props: Record<string, unknown>) => (
    <div data-testid="plotly-mock" data-traces={JSON.stringify(props.data)} />
  ),
}));

import VolSurface3DChart from "@/components/charts/VolSurface3DChart";
import type { VolSurfaceDataResponse } from "@/lib/types";

const mockSurface: VolSurfaceDataResponse = {
  moneyness: [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
  expiry_days: [7, 14, 30, 60, 90, 120, 180, 270, 365],
  iv_grid: Array.from({ length: 9 }, (_, i) =>
    Array.from({ length: 9 }, (_, j) => 20 + i * 0.5 + j * 0.3),
  ),
  atm_vol: 22.5,
  regime: 2,
  regime_name: "Risk-Off Crisis",
  timestamp: "2026-01-15T12:00:00",
};

describe("VolSurface3DChart", () => {
  it("renders without crashing", () => {
    const { container } = render(
      <VolSurface3DChart data={mockSurface} height={400} />,
    );
    expect(container).toBeTruthy();
  });

  it("renders a Plotly chart", () => {
    const { getByTestId } = render(<VolSurface3DChart data={mockSurface} />);
    expect(getByTestId("plotly-mock")).toBeInTheDocument();
  });

  it("uses surface trace type", () => {
    const { getByTestId } = render(<VolSurface3DChart data={mockSurface} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].type).toBe("surface");
  });

  it("passes moneyness as x-axis", () => {
    const { getByTestId } = render(<VolSurface3DChart data={mockSurface} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].x).toEqual(mockSurface.moneyness);
  });

  it("passes expiry days as y-axis", () => {
    const { getByTestId } = render(<VolSurface3DChart data={mockSurface} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].y).toEqual(mockSurface.expiry_days);
  });

  it("passes iv_grid as z-axis", () => {
    const { getByTestId } = render(<VolSurface3DChart data={mockSurface} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].z).toEqual(mockSurface.iv_grid);
  });

  it("handles different regimes", () => {
    const riskOnSurface = { ...mockSurface, regime: 1 };
    const { container } = render(<VolSurface3DChart data={riskOnSurface} />);
    expect(container).toBeTruthy();
  });
});
