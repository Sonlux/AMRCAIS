import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";

vi.mock("@/components/charts/PlotlyChart", () => ({
  __esModule: true,
  default: (props: Record<string, unknown>) => (
    <div data-testid="plotly-mock" data-traces={JSON.stringify(props.data)} />
  ),
}));

import YieldCurveSurfaceChart from "@/components/charts/YieldCurveSurfaceChart";
import type { YieldCurveDataResponse } from "@/lib/types";

const mockData: YieldCurveDataResponse = {
  tenors: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
  yields: [5.2, 5.1, 4.9, 4.6, 4.5, 4.3, 4.35, 4.4, 4.6, 4.65],
  curve_shape: "normal",
  slope_2_10: -0.2,
  slope_3m_10: -0.8,
  curvature: 0.15,
  regime: 1,
  regime_name: "Risk-On Growth",
  timestamp: "2026-01-01T00:00:00",
};

describe("YieldCurveSurfaceChart", () => {
  it("renders without crashing", () => {
    const { container } = render(
      <YieldCurveSurfaceChart data={mockData} height={300} />,
    );
    expect(container).toBeTruthy();
  });

  it("renders a Plotly chart", () => {
    const { getByTestId } = render(
      <YieldCurveSurfaceChart data={mockData} />,
    );
    const plot = getByTestId("plotly-mock");
    expect(plot).toBeInTheDocument();
  });

  it("passes yield data to traces", () => {
    const { getByTestId } = render(
      <YieldCurveSurfaceChart data={mockData} />,
    );
    const plot = getByTestId("plotly-mock");
    const traces = JSON.parse(plot.getAttribute("data-traces") || "[]");
    // First trace should contain our yield values
    expect(traces[0].y).toEqual(mockData.yields);
  });

  it("converts tenors to labels", () => {
    const { getByTestId } = render(
      <YieldCurveSurfaceChart data={mockData} />,
    );
    const plot = getByTestId("plotly-mock");
    const traces = JSON.parse(plot.getAttribute("data-traces") || "[]");
    // 0.25 → "3M", 10 → "10Y"
    expect(traces[0].x).toContain("3M");
    expect(traces[0].x).toContain("10Y");
  });

  it("handles inverted curve shape", () => {
    const invertedData = { ...mockData, curve_shape: "inverted" };
    const { container } = render(
      <YieldCurveSurfaceChart data={invertedData} />,
    );
    expect(container).toBeTruthy();
  });
});
