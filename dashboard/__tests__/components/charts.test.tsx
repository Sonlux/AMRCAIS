import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";

// Mock the PlotlyChart wrapper (which uses next/dynamic internally)
vi.mock("@/components/charts/PlotlyChart", () => ({
  __esModule: true,
  default: (props: Record<string, unknown>) => (
    <div data-testid="plotly-mock" data-traces={JSON.stringify(props.data)} />
  ),
}));

import RegimeDistributionChart from "@/components/charts/RegimeDistributionChart";
import RegimeReturnsChart from "@/components/charts/RegimeReturnsChart";
import ClassifierWeightsChart from "@/components/charts/ClassifierWeightsChart";
import TransitionMatrixChart from "@/components/charts/TransitionMatrixChart";
import CorrelationHeatmap from "@/components/charts/CorrelationHeatmap";

describe("RegimeDistributionChart", () => {
  it("renders a pie/donut chart", () => {
    const { getByTestId } = render(
      <RegimeDistributionChart distribution={{ "1": 40, "2": 20, "3": 15, "4": 25 }} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].type).toBe("pie");
  });
});

describe("RegimeReturnsChart", () => {
  const returns = [
    {
      regime: 1,
      regime_name: "Risk-On Growth",
      strategy_return: 0.12,
      benchmark_return: 0.08,
      days: 120,
      hit_rate: 0.65,
    },
    {
      regime: 2,
      regime_name: "Risk-Off Crisis",
      strategy_return: -0.05,
      benchmark_return: -0.15,
      days: 60,
      hit_rate: 0.72,
    },
  ];

  it("renders without crashing", () => {
    const { container } = render(<RegimeReturnsChart returns={returns} />);
    expect(container).toBeTruthy();
  });

  it("creates bar traces", () => {
    const { getByTestId } = render(<RegimeReturnsChart returns={returns} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces.length).toBe(2); // strategy + benchmark
    expect(traces[0].type).toBe("bar");
  });
});

describe("ClassifierWeightsChart", () => {
  it("renders weight bars", () => {
    const weights = { hmm: 0.3, ml: 0.25, correlation: 0.25, volatility: 0.2 };
    const { getByTestId } = render(
      <ClassifierWeightsChart weights={weights} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].type).toBe("bar");
    expect(traces[0].x).toEqual(Object.keys(weights));
  });
});

describe("TransitionMatrixChart", () => {
  it("renders a heatmap", () => {
    const matrix = [
      [10, 2, 1, 0],
      [1, 15, 3, 1],
      [0, 2, 8, 2],
      [1, 0, 1, 12],
    ];
    const names = ["Risk-On", "Risk-Off", "Stagflation", "DisInflation"];
    const { getByTestId } = render(
      <TransitionMatrixChart matrix={matrix} regimeNames={names} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].type).toBe("heatmap");
  });
});

describe("CorrelationHeatmap", () => {
  it("renders a heatmap with asset labels", () => {
    const matrix = [
      [1, 0.8, -0.3],
      [0.8, 1, -0.5],
      [-0.3, -0.5, 1],
    ];
    const assets = ["SPX", "TLT", "GLD"];
    const { getByTestId } = render(
      <CorrelationHeatmap matrix={matrix} assets={assets} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[0].type).toBe("heatmap");
    expect(traces[0].x).toEqual(assets);
    expect(traces[0].y).toEqual(assets);
  });
});
