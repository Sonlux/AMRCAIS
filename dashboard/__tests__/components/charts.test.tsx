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
import AccuracyLineChart from "@/components/charts/AccuracyLineChart";
import DrawdownChart from "@/components/charts/DrawdownChart";
import DisagreementVsSpxChart from "@/components/charts/DisagreementVsSpxChart";

describe("RegimeDistributionChart", () => {
  it("renders a pie/donut chart", () => {
    const { getByTestId } = render(
      <RegimeDistributionChart
        distribution={{ "1": 40, "2": 20, "3": 15, "4": 25 }}
      />,
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

describe("AccuracyLineChart", () => {
  const accuracyData = {
    classifiers: ["hmm", "ml", "ensemble"],
    series: [
      { date: "2024-01-01", accuracy: 0.85, classifier: "hmm" },
      { date: "2024-01-02", accuracy: 0.78, classifier: "hmm" },
      { date: "2024-01-01", accuracy: 0.9, classifier: "ml" },
      { date: "2024-01-02", accuracy: 0.82, classifier: "ml" },
      { date: "2024-01-01", accuracy: 0.88, classifier: "ensemble" },
      { date: "2024-01-02", accuracy: 0.84, classifier: "ensemble" },
    ],
    window: 30,
  };

  it("renders without crashing", () => {
    const { container } = render(<AccuracyLineChart data={accuracyData} />);
    expect(container).toBeTruthy();
  });

  it("creates one trace per classifier plus threshold", () => {
    const { getByTestId } = render(<AccuracyLineChart data={accuracyData} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    // 3 classifiers + 1 threshold line
    expect(traces.length).toBe(4);
  });

  it("uses scatter type for all traces", () => {
    const { getByTestId } = render(<AccuracyLineChart data={accuracyData} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    for (const trace of traces) {
      expect(trace.type).toBe("scatter");
    }
  });
});

describe("DrawdownChart", () => {
  const drawdownSeries = [
    { date: "2024-01-01", drawdown: 0 },
    { date: "2024-01-02", drawdown: -0.02 },
    { date: "2024-01-03", drawdown: -0.05 },
    { date: "2024-01-04", drawdown: -0.01 },
  ];

  it("renders without crashing", () => {
    const { container } = render(<DrawdownChart series={drawdownSeries} />);
    expect(container).toBeTruthy();
  });

  it("renders a filled area trace", () => {
    const { getByTestId } = render(<DrawdownChart series={drawdownSeries} />);
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces.length).toBe(1);
    expect(traces[0].fill).toBe("tozeroy");
  });

  it("returns null for empty series", () => {
    const { container } = render(<DrawdownChart series={[]} />);
    expect(container.innerHTML).toBe("");
  });
});

describe("DisagreementVsSpxChart", () => {
  const disagreement = [
    { date: "2024-01-01", disagreement: 0.3, threshold_exceeded: false },
    { date: "2024-01-02", disagreement: 0.7, threshold_exceeded: true },
    { date: "2024-01-03", disagreement: 0.5, threshold_exceeded: false },
  ];
  const prices = [
    {
      date: "2024-01-01",
      open: null,
      high: null,
      low: null,
      close: 4800,
      volume: null,
    },
    {
      date: "2024-01-02",
      open: null,
      high: null,
      low: null,
      close: 4750,
      volume: null,
    },
    {
      date: "2024-01-03",
      open: null,
      high: null,
      low: null,
      close: 4850,
      volume: null,
    },
  ];

  it("renders without crashing", () => {
    const { container } = render(
      <DisagreementVsSpxChart disagreement={disagreement} prices={prices} />,
    );
    expect(container).toBeTruthy();
  });

  it("creates two traces (disagreement + prices)", () => {
    const { getByTestId } = render(
      <DisagreementVsSpxChart disagreement={disagreement} prices={prices} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces.length).toBe(2);
  });

  it("uses dual y-axes", () => {
    const { getByTestId } = render(
      <DisagreementVsSpxChart disagreement={disagreement} prices={prices} />,
    );
    const traces = JSON.parse(
      getByTestId("plotly-mock").getAttribute("data-traces") || "[]",
    );
    expect(traces[1].yaxis).toBe("y2");
  });

  it("returns null for empty disagreement", () => {
    const { container } = render(
      <DisagreementVsSpxChart disagreement={[]} prices={prices} />,
    );
    expect(container.innerHTML).toBe("");
  });
});
