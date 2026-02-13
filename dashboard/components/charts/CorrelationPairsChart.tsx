"use client";

import PlotlyChart from "./PlotlyChart";

interface CorrelationPair {
  pair: string;
  value: number;
}

interface CorrelationPairsChartProps {
  pairs: CorrelationPair[];
  height?: number;
}

/**
 * Horizontal bar chart of top correlation pairs by absolute magnitude.
 * Used in the Correlations Monitor page.
 */
export default function CorrelationPairsChart({
  pairs,
  height = 280,
}: CorrelationPairsChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "bar",
          y: pairs.map((p) => p.pair),
          x: pairs.map((p) => p.value),
          orientation: "h" as const,
          marker: {
            color: pairs.map((p) => (p.value > 0 ? "#22c55e" : "#ef4444")),
          },
          hovertemplate: "%{y}<br>ρ = %{x:.3f}<extra></extra>",
        },
      ]}
      layout={{
        xaxis: {
          title: { text: "Correlation", font: { size: 10 } },
          range: [-1, 1],
        },
        yaxis: { tickfont: { size: 10 }, automargin: true },
        margin: { t: 10, l: 80, r: 20, b: 40 },
      }}
    />
  );
}
