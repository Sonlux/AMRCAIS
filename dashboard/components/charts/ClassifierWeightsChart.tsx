"use client";

import PlotlyChart from "./PlotlyChart";

interface ClassifierWeightsChartProps {
  weights: Record<string, number>;
  height?: number;
}

/**
 * Bar chart showing classifier weights.
 * Used in both Regime Explorer and Meta-Learning pages.
 */
export default function ClassifierWeightsChart({
  weights,
  height = 260,
}: ClassifierWeightsChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "bar",
          x: Object.keys(weights).map((k) => k.replace(/_/g, " ")),
          y: Object.values(weights),
          marker: {
            color: Object.values(weights).map((v) =>
              v >= 0.3 ? "#3b82f6" : "#555568",
            ),
          },
          hovertemplate: "%{x}<br>Weight: %{y:.3f}<extra></extra>",
        },
      ]}
      layout={{
        yaxis: {
          title: { text: "Weight", font: { size: 10 } },
          range: [0, 1],
        },
        margin: { t: 10, l: 50, r: 20, b: 80 },
        xaxis: { tickangle: -30, tickfont: { size: 10 } },
      }}
    />
  );
}
