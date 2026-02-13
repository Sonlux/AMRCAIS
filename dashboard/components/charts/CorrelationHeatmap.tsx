"use client";

import PlotlyChart from "./PlotlyChart";

interface CorrelationHeatmapProps {
  assets: string[];
  matrix: number[][];
  height?: number;
}

/**
 * Correlation matrix heatmap with red-white-blue colour scale.
 * Used in the Correlations Monitor page.
 */
export default function CorrelationHeatmap({
  assets,
  matrix,
  height = 400,
}: CorrelationHeatmapProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "heatmap",
          z: matrix,
          x: assets,
          y: assets,
          colorscale: [
            [0, "#ef4444"],
            [0.5, "#1a1a28"],
            [1, "#22c55e"],
          ],
          zmin: -1,
          zmax: 1,
          hovertemplate: "%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>",
          showscale: true,
          colorbar: {
            title: { text: "ρ", font: { size: 12, color: "#8888a0" } },
            tickfont: { size: 10, color: "#8888a0" },
            len: 0.8,
          },
        } as Plotly.Data,
      ]}
      layout={{
        xaxis: { tickfont: { size: 11 }, side: "bottom" },
        yaxis: {
          tickfont: { size: 11 },
          autorange: "reversed",
        },
        margin: { t: 10, l: 60, r: 20, b: 60 },
      }}
    />
  );
}
