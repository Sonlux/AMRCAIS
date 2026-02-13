"use client";

import PlotlyChart from "./PlotlyChart";
import type { YieldCurveDataResponse } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/constants";

interface YieldCurveSurfaceChartProps {
  data: YieldCurveDataResponse;
  height?: number;
}

/**
 * 3D-styled yield curve visualization.
 *
 * Renders the current yield curve as a filled area line chart with
 * tenor on the x-axis and yield on the y-axis. The curve color and
 * fill adapt to the current regime. Key metrics (slope, shape) are
 * annotated directly on the chart.
 */
export default function YieldCurveSurfaceChart({
  data,
  height = 360,
}: YieldCurveSurfaceChartProps) {
  const color = REGIME_COLORS[data.regime] ?? "#3b82f6";

  // Convert tenor numbers to labels
  const tenorLabels = data.tenors.map((t) => {
    if (t < 1) return `${Math.round(t * 12)}M`;
    return `${Math.round(t)}Y`;
  });

  const traces: Plotly.Data[] = [
    // Filled area under the curve
    {
      x: tenorLabels,
      y: data.yields,
      type: "scatter",
      mode: "lines+markers",
      fill: "tozeroy",
      fillcolor: `${color}18`,
      line: { color, width: 3, shape: "spline" },
      marker: { color, size: 6 },
      name: "Yield Curve",
      hovertemplate: "%{x}: %{y:.3f}%<extra></extra>",
    },
  ];

  // Add a reference line at zero
  if (data.yields.some((y) => y < 0)) {
    traces.push({
      x: [tenorLabels[0], tenorLabels[tenorLabels.length - 1]],
      y: [0, 0],
      type: "scatter",
      mode: "lines",
      line: { color: "#6b7280", width: 1, dash: "dot" },
      showlegend: false,
      hoverinfo: "skip",
    });
  }

  // Shape badge annotation
  const shapeLabel =
    data.curve_shape.charAt(0).toUpperCase() + data.curve_shape.slice(1);

  const slopeText =
    data.slope_2_10 != null
      ? `2s10s: ${data.slope_2_10 > 0 ? "+" : ""}${data.slope_2_10.toFixed(2)}%`
      : "";

  const annotations: Partial<Plotly.Annotations>[] = [
    {
      text: `<b>${shapeLabel}</b>${slopeText ? `  ·  ${slopeText}` : ""}`,
      xref: "paper",
      yref: "paper",
      x: 0,
      y: 1.06,
      showarrow: false,
      font: { size: 11, color: "#8888a0" },
    },
  ];

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        xaxis: {
          title: { text: "Tenor", font: { size: 11, color: "#8888a0" } },
          gridcolor: "#1a1a28",
          tickfont: { size: 10, color: "#8888a0" },
        },
        yaxis: {
          title: { text: "Yield (%)", font: { size: 11, color: "#8888a0" } },
          gridcolor: "#1a1a28",
          tickformat: ".2f",
          tickfont: { size: 10, color: "#8888a0" },
        },
        annotations,
        margin: { t: 40, r: 20, b: 50, l: 55 },
        showlegend: false,
      }}
    />
  );
}
