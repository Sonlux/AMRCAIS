"use client";

import PlotlyChart from "./PlotlyChart";
import type { VolSurfaceDataResponse } from "@/lib/types";
import { REGIME_COLORS } from "@/lib/constants";

interface VolSurface3DChartProps {
  data: VolSurfaceDataResponse;
  height?: number;
}

/**
 * 3D implied-volatility surface rendered with Plotly surface trace.
 *
 * X-axis = Moneyness (0.80 → 1.20)
 * Y-axis = Days to expiration
 * Z-axis = Implied Volatility (%)
 *
 * The colour scale adapts to the current regime:
 *   Risk-On green, Risk-Off red, Stagflation amber, Disinfl. blue.
 */
export default function VolSurface3DChart({
  data,
  height = 440,
}: VolSurface3DChartProps) {
  const accent = REGIME_COLORS[data.regime] ?? "#3b82f6";

  // Colourscale gradient: dark background → regime accent
  const colorscale: Array<[number, string]> = [
    [0, "#0d0d14"],
    [0.25, `${accent}44`],
    [0.5, `${accent}88`],
    [0.75, accent],
    [1, "#ffffff"],
  ];

  // Moneyness labels for tick text
  const mLabels = data.moneyness.map((m) => `${(m * 100).toFixed(0)}%`);

  const traces: Plotly.Data[] = [
    {
      type: "surface",
      x: data.moneyness,
      y: data.expiry_days,
      z: data.iv_grid,
      colorscale,
      showscale: true,
      colorbar: {
        title: { text: "IV %", font: { size: 11, color: "#8888a0" } },
        tickfont: { size: 10, color: "#8888a0" },
        thickness: 14,
        len: 0.6,
        outlinewidth: 0,
      },
      contours: {
        z: { show: true, usecolormap: true, highlightcolor: "#e4e4ef", project: { z: false } },
      },
      hovertemplate:
        "Moneyness: %{x:.0%}<br>DTE: %{y}d<br>IV: %{z:.1f}%<extra></extra>",
      lighting: {
        ambient: 0.6,
        diffuse: 0.5,
        specular: 0.3,
        roughness: 0.4,
      },
    } as Plotly.Data,
  ];

  return (
    <PlotlyChart
      data={traces}
      height={height}
      layout={{
        scene: {
          xaxis: {
            title: { text: "Moneyness", font: { size: 11, color: "#8888a0" } },
            tickvals: data.moneyness,
            ticktext: mLabels,
            tickfont: { size: 9, color: "#8888a0" },
            gridcolor: "#1a1a28",
            backgroundcolor: "transparent",
            showbackground: false,
          },
          yaxis: {
            title: { text: "DTE (days)", font: { size: 11, color: "#8888a0" } },
            tickfont: { size: 9, color: "#8888a0" },
            gridcolor: "#1a1a28",
            backgroundcolor: "transparent",
            showbackground: false,
          },
          zaxis: {
            title: { text: "IV (%)", font: { size: 11, color: "#8888a0" } },
            tickfont: { size: 9, color: "#8888a0" },
            gridcolor: "#1a1a28",
            backgroundcolor: "transparent",
            showbackground: false,
          },
          camera: {
            eye: { x: 1.6, y: -1.8, z: 0.9 },
            center: { x: 0, y: 0, z: -0.15 },
          },
          bgcolor: "transparent",
        },
        margin: { t: 10, r: 10, b: 10, l: 10 },
      }}
      config={{ displayModeBar: true, modeBarButtonsToRemove: ["toImage"] }}
    />
  );
}
