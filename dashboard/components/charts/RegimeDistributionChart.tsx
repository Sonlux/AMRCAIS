"use client";

import PlotlyChart from "./PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";

interface RegimeDistributionChartProps {
  distribution: Record<string, number>;
  height?: number;
}

/**
 * Donut chart showing regime distribution (days per regime).
 * Used in the Meta-Learning page.
 */
export default function RegimeDistributionChart({
  distribution,
  height = 260,
}: RegimeDistributionChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          type: "pie",
          labels: Object.keys(distribution).map(
            (k) => REGIME_NAMES[Number(k)] ?? k,
          ),
          values: Object.values(distribution),
          marker: {
            colors: Object.keys(distribution).map(
              (k) => REGIME_COLORS[Number(k)] ?? "#555568",
            ),
          },
          hole: 0.5,
          textinfo: "percent+label",
          hovertemplate:
            "%{label}<br>%{value} days (%{percent})<extra></extra>",
        } as unknown as Plotly.Data,
      ]}
      layout={{
        showlegend: false,
        margin: { t: 10, r: 10, b: 10, l: 10 },
      }}
    />
  );
}
