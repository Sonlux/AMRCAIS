"use client";

import PlotlyChart from "./PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";
import type { RegimeHistoryPoint } from "@/lib/types";

interface RegimeStripChartProps {
  history: RegimeHistoryPoint[];
  height?: number;
}

/**
 * Strip chart showing regime classification over time as colored markers.
 * Used in the Regime Explorer page.
 */
export default function RegimeStripChart({
  history,
  height = 180,
}: RegimeStripChartProps) {
  return (
    <PlotlyChart
      height={height}
      data={[
        {
          x: history.map((h) => h.date),
          y: history.map(() => 1),
          type: "bar" as const,
          marker: {
            color: history.map((h) => REGIME_COLORS[h.regime] ?? "#555568"),
          },
          hovertemplate: history.map(
            (h) =>
              `${h.date}<br>${REGIME_NAMES[h.regime]}<br>Confidence: ${(h.confidence * 100).toFixed(0)}%<extra></extra>`,
          ),
        },
      ]}
      layout={{
        yaxis: { visible: false },
        xaxis: { type: "date", tickfont: { size: 10 } },
        bargap: 0,
        margin: { t: 10, l: 20, r: 20, b: 40 },
      }}
    />
  );
}
