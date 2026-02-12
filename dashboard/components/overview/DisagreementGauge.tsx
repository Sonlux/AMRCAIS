"use client";

import PlotlyChart from "@/components/charts/PlotlyChart";
import { REGIME_COLORS, REGIME_NAMES } from "@/lib/constants";

interface DisagreementGaugeProps {
  value: number;
  threshold: number;
}

export default function DisagreementGauge({
  value,
  threshold,
}: DisagreementGaugeProps) {
  const pctValue = Math.round(value * 100);
  const color =
    value > threshold
      ? "#ef4444"
      : value > threshold * 0.7
        ? "#f59e0b"
        : "#22c55e";

  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-text-muted">
        Regime Disagreement
      </p>
      <PlotlyChart
        height={180}
        data={[
          {
            type: "indicator",
            mode: "gauge+number",
            value: pctValue,
            number: { suffix: "%", font: { size: 28, color: "#e4e4ef" } },
            gauge: {
              axis: {
                range: [0, 100],
                tickcolor: "#2a2a3a",
                dtick: 25,
                tickfont: { size: 10, color: "#555568" },
              },
              bar: { color, thickness: 0.6 },
              bgcolor: "#1a1a28",
              borderwidth: 0,
              steps: [
                { range: [0, threshold * 70], color: "rgba(34,197,94,0.08)" },
                {
                  range: [threshold * 70, threshold * 100],
                  color: "rgba(245,158,11,0.08)",
                },
                {
                  range: [threshold * 100, 100],
                  color: "rgba(239,68,68,0.08)",
                },
              ],
              threshold: {
                line: { color: "#ef4444", width: 2 },
                thickness: 0.8,
                value: threshold * 100,
              },
            },
          } as Plotly.Data,
        ]}
        layout={{
          margin: { t: 10, r: 20, b: 10, l: 20 },
        }}
      />
      {value > threshold && (
        <p className="mt-1 text-center text-xs font-medium text-regime-3">
          ⚠ High disagreement — potential regime transition
        </p>
      )}
    </div>
  );
}
