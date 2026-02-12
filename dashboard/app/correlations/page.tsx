"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCorrelations, fetchAssets, fetchPrices } from "@/lib/api";
import { TRACKED_ASSETS, STALE_TIME, REFETCH_INTERVAL } from "@/lib/constants";
import { num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

const WINDOW_OPTIONS = [20, 40, 60, 90, 120];

export default function CorrelationsPage() {
  const [window, setWindow] = useState(60);

  const corrQ = useQuery({
    queryKey: ["data", "correlations", window],
    queryFn: () => fetchCorrelations(window),
    staleTime: STALE_TIME,
  });

  const assetsQ = useQuery({
    queryKey: ["data", "assets"],
    queryFn: fetchAssets,
    staleTime: STALE_TIME,
  });

  const corr = corrQ.data;

  // Compute summary stats from matrix
  const offDiag: number[] = [];
  if (corr) {
    for (let i = 0; i < corr.matrix.length; i++) {
      for (let j = 0; j < corr.matrix[i].length; j++) {
        if (i !== j) offDiag.push(corr.matrix[i][j]);
      }
    }
  }
  const avgCorr = offDiag.length
    ? offDiag.reduce((a, b) => a + b, 0) / offDiag.length
    : 0;
  const maxCorr = offDiag.length ? Math.max(...offDiag) : 0;
  const minCorr = offDiag.length ? Math.min(...offDiag) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-foreground">
            Correlation Monitor
          </h1>
          <p className="text-sm text-text-secondary">
            Cross-asset correlation heatmap and anomaly detection
          </p>
        </div>

        {/* Window selector */}
        <div className="flex items-center gap-1 rounded-lg bg-surface-elevated p-1">
          {WINDOW_OPTIONS.map((w) => (
            <button
              key={w}
              onClick={() => setWindow(w)}
              className={cn(
                "rounded px-3 py-1 text-xs font-medium transition-colors",
                window === w
                  ? "bg-surface text-foreground"
                  : "text-text-secondary hover:text-foreground",
              )}
            >
              {w}d
            </button>
          ))}
        </div>
      </div>

      {/* Summary metrics */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <MetricsCard
          label="Rolling Window"
          value={`${window} days`}
          sub="Lookback period"
        />
        <MetricsCard
          label="Avg Correlation"
          value={num(avgCorr)}
          sub="Off-diagonal mean"
          color={Math.abs(avgCorr) > 0.5 ? "#f59e0b" : undefined}
        />
        <MetricsCard
          label="Max Correlation"
          value={num(maxCorr)}
          sub="Strongest positive"
          color={maxCorr > 0.8 ? "#ef4444" : "#22c55e"}
        />
        <MetricsCard
          label="Min Correlation"
          value={num(minCorr)}
          sub="Strongest negative"
          color={minCorr < -0.5 ? "#3b82f6" : undefined}
        />
      </div>

      {/* Correlation heatmap */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Cross-Asset Correlation Matrix ({window}-day rolling)
        </p>
        {corr ? (
          <PlotlyChart
            height={420}
            data={[
              {
                type: "heatmap",
                z: corr.matrix,
                x: corr.assets,
                y: corr.assets,
                colorscale: [
                  [0, "#ef4444"],
                  [0.5, "#0a0a0f"],
                  [1, "#22c55e"],
                ],
                zmin: -1,
                zmax: 1,
                showscale: true,
                colorbar: {
                  tickfont: { color: "#8888a0", size: 10 },
                  title: {
                    text: "Correlation",
                    font: { size: 10, color: "#8888a0" },
                  },
                  len: 0.8,
                },
                hovertemplate: "%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
              } as Plotly.Data,
            ]}
            layout={{
              xaxis: {
                tickfont: { size: 11, color: "#e4e4ef" },
                side: "bottom",
              },
              yaxis: {
                tickfont: { size: 11, color: "#e4e4ef" },
                autorange: "reversed",
              },
              margin: { t: 10, l: 60, r: 20, b: 60 },
            }}
          />
        ) : corrQ.isError ? (
          <ErrorState onRetry={() => corrQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-96" />
        )}
      </div>

      {/* Correlation bar chart — top pairs */}
      {corr && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Strongest Correlations (Off-Diagonal)
          </p>
          {(() => {
            const pairs: { pair: string; value: number }[] = [];
            for (let i = 0; i < corr.assets.length; i++) {
              for (let j = i + 1; j < corr.assets.length; j++) {
                pairs.push({
                  pair: `${corr.assets[i]}/${corr.assets[j]}`,
                  value: corr.matrix[i][j],
                });
              }
            }
            pairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
            const top = pairs.slice(0, 10);

            return (
              <PlotlyChart
                height={250}
                data={[
                  {
                    type: "bar",
                    x: top.map((p) => p.pair),
                    y: top.map((p) => p.value),
                    marker: {
                      color: top.map((p) =>
                        p.value >= 0 ? "#22c55e" : "#ef4444",
                      ),
                    },
                    hovertemplate: "%{x}<br>ρ = %{y:.3f}<extra></extra>",
                  },
                ]}
                layout={{
                  yaxis: {
                    title: { text: "Correlation", font: { size: 10 } },
                    range: [-1, 1],
                  },
                  margin: { t: 10, l: 50, r: 20, b: 80 },
                  xaxis: { tickangle: -45, tickfont: { size: 10 } },
                }}
              />
            );
          })()}
        </div>
      )}
    </div>
  );
}
