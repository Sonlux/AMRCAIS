"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchContagionAnalysis, fetchSpillover } from "@/lib/api";
import { STALE_TIME, REFETCH_INTERVAL } from "@/lib/constants";
import { pct, num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import SignalCard from "@/components/ui/SignalCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function ContagionPage() {
  const contagionQ = useQuery({
    queryKey: ["phase2", "contagion"],
    queryFn: fetchContagionAnalysis,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const spilloverQ = useQuery({
    queryKey: ["phase2", "spillover"],
    queryFn: fetchSpillover,
    staleTime: STALE_TIME,
  });

  const contagion = contagionQ.data;
  const spillover = spilloverQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Contagion Network
        </h1>
        <p className="text-sm text-text-secondary">
          Cross-asset contagion analysis, Granger causality, and spillover
          dynamics
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {contagion ? (
          <>
            <MetricsCard
              label="Network Density"
              value={pct(contagion.network_density)}
              color={contagion.network_density > 0.5 ? "#ef4444" : "#22c55e"}
            />
            <MetricsCard
              label="Significant Links"
              value={contagion.n_significant_links}
              sub="Granger-causal pairs"
            />
            <MetricsCard
              label="Spillover Index"
              value={spillover ? num(spillover.total_spillover_index) : "—"}
              color={
                spillover && spillover.total_spillover_index > 50
                  ? "#ef4444"
                  : "#22c55e"
              }
            />
            <MetricsCard
              label="Contagion Flags"
              value={Object.keys(contagion.contagion_flags).filter(k => contagion.contagion_flags[k]).length}
              color={
                Object.keys(contagion.contagion_flags).filter(k => contagion.contagion_flags[k]).length > 0 ? "#f59e0b" : "#22c55e"
              }
            />
            <SignalCard
              module="Contagion"
              signal={contagion.signal.signal}
              strength={contagion.signal.strength}
              className="col-span-1"
            />
          </>
        ) : contagionQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => contagionQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Contagion flags */}
      {contagion && (() => {
        const activeFlags = Object.entries(contagion.contagion_flags)
          .filter(([, v]) => v)
          .map(([k]) => k);
        return activeFlags.length > 0 ? (
        <div className="rounded-lg border border-regime-3/30 bg-regime-3/5 p-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-regime-3">
            Active Contagion Flags
          </p>
          <ul className="space-y-1">
            {activeFlags.map((flag, i) => (
              <li key={i} className="text-sm text-text-secondary">
                • {flag}
              </li>
            ))}
          </ul>
        </div>
        ) : null;
      })()}

      {/* Network Graph + Spillover Heatmap */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Network graph */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Granger Causality Network
          </p>
          {contagion && contagion.granger_network.length > 0 ? (
            <PlotlyChart
              height={350}
              data={(() => {
                const assets = [
                  ...new Set([
                    ...contagion.granger_network.map((l) => l.cause),
                    ...contagion.granger_network.map((l) => l.effect),
                  ]),
                ];
                const n = assets.length;
                const angleStep = (2 * Math.PI) / n;
                const pos: Record<string, { x: number; y: number }> = {};
                assets.forEach((a, i) => {
                  pos[a] = {
                    x: Math.cos(i * angleStep),
                    y: Math.sin(i * angleStep),
                  };
                });

                const edgeTraces = contagion.granger_network.map((link) => ({
                  type: "scatter" as const,
                  mode: "lines" as const,
                  x: [pos[link.cause]?.x, pos[link.effect]?.x],
                  y: [pos[link.cause]?.y, pos[link.effect]?.y],
                  line: {
                    color: `rgba(99,102,241,${Math.max(0.2, 1 - link.p_value)})`,
                    width: Math.max(1, link.f_stat / 5),
                  },
                  showlegend: false,
                  hoverinfo: "text" as const,
                  text: `${link.cause} → ${link.effect}<br>p=${link.p_value.toFixed(3)}, F=${link.f_stat.toFixed(2)}, lag=${link.lag}`,
                }));

                const nodeTrace = {
                  type: "scatter" as const,
                  mode: "text+markers" as const,
                  x: assets.map((a) => pos[a]?.x),
                  y: assets.map((a) => pos[a]?.y),
                  text: assets,
                  textposition: "top center" as const,
                  textfont: { color: "#8888a0", size: 11 },
                  marker: {
                    size: 16,
                    color: "#6366f1",
                    line: { color: "#1a1a28", width: 2 },
                  },
                  showlegend: false,
                  hoverinfo: "text" as const,
                };

                return [...edgeTraces, nodeTrace];
              })()}
              layout={{
                xaxis: { visible: false },
                yaxis: { visible: false, scaleanchor: "x" as const },
                showlegend: false,
              }}
            />
          ) : contagionQ.isError ? (
            <ErrorState onRetry={() => contagionQ.refetch()} />
          ) : contagion ? (
            <p className="text-sm text-text-secondary">
              No significant Granger links detected
            </p>
          ) : (
            <ChartSkeleton height="h-80" />
          )}
        </div>

        {/* Spillover heatmap */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Spillover Matrix
          </p>
          {spillover && spillover.assets.length > 0 ? (
            <PlotlyChart
              height={350}
              data={[
                {
                  type: "heatmap" as const,
                  z: spillover.pairwise.length > 0
                    ? spillover.pairwise
                    : spillover.assets.map(() =>
                        spillover.assets.map(() => 0),
                      ),
                  x: spillover.assets,
                  y: spillover.assets,
                  colorscale: [
                    [0, "#0a0a0f"],
                    [0.5, "#6366f1"],
                    [1, "#ef4444"],
                  ],
                  hovertemplate:
                    "%{y} → %{x}<br>Spillover: %{z:.2f}<extra></extra>",
                  colorbar: { thickness: 12, tickfont: { size: 9 } },
                },
              ]}
              layout={{
                xaxis: { tickfont: { size: 10 } },
                yaxis: { tickfont: { size: 10 } },
              }}
            />
          ) : spilloverQ.isError ? (
            <ErrorState onRetry={() => spilloverQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-80" />
          )}
        </div>
      </div>

      {/* Net Spillover Bar Chart */}
      {spillover && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Net Spillover (Transmitter vs Receiver)
          </p>
          <PlotlyChart
            height={260}
            data={[
              {
                type: "bar" as const,
                x: Object.keys(spillover.net_spillover),
                y: Object.values(spillover.net_spillover),
                marker: {
                  color: Object.values(spillover.net_spillover).map((v) =>
                    v >= 0 ? "#ef4444" : "#3b82f6",
                  ),
                },
                hovertemplate: "%{x}<br>Net Spillover: %{y:.2f}<extra></extra>",
              },
            ]}
            layout={{
              yaxis: {
                title: { text: "Net Spillover", font: { size: 10 } },
                zeroline: true,
                zerolinecolor: "#6b7280",
              },
              annotations: [
                {
                  xref: "paper" as const,
                  yref: "paper" as const,
                  x: 1,
                  y: 1,
                  text: "Positive = net transmitter",
                  showarrow: false,
                  font: { size: 9, color: "#6b7280" },
                },
              ],
            }}
          />
        </div>
      )}

      {/* Granger links table */}
      {contagion && contagion.granger_network.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Significant Granger Links ({contagion.granger_network.length})
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-muted">
                  <th className="pb-2 pr-4">Source</th>
                  <th className="pb-2 pr-4">Target</th>
                  <th className="pb-2 pr-4">p-value</th>
                  <th className="pb-2 pr-4">F-stat</th>
                  <th className="pb-2">Lag</th>
                </tr>
              </thead>
              <tbody>
                {contagion.granger_network
                  .sort((a, b) => a.p_value - b.p_value)
                  .map((link, i) => (
                    <tr
                      key={i}
                      className={cn(
                        "border-b border-border/50",
                        i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                      )}
                    >
                      <td className="py-2 pr-4 font-medium text-foreground">
                        {link.cause}
                      </td>
                      <td className="py-2 pr-4 font-medium text-foreground">
                        {link.effect}
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {link.p_value.toFixed(4)}
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {link.f_stat.toFixed(2)}
                      </td>
                      <td className="py-2 font-mono text-text-secondary">
                        {link.lag}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
