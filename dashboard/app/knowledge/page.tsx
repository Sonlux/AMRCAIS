"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchKnowledgeSummary,
  fetchTransitions,
  fetchAnomalies,
} from "@/lib/api";
import { REGIME_COLORS, REGIME_NAMES, STALE_TIME } from "@/lib/constants";
import { pct, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import RegimeBadge from "@/components/ui/RegimeBadge";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, TableSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function KnowledgePage() {
  const summaryQ = useQuery({
    queryKey: ["phase5", "knowledge-summary"],
    queryFn: fetchKnowledgeSummary,
    staleTime: STALE_TIME,
  });

  const transQ = useQuery({
    queryKey: ["phase5", "transitions"],
    queryFn: () => fetchTransitions(50),
    staleTime: STALE_TIME,
  });

  const anomalyQ = useQuery({
    queryKey: ["phase5", "anomalies"],
    queryFn: () => fetchAnomalies(50),
    staleTime: STALE_TIME,
  });

  const anomalies = anomalyQ.data;
  const summary = summaryQ.data;
  const transitions = transQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Knowledge Base
        </h1>
        <p className="text-sm text-text-secondary">
          Historical regime transitions, correlation anomalies, and macro impact
          tracking
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {summary ? (
          <>
            <MetricsCard
              label="Total Transitions"
              value={summary.total_transitions}
            />
            <MetricsCard
              label="Total Anomalies"
              value={summary.total_anomalies}
            />
            <MetricsCard
              label="Macro Impacts"
              value={summary.total_macro_impacts}
            />
            <MetricsCard
              label="Regime Coverage"
              value={Object.keys(summary.regime_coverage).length}
              sub="regimes with data"
            />
          </>
        ) : summaryQ.isError ? (
          <div className="col-span-4">
            <ErrorState onRetry={() => summaryQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Regime coverage chart */}
      {summary && Object.keys(summary.regime_coverage).length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Knowledge Coverage by Regime
          </p>
          <PlotlyChart
            height={250}
            data={[
              {
                type: "bar" as const,
                x: Object.keys(summary.regime_coverage).map(
                  (k) => REGIME_NAMES[Number(k)] ?? k,
                ),
                y: Object.values(summary.regime_coverage),
                marker: {
                  color: Object.keys(summary.regime_coverage).map(
                    (k) => REGIME_COLORS[Number(k)] ?? "#6b7280",
                  ),
                },
                hovertemplate: "%{x}<br>Records: %{y}<extra></extra>",
              },
            ]}
            layout={{
              yaxis: {
                title: { text: "Records", font: { size: 10 } },
              },
            }}
          />
        </div>
      )}

      {/* Transition History + Anomalies */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Transition history */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Recent Transitions ({transitions?.total ?? 0})
          </p>
          {transitions && transitions.transitions.length > 0 ? (
            <div className="max-h-96 space-y-2 overflow-y-auto">
              {transitions.transitions.map((t) => (
                <div
                  key={t.id}
                  className="rounded-md border border-border/50 bg-surface-elevated p-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <RegimeBadge regime={t.from_regime} size="sm" />
                      <span className="text-text-muted">→</span>
                      <RegimeBadge regime={t.to_regime} size="sm" />
                    </div>
                    <span className="text-xs text-text-muted">
                      {t.timestamp}
                    </span>
                  </div>
                  <div className="mt-2 flex gap-4 text-xs text-text-secondary">
                    <span>Confidence: {pct(t.confidence)}</span>
                    <span>Disagreement: {pct(t.disagreement)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : transitions ? (
            <p className="text-sm text-text-secondary">
              No transition records yet
            </p>
          ) : (
            <TableSkeleton rows={5} />
          )}
        </div>

        {/* Anomalies */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Correlation Anomalies ({anomalies?.total ?? 0})
          </p>
          {anomalies && anomalies.anomalies.length > 0 ? (
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface">
                  <tr className="border-b border-border text-left text-xs text-text-muted">
                    <th className="pb-2 pr-3">Type</th>
                    <th className="pb-2 pr-3">Pair</th>
                    <th className="pb-2 pr-3">Z-Score</th>
                    <th className="pb-2 pr-3">Expected</th>
                    <th className="pb-2 pr-3">Actual</th>
                    <th className="pb-2">Regime</th>
                  </tr>
                </thead>
                <tbody>
                  {anomalies.anomalies.map((a, i) => (
                    <tr
                      key={a.id}
                      className={cn(
                        "border-b border-border/50",
                        i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                      )}
                    >
                      <td className="py-1.5 pr-3 text-xs text-foreground">
                        {a.anomaly_type}
                      </td>
                      <td className="py-1.5 pr-3 font-medium text-foreground">
                        {a.asset_pair}
                      </td>
                      <td
                        className="py-1.5 pr-3 font-mono"
                        style={{
                          color:
                            Math.abs(a.z_score) > 2 ? "#ef4444" : "#f59e0b",
                        }}
                      >
                        {a.z_score.toFixed(2)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">
                        {a.expected_value.toFixed(3)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono text-text-secondary">
                        {a.actual_value.toFixed(3)}
                      </td>
                      <td className="py-1.5">
                        <RegimeBadge regime={a.regime} size="sm" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : anomalies ? (
            <p className="text-sm text-text-secondary">
              No anomalies recorded yet
            </p>
          ) : (
            <TableSkeleton rows={5} />
          )}
        </div>
      </div>
    </div>
  );
}
