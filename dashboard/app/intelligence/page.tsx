"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchTransitionForecast,
  fetchMultiTimeframe,
  fetchSurpriseIndex,
  fetchActiveSurprises,
  fetchDecayCurves,
  fetchNarrative,
} from "@/lib/api";
import {
  REGIME_COLORS,
  STALE_TIME,
  REFETCH_INTERVAL,
} from "@/lib/constants";
import { pct, num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import RegimeBadge from "@/components/ui/RegimeBadge";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function IntelligencePage() {
  const forecastQ = useQuery({
    queryKey: ["phase2", "transition-forecast"],
    queryFn: () => fetchTransitionForecast(30),
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const mtfQ = useQuery({
    queryKey: ["phase2", "multi-timeframe"],
    queryFn: fetchMultiTimeframe,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const surpriseQ = useQuery({
    queryKey: ["phase2", "surprise-index"],
    queryFn: fetchSurpriseIndex,
    staleTime: STALE_TIME,
  });

  const activeQ = useQuery({
    queryKey: ["phase2", "active-surprises"],
    queryFn: fetchActiveSurprises,
    staleTime: STALE_TIME,
  });

  const decayQ = useQuery({
    queryKey: ["phase2", "decay-curves"],
    queryFn: () => fetchDecayCurves(30),
    staleTime: STALE_TIME,
  });

  const narrativeQ = useQuery({
    queryKey: ["phase2", "narrative"],
    queryFn: fetchNarrative,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const forecast = forecastQ.data;
  const mtf = mtfQ.data;
  const surprise = surpriseQ.data;
  const active = activeQ.data;
  const decay = decayQ.data;
  const narrative = narrativeQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Intelligence Hub
        </h1>
        <p className="text-sm text-text-secondary">
          Transition forecasts, multi-timeframe analysis, surprise decay, and
          market narrative
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {forecast ? (
          <>
            <MetricsCard
              label="Transition Risk"
              value={forecast.transition_risk}
              color={
                forecast.transition_risk === "high"
                  ? "#ef4444"
                  : forecast.transition_risk === "medium"
                    ? "#f59e0b"
                    : "#22c55e"
              }
            />
            <MetricsCard
              label="Most Likely Next"
              value={forecast.most_likely_next_name}
              color={REGIME_COLORS[forecast.most_likely_next]}
            />
            <MetricsCard
              label="Forecast Confidence"
              value={pct(forecast.confidence)}
            />
            <MetricsCard
              label="Surprise Index"
              value={surprise ? num(surprise.index) : "—"}
              sub={surprise?.direction}
              color={
                surprise
                  ? surprise.direction === "positive"
                    ? "#22c55e"
                    : surprise.direction === "negative"
                      ? "#ef4444"
                      : "#6b7280"
                  : undefined
              }
            />
            <MetricsCard
              label="Active Surprises"
              value={surprise?.active_surprises ?? "—"}
              sub={`of ${surprise?.total_historical ?? 0} total`}
            />
          </>
        ) : forecastQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => forecastQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Transition Forecast + Multi-Timeframe */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Transition probabilities */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Transition Probabilities ({forecast?.horizon_days ?? 30}d Horizon)
          </p>
          {forecast ? (
            <PlotlyChart
              height={300}
              data={[
                {
                  type: "bar" as const,
                  x: Object.keys(forecast.blended_probs),
                  y: Object.values(forecast.blended_probs),
                  marker: {
                    color: Object.keys(forecast.blended_probs).map(
                      (_, i) => REGIME_COLORS[i + 1] ?? "#6b7280",
                    ),
                  },
                  hovertemplate: "%{x}<br>Probability: %{y:.1%}<extra></extra>",
                },
              ]}
              layout={{
                yaxis: {
                  title: { text: "Probability", font: { size: 10 } },
                  tickformat: ".0%",
                },
              }}
            />
          ) : forecastQ.isError ? (
            <ErrorState onRetry={() => forecastQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>

        {/* Multi-Timeframe */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Multi-Timeframe Regime Analysis
          </p>
          {mtf ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">
                  Agreement Score
                </span>
                <span
                  className="font-mono text-lg font-semibold"
                  style={{
                    color: mtf.agreement_score > 0.7 ? "#22c55e" : "#f59e0b",
                  }}
                >
                  {pct(mtf.agreement_score)}
                </span>
              </div>
              {mtf.conflict_detected && (
                <div className="rounded-md border border-regime-3/30 bg-regime-3/5 px-3 py-2 text-xs text-regime-3">
                  Timeframe conflict detected — trade with caution
                </div>
              )}
              {(["daily", "weekly", "monthly"] as const).map((tf) => (
                <div
                  key={tf}
                  className="flex items-center justify-between rounded-md bg-surface-elevated px-3 py-2"
                >
                  <span className="text-sm font-medium capitalize text-text-secondary">
                    {tf}
                  </span>
                  <div className="flex items-center gap-3">
                    <RegimeBadge
                      regime={mtf[tf].regime}
                      confidence={mtf[tf].confidence}
                      size="sm"
                    />
                  </div>
                </div>
              ))}
              <div className="flex items-center justify-between rounded-md border border-accent/20 bg-accent/5 px-3 py-2">
                <span className="text-sm font-medium text-text-secondary">
                  Trade Signal
                </span>
                <span className="text-sm font-semibold text-accent">
                  {mtf.trade_signal}
                </span>
              </div>
            </div>
          ) : mtfQ.isError ? (
            <ErrorState onRetry={() => mtfQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>
      </div>

      {/* Surprise Decay Curves */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Surprise Decay Curves
        </p>
        {decay && Object.keys(decay.curves).length > 0 ? (
          <PlotlyChart
            height={300}
            data={Object.entries(decay.curves).map(([indicator, points], i) => ({
              type: "scatter" as const,
              mode: "lines" as const,
              x: points.map((p) => p.day),
              y: points.map((p) => p.impact),
              name: indicator,
              line: {
                color: [
                  "#6366f1",
                  "#22c55e",
                  "#ef4444",
                  "#f59e0b",
                  "#3b82f6",
                  "#8b5cf6",
                ][i % 6],
              },
              hovertemplate: `${indicator}<br>Day %{x}<br>Impact: %{y:.4f}<extra></extra>`,
            }))}
            layout={{
              xaxis: { title: { text: "Days Forward", font: { size: 10 } } },
              yaxis: { title: { text: "Impact", font: { size: 10 } } },
              showlegend: true,
              legend: { font: { size: 9 } },
            }}
          />
        ) : decayQ.isError ? (
          <ErrorState onRetry={() => decayQ.refetch()} />
        ) : decay ? (
          <p className="text-sm text-text-secondary">
            No active surprises to display
          </p>
        ) : (
          <ChartSkeleton height="h-64" />
        )}
      </div>

      {/* Active Surprises Table */}
      {active && active.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Active Macro Surprises ({active.length})
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-muted">
                  <th className="pb-2 pr-4">Indicator</th>
                  <th className="pb-2 pr-4">Surprise</th>
                  <th className="pb-2 pr-4">Release Date</th>
                  <th className="pb-2 pr-4">Half-Life (days)</th>
                  <th className="pb-2 pr-4">Weight</th>
                  <th className="pb-2">Regime</th>
                </tr>
              </thead>
              <tbody>
                {active.map((s, i) => (
                  <tr
                    key={i}
                    className={cn(
                      "border-b border-border/50",
                      i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                    )}
                  >
                    <td className="py-2 pr-4 font-medium text-foreground">
                      {s.indicator}
                    </td>
                    <td
                      className="py-2 pr-4 font-mono"
                      style={{
                        color: s.surprise >= 0 ? "#22c55e" : "#ef4444",
                      }}
                    >
                      {s.surprise >= 0 ? "+" : ""}
                      {s.surprise.toFixed(3)}
                    </td>
                    <td className="py-2 pr-4 text-text-secondary">
                      {s.release_date}
                    </td>
                    <td className="py-2 pr-4 font-mono text-text-secondary">
                      {s.half_life_days.toFixed(1)}
                    </td>
                    <td className="py-2 pr-4 font-mono text-text-secondary">
                      {s.initial_weight.toFixed(3)}
                    </td>
                    <td className="py-2">
                      <RegimeBadge regime={s.regime_at_release} size="sm" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Market Narrative */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Market Narrative
        </p>
        {narrative ? (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold text-foreground">
              {narrative.headline}
            </h2>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-md bg-surface-elevated p-3">
                <p className="mb-1 text-xs font-medium uppercase text-text-muted">
                  Regime
                </p>
                <p className="text-sm text-text-secondary">
                  {narrative.regime_section}
                </p>
              </div>
              <div className="rounded-md bg-surface-elevated p-3">
                <p className="mb-1 text-xs font-medium uppercase text-text-muted">
                  Signals
                </p>
                <p className="text-sm text-text-secondary">
                  {narrative.signal_section}
                </p>
              </div>
              <div className="rounded-md bg-surface-elevated p-3">
                <p className="mb-1 text-xs font-medium uppercase text-text-muted">
                  Risk
                </p>
                <p className="text-sm text-text-secondary">
                  {narrative.risk_section}
                </p>
              </div>
              <div className="rounded-md bg-surface-elevated p-3">
                <p className="mb-1 text-xs font-medium uppercase text-text-muted">
                  Positioning
                </p>
                <p className="text-sm text-text-secondary">
                  {narrative.positioning_section}
                </p>
              </div>
            </div>
            {narrative.data_sources.length > 0 && (
              <p className="text-xs text-text-muted">
                Sources: {narrative.data_sources.join(", ")}
              </p>
            )}
          </div>
        ) : narrativeQ.isError ? (
          <ErrorState onRetry={() => narrativeQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-40" />
        )}
      </div>
    </div>
  );
}
