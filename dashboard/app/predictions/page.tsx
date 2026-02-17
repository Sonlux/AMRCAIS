"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchReturnForecasts,
  fetchAlphaSignals,
} from "@/lib/api";
import {
  REGIME_COLORS,
  REGIME_NAMES,
  STALE_TIME,
  REFETCH_INTERVAL,
  SIGNAL_COLORS,
} from "@/lib/constants";
import { pct, num } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function PredictionsPage() {
  const forecastsQ = useQuery({
    queryKey: ["phase3", "return-forecasts"],
    queryFn: fetchReturnForecasts,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const alphaQ = useQuery({
    queryKey: ["phase3", "alpha-signals"],
    queryFn: fetchAlphaSignals,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const forecasts = forecastsQ.data;
  const alpha = alphaQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Prediction Engine
        </h1>
        <p className="text-sm text-text-secondary">
          Regime-conditional return forecasts and alpha signal detection
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {forecasts ? (
          <>
            <MetricsCard
              label="Current Regime"
              value={forecasts.regime_name}
              color={REGIME_COLORS[forecasts.current_regime]}
            />
            <MetricsCard
              label="Assets Forecasted"
              value={forecasts.forecasts.length}
            />
            <MetricsCard
              label="Composite Alpha"
              value={alpha ? num(alpha.composite_score) : "—"}
              color={
                alpha
                  ? alpha.composite_score > 0
                    ? "#22c55e"
                    : "#ef4444"
                  : undefined
              }
            />
            <MetricsCard
              label="Active Anomalies"
              value={alpha?.n_active_anomalies ?? "—"}
            />
            <MetricsCard
              label="Top Signal"
              value={alpha?.top_signal ?? "—"}
              sub={alpha?.regime_context}
            />
          </>
        ) : forecastsQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => forecastsQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Return Forecasts Chart */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Expected Returns by Asset (Regime-Conditional)
        </p>
        {forecasts && forecasts.forecasts.length > 0 ? (
          <PlotlyChart
            height={320}
            data={[
              {
                type: "bar" as const,
                x: forecasts.forecasts.map((f) => f.asset),
                y: forecasts.forecasts.map((f) => f.expected_return),
                marker: {
                  color: forecasts.forecasts.map((f) =>
                    f.expected_return >= 0 ? "#22c55e" : "#ef4444",
                  ),
                },
                error_y: {
                  type: "data" as const,
                  array: forecasts.forecasts.map((f) => f.volatility),
                  visible: true,
                  color: "#6b7280",
                },
                hovertemplate:
                  "%{x}<br>Expected: %{y:.2%}<br>Vol: ±%{error_y.array:.2%}<extra></extra>",
              },
            ]}
            layout={{
              yaxis: {
                title: { text: "Expected Return", font: { size: 10 } },
                tickformat: ".1%",
                zeroline: true,
                zerolinecolor: "#6b7280",
              },
            }}
          />
        ) : forecastsQ.isError ? (
          <ErrorState onRetry={() => forecastsQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-72" />
        )}
      </div>

      {/* Forecast detail cards */}
      {forecasts && forecasts.forecasts.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {forecasts.forecasts.map((f) => (
            <div
              key={f.asset}
              className="rounded-lg border border-border bg-surface p-4"
            >
              <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-semibold text-foreground">
                  {f.asset}
                </span>
                <span
                  className="rounded px-2 py-0.5 text-xs font-semibold"
                  style={{
                    backgroundColor:
                      f.expected_return >= 0
                        ? "rgba(34,197,94,0.1)"
                        : "rgba(239,68,68,0.1)",
                    color: f.expected_return >= 0 ? "#22c55e" : "#ef4444",
                  }}
                >
                  {f.expected_return >= 0 ? "+" : ""}
                  {(f.expected_return * 100).toFixed(2)}%
                </span>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-text-muted">Volatility</span>
                  <span className="font-mono text-text-secondary">
                    {(f.volatility * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">R² (Regime)</span>
                  <span className="font-mono text-text-secondary">
                    {f.r_squared_regime.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">R² Improvement</span>
                  <span
                    className="font-mono"
                    style={{
                      color:
                        f.r_squared_improvement > 0 ? "#22c55e" : "#ef4444",
                    }}
                  >
                    {f.r_squared_improvement > 0 ? "+" : ""}
                    {(f.r_squared_improvement * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Kelly Fraction</span>
                  <span className="font-mono text-text-secondary">
                    {(f.kelly_fraction * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-text-muted">Confidence</span>
                  <span className="font-mono text-text-secondary">
                    {pct(f.confidence)}
                  </span>
                </div>
              </div>
              {/* Factor contributions */}
              {Object.keys(f.factor_contributions).length > 0 && (
                <div className="mt-3 border-t border-border/50 pt-2">
                  <p className="mb-1 text-xs text-text-muted">
                    Factor Contributions
                  </p>
                  <div className="space-y-1">
                    {Object.entries(f.factor_contributions).map(
                      ([factor, contrib]) => (
                        <div
                          key={factor}
                          className="flex items-center justify-between"
                        >
                          <span className="text-xs text-text-secondary">
                            {factor}
                          </span>
                          <span
                            className="font-mono text-xs"
                            style={{
                              color:
                                (contrib as number) >= 0
                                  ? "#22c55e"
                                  : "#ef4444",
                            }}
                          >
                            {(contrib as number) >= 0 ? "+" : ""}
                            {((contrib as number) * 100).toFixed(3)}%
                          </span>
                        </div>
                      ),
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Alpha Signals */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Alpha Signals
        </p>
        {alpha && alpha.signals.length > 0 ? (
          <div className="space-y-3">
            {alpha.signals.map((sig, i) => {
              const dirColor = SIGNAL_COLORS[sig.direction] ?? "#6b7280";
              return (
                <div
                  key={i}
                  className="rounded-md border border-border/50 bg-surface-elevated p-3"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className="rounded px-2 py-0.5 text-xs font-semibold"
                        style={{
                          backgroundColor: `${dirColor}18`,
                          color: dirColor,
                        }}
                      >
                        {sig.direction}
                      </span>
                      <span className="text-sm font-medium text-foreground">
                        {sig.anomaly_type.replace(/_/g, " ")}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono text-xs text-text-secondary">
                        Strength: {pct(sig.strength)}
                      </span>
                      <span className="font-mono text-xs text-text-secondary">
                        Win Rate: {pct(sig.historical_win_rate)}
                      </span>
                    </div>
                  </div>
                  <p className="mt-1 text-xs text-text-secondary">
                    {sig.rationale}
                  </p>
                  <div className="mt-2 flex gap-4 text-xs text-text-muted">
                    <span>Confidence: {pct(sig.confidence)}</span>
                    <span>Hold: {sig.holding_period_days}d</span>
                    <span>
                      Regime: {REGIME_NAMES[sig.regime] ?? sig.regime}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        ) : alphaQ.isError ? (
          <ErrorState onRetry={() => alphaQ.refetch()} />
        ) : alpha ? (
          <p className="text-sm text-text-secondary">
            No active alpha signals detected
          </p>
        ) : (
          <ChartSkeleton height="h-40" />
        )}
      </div>
    </div>
  );
}
