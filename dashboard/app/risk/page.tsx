"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchTailRisk, fetchPortfolioOptimize } from "@/lib/api";
import {
  REGIME_COLORS,
  STALE_TIME,
  REFETCH_INTERVAL,
  TRACKED_ASSETS,
} from "@/lib/constants";
import { pct, pctRaw, num, cn } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { CardSkeleton, ChartSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function RiskPage() {
  const tailQ = useQuery({
    queryKey: ["phase3", "tail-risk"],
    queryFn: fetchTailRisk,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const portfolioQ = useQuery({
    queryKey: ["phase3", "portfolio-optimize"],
    queryFn: fetchPortfolioOptimize,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const tail = tailQ.data;
  const portfolio = portfolioQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">
          Risk &amp; Portfolio
        </h1>
        <p className="text-sm text-text-secondary">
          Tail risk analysis, scenario VaR, portfolio optimization, and hedge
          recommendations
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {tail ? (
          <>
            <MetricsCard
              label="Weighted VaR (95%)"
              value={pctRaw(tail.weighted_var)}
              color="#ef4444"
            />
            <MetricsCard
              label="Weighted CVaR (95%)"
              value={pctRaw(tail.weighted_cvar)}
              color="#ef4444"
            />
            <MetricsCard
              label="Worst Scenario"
              value={tail.worst_scenario}
              sub={`VaR: ${pctRaw(tail.worst_scenario_var)}`}
              color="#ef4444"
            />
            <MetricsCard
              label="Tail Risk Driver"
              value={tail.tail_risk_driver}
            />
            <MetricsCard
              label="Expected Sharpe"
              value={portfolio ? num(portfolio.sharpe_ratio) : "—"}
              color={
                portfolio && portfolio.sharpe_ratio >= 1 ? "#22c55e" : "#f59e0b"
              }
            />
          </>
        ) : tailQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => tailQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Scenario VaR Chart + Portfolio Weights */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Scenario VaR */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Scenario Analysis — VaR by Regime
          </p>
          {tail && tail.scenarios.length > 0 ? (
            <PlotlyChart
              height={300}
              data={[
                {
                  type: "bar" as const,
                  name: "VaR 95%",
                  x: tail.scenarios.map((s) => s.scenario_name),
                  y: tail.scenarios.map((s) => s.var_95),
                  marker: {
                    color: tail.scenarios.map(
                      (s) => REGIME_COLORS[s.scenario_regime] ?? "#6b7280",
                    ),
                  },
                  hovertemplate:
                    "%{x}<br>VaR: %{y:.2%}<br>Prob: " +
                    tail.scenarios
                      .map((s) => (s.probability * 100).toFixed(1) + "%")
                      .join(",") +
                    "<extra></extra>",
                },
                {
                  type: "bar" as const,
                  name: "CVaR 95%",
                  x: tail.scenarios.map((s) => s.scenario_name),
                  y: tail.scenarios.map((s) => s.cvar_95),
                  marker: {
                    color: tail.scenarios.map(
                      (s) =>
                        (REGIME_COLORS[s.scenario_regime] ?? "#6b7280") + "80",
                    ),
                  },
                  hovertemplate: "%{x}<br>CVaR: %{y:.2%}<extra></extra>",
                },
              ]}
              layout={{
                barmode: "group",
                yaxis: {
                  title: { text: "Loss", font: { size: 10 } },
                  tickformat: ".1%",
                },
                legend: { font: { size: 9 } },
              }}
            />
          ) : tailQ.isError ? (
            <ErrorState onRetry={() => tailQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>

        {/* Optimal Portfolio Weights */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Optimal Portfolio Allocation
          </p>
          {portfolio ? (
            <PlotlyChart
              height={300}
              data={[
                {
                  type: "pie" as const,
                  labels: Object.keys(portfolio.blended_weights),
                  values: Object.values(portfolio.blended_weights),
                  hole: 0.4,
                  marker: {
                    colors: [
                      "#22c55e",
                      "#3b82f6",
                      "#f59e0b",
                      "#8b5cf6",
                      "#ef4444",
                      "#06b6d4",
                    ],
                  },
                  textinfo: "label+percent" as const,
                  textfont: { size: 10, color: "#d4d4d8" },
                  hovertemplate:
                    "%{label}<br>Weight: %{percent}<extra></extra>",
                },
              ]}
              layout={{
                showlegend: false,
                annotations: [
                  {
                    text: "Blended",
                    font: { size: 12, color: "#8888a0" },
                    showarrow: false,
                  },
                ],
              }}
            />
          ) : portfolioQ.isError ? (
            <ErrorState onRetry={() => portfolioQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-64" />
          )}
        </div>
      </div>

      {/* Portfolio stats */}
      {portfolio && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <MetricsCard
            label="Expected Return"
            value={pctRaw(portfolio.expected_return)}
            color={portfolio.expected_return >= 0 ? "#22c55e" : "#ef4444"}
          />
          <MetricsCard
            label="Expected Volatility"
            value={pctRaw(portfolio.expected_volatility)}
          />
          <MetricsCard
            label="Max DD Constraint"
            value={pctRaw(portfolio.max_drawdown_constraint)}
            color="#ef4444"
          />
          <MetricsCard
            label="Transaction Cost"
            value={pctRaw(portfolio.transaction_cost_estimate)}
          />
        </div>
      )}

      {/* Rebalance trigger */}
      {portfolio && portfolio.rebalance_trigger && (
        <div className="rounded-lg border border-regime-3/30 bg-regime-3/5 p-4">
          <p className="text-sm font-semibold text-regime-3">
            Rebalance Triggered
          </p>
          <p className="mt-1 text-xs text-text-secondary">
            {portfolio.rebalance_reason}
          </p>
        </div>
      )}

      {/* Regime Allocations table */}
      {portfolio && portfolio.regime_allocations.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Regime-Conditional Allocations
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-text-muted">
                  <th className="pb-2 pr-4">Regime</th>
                  <th className="pb-2 pr-4">Probability</th>
                  <th className="pb-2 pr-4">Expected Return</th>
                  <th className="pb-2 pr-4">Volatility</th>
                  {TRACKED_ASSETS.map((a) => (
                    <th key={a} className="pb-2 pr-4">
                      {a}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {portfolio.regime_allocations.map((ra, i) => (
                  <tr
                    key={ra.regime}
                    className={cn(
                      "border-b border-border/50",
                      i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                    )}
                  >
                    <td className="py-2 pr-4">
                      <span
                        className="font-medium"
                        style={{ color: REGIME_COLORS[ra.regime] }}
                      >
                        {ra.regime_name}
                      </span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-text-secondary">
                      {pct(ra.probability)}
                    </td>
                    <td
                      className="py-2 pr-4 font-mono"
                      style={{
                        color: ra.expected_return >= 0 ? "#22c55e" : "#ef4444",
                      }}
                    >
                      {pctRaw(ra.expected_return)}
                    </td>
                    <td className="py-2 pr-4 font-mono text-text-secondary">
                      {pctRaw(ra.expected_volatility)}
                    </td>
                    {TRACKED_ASSETS.map((a) => (
                      <td
                        key={a}
                        className="py-2 pr-4 font-mono text-text-secondary"
                      >
                        {ra.weights[a] != null ? pct(ra.weights[a]) : "—"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Hedge Recommendations */}
      {tail && tail.hedge_recommendations.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Hedge Recommendations
          </p>
          <div className="space-y-3">
            {tail.hedge_recommendations.map((h, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-md border border-border/50 bg-surface-elevated p-3"
              >
                <span
                  className={cn(
                    "mt-0.5 rounded px-2 py-0.5 text-xs font-semibold",
                    h.priority === 1
                      ? "bg-regime-2/10 text-regime-2"
                      : h.priority === 2
                        ? "bg-regime-3/10 text-regime-3"
                        : "bg-regime-4/10 text-regime-4",
                  )}
                >
                  P{h.priority}
                </span>
                <div>
                  <p className="text-sm font-medium text-foreground">
                    {(h.hedge_type ?? "").replace(/_/g, " ")} —{" "}
                    {h.instrument ?? ""}
                  </p>
                  <p className="mt-0.5 text-xs text-text-secondary">
                    {h.rationale}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
