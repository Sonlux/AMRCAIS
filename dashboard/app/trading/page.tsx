"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchPortfolio,
  fetchPortfolioMetrics,
  fetchPortfolioEquity,
  fetchRegimeAttribution,
  fetchTrades,
} from "@/lib/api";
import { REGIME_COLORS, STALE_TIME, REFETCH_INTERVAL } from "@/lib/constants";
import { pct, pctRaw, num, cn, currency } from "@/lib/utils";

import MetricsCard from "@/components/ui/MetricsCard";
import RegimeBadge from "@/components/ui/RegimeBadge";
import ErrorState from "@/components/ui/ErrorState";
import {
  CardSkeleton,
  ChartSkeleton,
  TableSkeleton,
} from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function TradingPage() {
  const portfolioQ = useQuery({
    queryKey: ["phase4", "portfolio"],
    queryFn: fetchPortfolio,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const metricsQ = useQuery({
    queryKey: ["phase4", "portfolio-metrics"],
    queryFn: fetchPortfolioMetrics,
    refetchInterval: REFETCH_INTERVAL,
    staleTime: STALE_TIME,
  });

  const equityQ = useQuery({
    queryKey: ["phase4", "portfolio-equity"],
    queryFn: () => fetchPortfolioEquity(500),
    staleTime: STALE_TIME,
  });

  const attrQ = useQuery({
    queryKey: ["phase4", "regime-attribution"],
    queryFn: fetchRegimeAttribution,
    staleTime: STALE_TIME,
  });

  const tradesQ = useQuery({
    queryKey: ["phase4", "trades"],
    queryFn: () => fetchTrades(100),
    staleTime: STALE_TIME,
  });

  const portfolio = portfolioQ.data;
  const metrics = metricsQ.data;
  const equity = equityQ.data;
  const attr = attrQ.data;
  const trades = tradesQ.data;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">Paper Trading</h1>
        <p className="text-sm text-text-secondary">
          Live paper portfolio, performance metrics, regime attribution, and
          trade history
        </p>
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        {portfolio ? (
          <>
            <MetricsCard
              label="Total Equity"
              value={currency(portfolio.total_equity)}
              color={portfolio.total_return_pct >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Total Return"
              value={pctRaw(portfolio.total_return_pct)}
              color={portfolio.total_return_pct >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Positions"
              value={portfolio.num_positions}
              sub={`Cash: ${currency(portfolio.cash)}`}
            />
            <MetricsCard
              label="Rebalances"
              value={portfolio.rebalance_count}
              sub={
                portfolio.last_rebalance
                  ? `Last: ${portfolio.last_rebalance}`
                  : "None"
              }
            />
            <MetricsCard
              label="Sharpe Ratio"
              value={metrics ? num(metrics.sharpe_ratio) : "—"}
              color={
                metrics && metrics.sharpe_ratio >= 1 ? "#22c55e" : "#f59e0b"
              }
            />
          </>
        ) : portfolioQ.isError ? (
          <div className="col-span-5">
            <ErrorState onRetry={() => portfolioQ.refetch()} />
          </div>
        ) : (
          Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)
        )}
      </div>

      {/* Performance metrics row */}
      {metrics && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <MetricsCard
            label="Max Drawdown"
            value={pctRaw(metrics.max_drawdown)}
            color="#ef4444"
          />
          <MetricsCard
            label="Win Rate"
            value={pct(metrics.win_rate)}
            color={metrics.win_rate > 0.5 ? "#22c55e" : "#ef4444"}
          />
          <MetricsCard
            label="Volatility (Ann.)"
            value={pctRaw(metrics.volatility)}
          />
          <MetricsCard
            label="Calmar Ratio"
            value={num(metrics.calmar_ratio)}
            color={metrics.calmar_ratio >= 1 ? "#22c55e" : "#f59e0b"}
          />
        </div>
      )}

      {/* Equity Curve */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Equity Curve
        </p>
        {equity && equity.curve.length > 0 ? (
          <PlotlyChart
            height={320}
            data={[
              {
                type: "scatter" as const,
                mode: "lines" as const,
                x: equity.curve.map((p) => p.date),
                y: equity.curve.map((p) => p.equity),
                line: { color: "#6366f1", width: 2 },
                fill: "tozeroy" as const,
                fillcolor: "rgba(99,102,241,0.08)",
                hovertemplate: "%{x}<br>Equity: $%{y:,.0f}<extra></extra>",
              },
            ]}
            layout={{
              yaxis: {
                title: { text: "Equity ($)", font: { size: 10 } },
                tickprefix: "$",
              },
            }}
          />
        ) : equityQ.isError ? (
          <ErrorState onRetry={() => equityQ.refetch()} />
        ) : (
          <ChartSkeleton height="h-72" />
        )}
      </div>

      {/* Positions + Regime Attribution */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Current positions */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Current Positions ({portfolio?.num_positions ?? 0})
          </p>
          {portfolio && portfolio.positions.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-text-muted">
                    <th className="pb-2 pr-4">Asset</th>
                    <th className="pb-2 pr-4">Shares</th>
                    <th className="pb-2 pr-4">Value</th>
                    <th className="pb-2 pr-4">Weight</th>
                    <th className="pb-2">P&amp;L</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio.positions.map((pos, i) => (
                    <tr
                      key={pos.asset}
                      className={cn(
                        "border-b border-border/50",
                        i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                      )}
                    >
                      <td className="py-2 pr-4 font-medium text-foreground">
                        {pos.asset}
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {num(pos.shares)}
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {currency(pos.current_value)}
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {pct(pos.weight)}
                      </td>
                      <td
                        className="py-2 font-mono"
                        style={{
                          color:
                            pos.unrealized_pnl >= 0 ? "#22c55e" : "#ef4444",
                        }}
                      >
                        {pos.unrealized_pnl >= 0 ? "+" : ""}
                        {currency(pos.unrealized_pnl)} (
                        {pctRaw(pos.unrealized_pnl_pct)})
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : portfolio ? (
            <p className="text-sm text-text-secondary">No open positions</p>
          ) : (
            <TableSkeleton rows={4} />
          )}
        </div>

        {/* Regime attribution */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            P&amp;L by Regime
          </p>
          {attr && Array.isArray(attr.attribution) && attr.attribution.length > 0 ? (
            <PlotlyChart
              height={280}
              data={[
                {
                  type: "bar" as const,
                  x: attr.attribution.map((a) => a.regime_name),
                  y: attr.attribution.map((a) => a.pnl),
                  marker: {
                    color: attr.attribution.map(
                      (a) => REGIME_COLORS[a.regime] ?? "#6b7280",
                    ),
                  },
                  hovertemplate:
                    "%{x}<br>P&L: $%{y:,.0f}<br>Days: " +
                    attr.attribution.map((a) => a.days).join(",") +
                    "<extra></extra>",
                },
              ]}
              layout={{
                yaxis: {
                  title: { text: "P&L ($)", font: { size: 10 } },
                  tickprefix: "$",
                  zeroline: true,
                  zerolinecolor: "#6b7280",
                },
              }}
            />
          ) : attrQ.isError ? (
            <ErrorState onRetry={() => attrQ.refetch()} />
          ) : (
            <ChartSkeleton height="h-60" />
          )}
        </div>
      </div>

      {/* Trade Log */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Trade History ({trades?.total ?? 0})
        </p>
        {trades && trades.orders.length > 0 ? (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-border text-left text-xs text-text-muted">
                  <th className="pb-2 pr-4">Time</th>
                  <th className="pb-2 pr-4">Asset</th>
                  <th className="pb-2 pr-4">Side</th>
                  <th className="pb-2 pr-4">Shares</th>
                  <th className="pb-2 pr-4">Price</th>
                  <th className="pb-2 pr-4">Value</th>
                  <th className="pb-2 pr-4">Regime</th>
                  <th className="pb-2">Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.orders.map((order, i) => (
                  <tr
                    key={order.order_id}
                    className={cn(
                      "border-b border-border/50",
                      i % 2 === 0 ? "bg-surface" : "bg-surface-elevated",
                    )}
                  >
                    <td className="py-1.5 pr-4 text-xs text-text-muted">
                      {order.timestamp}
                    </td>
                    <td className="py-1.5 pr-4 font-medium text-foreground">
                      {order.asset}
                    </td>
                    <td className="py-1.5 pr-4">
                      <span
                        className="rounded px-1.5 py-0.5 text-xs font-medium"
                        style={{
                          backgroundColor:
                            order.side === "buy"
                              ? "rgba(34,197,94,0.1)"
                              : "rgba(239,68,68,0.1)",
                          color: order.side === "buy" ? "#22c55e" : "#ef4444",
                        }}
                      >
                        {order.side}
                      </span>
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-text-secondary">
                      {num(order.shares)}
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-text-secondary">
                      ${order.price.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-text-secondary">
                      {currency(order.value)}
                    </td>
                    <td className="py-1.5 pr-4">
                      <RegimeBadge regime={order.regime} size="sm" />
                    </td>
                    <td className="py-1.5 text-xs text-text-secondary">
                      {order.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : trades ? (
          <p className="text-sm text-text-secondary">No trades executed yet</p>
        ) : (
          <TableSkeleton rows={5} />
        )}
      </div>
    </div>
  );
}
