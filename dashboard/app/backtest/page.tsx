"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { runBacktest, fetchBacktestResults, fetchAssets } from "@/lib/api";
import {
  REGIME_COLORS,
  REGIME_NAMES,
  STALE_TIME,
  TRACKED_ASSETS,
} from "@/lib/constants";
import { pct, pctRaw, num, currency, cn } from "@/lib/utils";
import type { BacktestRequest, BacktestResultResponse } from "@/lib/types";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { ChartSkeleton, CardSkeleton } from "@/components/ui/Skeleton";
import PlotlyChart from "@/components/charts/PlotlyChart";

export default function BacktestPage() {
  const [form, setForm] = useState<BacktestRequest>({
    start_date: "2015-01-01",
    end_date: "2024-01-01",
    strategy: "regime_following",
    initial_capital: 100000,
    assets: [...TRACKED_ASSETS],
  });

  const [result, setResult] = useState<BacktestResultResponse | null>(null);

  const backtestMut = useMutation({
    mutationFn: runBacktest,
    onSuccess: (data) => setResult(data),
  });

  const resultsQ = useQuery({
    queryKey: ["backtest", "results"],
    queryFn: fetchBacktestResults,
    staleTime: STALE_TIME,
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-foreground">Backtest Lab</h1>
        <p className="text-sm text-text-secondary">
          Walk-forward regime-following strategy backtesting
        </p>
      </div>

      {/* Config form */}
      <div className="rounded-lg border border-border bg-surface p-4">
        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
          Backtest Configuration
        </p>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Start Date
            </label>
            <input
              type="date"
              value={form.start_date}
              onChange={(e) =>
                setForm((f) => ({ ...f, start_date: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              End Date
            </label>
            <input
              type="date"
              value={form.end_date}
              onChange={(e) =>
                setForm((f) => ({ ...f, end_date: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Strategy
            </label>
            <select
              value={form.strategy}
              onChange={(e) =>
                setForm((f) => ({ ...f, strategy: e.target.value }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            >
              <option value="regime_following">Regime Following</option>
              <option value="momentum">Momentum</option>
              <option value="mean_reversion">Mean Reversion</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-text-secondary">
              Initial Capital
            </label>
            <input
              type="number"
              value={form.initial_capital}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  initial_capital: Number(e.target.value),
                }))
              }
              className="w-full rounded border border-border bg-surface-elevated px-3 py-1.5 text-sm text-foreground outline-none focus:border-accent"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={() => backtestMut.mutate(form)}
              disabled={backtestMut.isPending}
              className="w-full rounded bg-accent px-4 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-accent/90 disabled:opacity-50"
            >
              {backtestMut.isPending ? "Running…" : "Run Backtest"}
            </button>
          </div>
        </div>
        {backtestMut.isError && (
          <p className="mt-2 text-xs text-regime-2">
            Error: {backtestMut.error.message}
          </p>
        )}
      </div>

      {/* Results */}
      {result && (
        <>
          {/* KPI cards */}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
            <MetricsCard
              label="Total Return"
              value={pctRaw(result.total_return)}
              color={result.total_return >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Sharpe Ratio"
              value={num(result.sharpe_ratio)}
              color={result.sharpe_ratio >= 1 ? "#22c55e" : "#f59e0b"}
            />
            <MetricsCard
              label="Max Drawdown"
              value={pctRaw(result.max_drawdown)}
              color="#ef4444"
            />
            <MetricsCard
              label="Benchmark Return"
              value={pctRaw(result.benchmark_return)}
              color={result.benchmark_return >= 0 ? "#22c55e" : "#ef4444"}
            />
            <MetricsCard
              label="Alpha"
              value={pctRaw(result.total_return - result.benchmark_return)}
              color={
                result.total_return - result.benchmark_return >= 0
                  ? "#22c55e"
                  : "#ef4444"
              }
            />
          </div>

          {/* Equity curve */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Equity Curve
            </p>
            <PlotlyChart
              height={350}
              data={[
                {
                  x: result.equity_curve.map((e) => e.date),
                  y: result.equity_curve.map((e) => e.value),
                  type: "scatter",
                  mode: "lines",
                  line: { color: "#3b82f6", width: 1.5 },
                  name: "Strategy",
                  fill: "tozeroy",
                  fillcolor: "rgba(59,130,246,0.06)",
                },
              ]}
              layout={{
                yaxis: {
                  title: { text: "Portfolio Value", font: { size: 10 } },
                  tickprefix: "$",
                },
                xaxis: { type: "date" },
                showlegend: false,
              }}
            />
          </div>

          {/* Returns by regime */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Returns by Regime
            </p>
            <PlotlyChart
              height={280}
              data={[
                {
                  type: "bar",
                  x: result.regime_returns.map((r) => r.regime_name),
                  y: result.regime_returns.map((r) => r.strategy_return * 100),
                  name: "Strategy",
                  marker: {
                    color: result.regime_returns.map(
                      (r) => REGIME_COLORS[r.regime] ?? "#6b7280",
                    ),
                  },
                  hovertemplate:
                    "%{x}<br>Return: %{y:.1f}%<extra>Strategy</extra>",
                },
                {
                  type: "bar",
                  x: result.regime_returns.map((r) => r.regime_name),
                  y: result.regime_returns.map((r) => r.benchmark_return * 100),
                  name: "Benchmark",
                  marker: { color: "#555568" },
                  hovertemplate:
                    "%{x}<br>Return: %{y:.1f}%<extra>Benchmark</extra>",
                },
              ]}
              layout={{
                barmode: "group",
                showlegend: true,
                legend: { orientation: "h", y: -0.2, font: { size: 10 } },
                yaxis: {
                  title: { text: "Return (%)", font: { size: 10 } },
                },
              }}
            />
          </div>

          {/* Regime return table */}
          <div className="rounded-lg border border-border bg-surface p-4">
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
              Regime Breakdown
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-text-muted">
                    <th className="pb-2 pr-4">Regime</th>
                    <th className="pb-2 pr-4">Days</th>
                    <th className="pb-2 pr-4">Strategy Return</th>
                    <th className="pb-2 pr-4">Benchmark Return</th>
                    <th className="pb-2">Hit Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {result.regime_returns.map((r) => (
                    <tr key={r.regime} className="border-b border-border/50">
                      <td className="py-2 pr-4">
                        <span
                          className="font-medium"
                          style={{
                            color: REGIME_COLORS[r.regime] ?? "#6b7280",
                          }}
                        >
                          {r.regime_name}
                        </span>
                      </td>
                      <td className="py-2 pr-4 font-mono text-text-secondary">
                        {r.days}
                      </td>
                      <td
                        className="py-2 pr-4 font-mono"
                        style={{
                          color: r.strategy_return >= 0 ? "#22c55e" : "#ef4444",
                        }}
                      >
                        {pctRaw(r.strategy_return)}
                      </td>
                      <td
                        className="py-2 pr-4 font-mono"
                        style={{
                          color:
                            r.benchmark_return >= 0 ? "#22c55e" : "#ef4444",
                        }}
                      >
                        {pctRaw(r.benchmark_return)}
                      </td>
                      <td className="py-2 font-mono text-text-secondary">
                        {pct(r.hit_rate)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Previous results list */}
      {resultsQ.data && resultsQ.data.length > 0 && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Previous Backtest Results
          </p>
          <div className="space-y-2">
            {resultsQ.data.map((r: BacktestResultResponse) => (
              <button
                key={r.id}
                onClick={() => setResult(r)}
                className="flex w-full items-center justify-between rounded-md bg-surface-elevated px-3 py-2 text-sm transition-colors hover:bg-border"
              >
                <span className="text-text-secondary">
                  {r.start_date} → {r.end_date}
                </span>
                <span className="font-mono text-xs">
                  <span
                    style={{
                      color: r.total_return >= 0 ? "#22c55e" : "#ef4444",
                    }}
                  >
                    {pctRaw(r.total_return)}
                  </span>
                  <span className="mx-2 text-text-muted">|</span>
                  <span className="text-text-secondary">
                    SR {num(r.sharpe_ratio)}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
