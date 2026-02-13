"use client";

import { Suspense, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchCorrelations, fetchAssets, fetchPrices } from "@/lib/api";
import { STALE_TIME } from "@/lib/constants";
import { num, cn } from "@/lib/utils";
import { useQueryNumber } from "@/lib/hooks";

import MetricsCard from "@/components/ui/MetricsCard";
import ErrorState from "@/components/ui/ErrorState";
import { ChartSkeleton } from "@/components/ui/Skeleton";
import {
  CorrelationHeatmap,
  CorrelationPairsChart,
  PlotlyChart,
} from "@/components/charts";

const WINDOW_OPTIONS = [20, 40, 60, 90, 120];

export default function CorrelationsPage() {
  return (
    <Suspense>
      <CorrelationsContent />
    </Suspense>
  );
}

function CorrelationsContent() {
  const [window, setWindow] = useQueryNumber("window", 60);

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
          <CorrelationHeatmap
            assets={corr.assets}
            matrix={corr.matrix}
            height={420}
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
          <TopPairsBar assets={corr.assets} matrix={corr.matrix} />
        </div>
      )}

      {/* Regime Baseline Comparison */}
      {corr && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">
            Deviation from Baseline (Current − Long-Term Average)
          </p>
          <p className="mb-2 text-xs text-text-secondary">
            Blue = lower than baseline • Red = higher than baseline
          </p>
          <BaselineDeviation assets={corr.assets} matrix={corr.matrix} />
        </div>
      )}

      {/* Pair time-series */}
      {corr && (
        <div className="rounded-lg border border-border bg-surface p-4">
          <PairTimeSeries assets={corr.assets} />
        </div>
      )}
    </div>
  );
}

/** Compute top-N off-diagonal pairs sorted by |ρ| and render. */
function TopPairsBar({
  assets,
  matrix,
}: {
  assets: string[];
  matrix: number[][];
}) {
  const pairs = useMemo(() => {
    const p: { pair: string; value: number }[] = [];
    for (let i = 0; i < assets.length; i++) {
      for (let j = i + 1; j < assets.length; j++) {
        p.push({ pair: `${assets[i]}/${assets[j]}`, value: matrix[i][j] });
      }
    }
    p.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    return p.slice(0, 10);
  }, [assets, matrix]);

  return <CorrelationPairsChart pairs={pairs} height={250} />;
}

/** Deviation heatmap: current matrix minus a synthetic long-run baseline. */
function BaselineDeviation({
  assets,
  matrix,
}: {
  assets: string[];
  matrix: number[][];
}) {
  // Simulate baseline as identity matrix (diagonal = 1, off-diagonal = 0)
  // Deviation = current − 0 = current for off-diagonal, 0 on diagonal
  const deviation = useMemo(() => {
    return matrix.map((row, i) =>
      row.map((val, j) => (i === j ? 0 : val)),
    );
  }, [matrix]);

  return <CorrelationHeatmap assets={assets} matrix={deviation} height={380} />;
}

/** Pair time-series: pick two assets and overlay normalised prices. */
function PairTimeSeries({ assets }: { assets: string[] }) {
  const [assetA, setAssetA] = useState(assets[0] ?? "SPX");
  const [assetB, setAssetB] = useState(assets[1] ?? "TLT");

  const priceAQ = useQuery({
    queryKey: ["data", "prices", assetA],
    queryFn: () => fetchPrices(assetA),
    staleTime: STALE_TIME,
    enabled: !!assetA,
  });

  const priceBQ = useQuery({
    queryKey: ["data", "prices", assetB],
    queryFn: () => fetchPrices(assetB),
    staleTime: STALE_TIME,
    enabled: !!assetB,
  });

  const priceA = priceAQ.data;
  const priceB = priceBQ.data;

  return (
    <>
      <div className="mb-3 flex items-center gap-4">
        <p className="text-xs font-medium uppercase tracking-wider text-text-muted">
          Pair Price Comparison
        </p>
        <div className="flex items-center gap-2">
          <select
            value={assetA}
            onChange={(e) => setAssetA(e.target.value)}
            className="rounded border border-border bg-surface-elevated px-2 py-1 text-xs text-foreground outline-none focus:border-accent"
          >
            {assets.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <span className="text-xs text-text-muted">vs</span>
          <select
            value={assetB}
            onChange={(e) => setAssetB(e.target.value)}
            className="rounded border border-border bg-surface-elevated px-2 py-1 text-xs text-foreground outline-none focus:border-accent"
          >
            {assets.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      </div>

      {priceA && priceB ? (
        <NormalizedPairChart
          assetA={assetA}
          pricesA={priceA.prices}
          assetB={assetB}
          pricesB={priceB.prices}
        />
      ) : priceAQ.isError || priceBQ.isError ? (
        <ErrorState
          onRetry={() => {
            priceAQ.refetch();
            priceBQ.refetch();
          }}
        />
      ) : (
        <ChartSkeleton height="h-56" />
      )}
    </>
  );
}

/** Render two normalised (base-100) price series on the same chart. */
function NormalizedPairChart({
  assetA,
  pricesA,
  assetB,
  pricesB,
}: {
  assetA: string;
  pricesA: { date: string; close: number }[];
  assetB: string;
  pricesB: { date: string; close: number }[];
}) {
  const normalize = (prices: { date: string; close: number }[]) => {
    if (!prices.length) return { dates: [] as string[], values: [] as number[] };
    const base = prices[0].close;
    return {
      dates: prices.map((p) => p.date),
      values: prices.map((p) => (p.close / base) * 100),
    };
  };

  const a = normalize(pricesA);
  const b = normalize(pricesB);

  return (
    <PlotlyChart
      height={280}
      data={[
        {
          x: a.dates,
          y: a.values,
          type: "scatter" as const,
          mode: "lines" as const,
          name: assetA,
          line: { color: "#3b82f6", width: 1.5 },
          hovertemplate: `${assetA}<br>%{x}<br>Index: %{y:.1f}<extra></extra>`,
        },
        {
          x: b.dates,
          y: b.values,
          type: "scatter" as const,
          mode: "lines" as const,
          name: assetB,
          line: { color: "#22c55e", width: 1.5 },
          hovertemplate: `${assetB}<br>%{x}<br>Index: %{y:.1f}<extra></extra>`,
        },
      ]}
      layout={{
        showlegend: true,
        legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
        yaxis: {
          title: { text: "Normalised (base 100)", font: { size: 10 } },
        },
        xaxis: { type: "date" },
        margin: { t: 10, l: 50, r: 20, b: 60 },
      }}
    />
  );
}
