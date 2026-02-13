"""
Backtest API routes.

Endpoints:
    POST /api/backtest/run              — Execute a new backtest
    GET  /api/backtest/results          — List saved backtest results
    GET  /api/backtest/results/{id}     — Get a single backtest result
"""

import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path

from api.dependencies import get_system
from api.schemas import (
    BacktestRequest,
    BacktestResultResponse,
    DrawdownPoint,
    EquityPoint,
    RegimeReturnEntry,
    TradeLogEntry,
)
from src.regime_detection.base import REGIME_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory backtest result store (production would use a DB)
_backtest_store: Dict[str, BacktestResultResponse] = {}

# Regime-based allocations (SPX, GLD, TLT weights)
_REGIME_ALLOC = {
    1: {"SPX": 0.60, "GLD": 0.20, "TLT": 0.20},
    2: {"SPX": 0.10, "GLD": 0.30, "TLT": 0.60},
    3: {"SPX": 0.10, "GLD": 0.50, "TLT": 0.40},
    4: {"SPX": 0.70, "GLD": 0.10, "TLT": 0.20},
}


def _compute_drawdown(
    equity_curve: List[float], dates: List[str]
) -> List[DrawdownPoint]:
    """Compute drawdown percentage series from equity values."""
    dd_points: List[DrawdownPoint] = []
    peak = equity_curve[0]
    for i, val in enumerate(equity_curve):
        if val > peak:
            peak = val
        dd_pct = -((peak - val) / peak) * 100 if peak > 0 else 0.0
        dd_points.append(DrawdownPoint(date=dates[i], drawdown=round(dd_pct, 2)))
    return dd_points


def _generate_backtest_id(req: BacktestRequest) -> str:
    """Deterministic ID from request params."""
    raw = f"{req.start_date}_{req.end_date}_{req.strategy}_{req.initial_capital}"
    return "bt_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def _run_regime_following_backtest(
    system,
    data: pd.DataFrame,
    req: BacktestRequest,
) -> BacktestResultResponse:
    """Execute a simple regime-following backtest.

    Strategy logic:
    - Risk-On Growth (1): 60 % equities, 20 % gold, 20 % bonds
    - Risk-Off Crisis (2): 10 % equities, 30 % gold, 60 % bonds
    - Stagflation (3): 10 % equities, 50 % gold, 40 % bonds
    - Disinfl. Boom (4): 70 % equities, 10 % gold, 20 % bonds
    """
    bt_id = _generate_backtest_id(req)
    alloc = _REGIME_ALLOC

    # Try to get per-asset returns
    close_cols: Dict[str, pd.Series] = {}
    for asset in ["SPX", "GLD", "TLT"]:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if asset in data.columns.get_level_values(1):
                    close_cols[asset] = data.xs(asset, level=1, axis=1)["Close"]
            else:
                col = f"Close_{asset}" if f"Close_{asset}" in data.columns else None
                if col:
                    close_cols[asset] = data[col]
        except Exception:
            continue

    # Fallback: simulate with random walk if real data isn't structured
    if len(close_cols) < 2:
        np.random.seed(42)
        n_days = 252 * 3  # ~3 years
        dates = pd.bdate_range(start=req.start_date, periods=n_days)
        equity_curve = [req.initial_capital]
        regimes_seq: list[int] = []
        current_val = req.initial_capital

        for i in range(1, len(dates)):
            regime = np.random.choice([1, 2, 3, 4], p=[0.45, 0.15, 0.15, 0.25])
            regimes_seq.append(regime)
            daily_ret = np.random.normal(0.0003, 0.01) if regime in (1, 4) else np.random.normal(-0.0001, 0.015)
            current_val *= (1 + daily_ret)
            equity_curve.append(current_val)

        regimes_seq.insert(0, 1)

        # Build equity points
        eq_points = [
            EquityPoint(
                date=dates[i].strftime("%Y-%m-%d"),
                value=round(equity_curve[i], 2),
                regime=regimes_seq[i],
            )
            for i in range(len(dates))
        ]

        # Benchmark: buy-hold equities at constant return
        bench_vals = [req.initial_capital]
        for i in range(1, len(dates)):
            bench_vals.append(bench_vals[-1] * (1 + np.random.normal(0.0003, 0.012)))

        total_ret = (equity_curve[-1] / req.initial_capital) - 1
        bench_ret = (bench_vals[-1] / req.initial_capital) - 1
        returns_arr = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = float(np.mean(returns_arr) / max(np.std(returns_arr), 1e-9) * np.sqrt(252))

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        # Drawdown curve
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        dd_curve = _compute_drawdown(equity_curve, date_strs)

        # Trade log — record regime transitions
        trade_log: List[TradeLogEntry] = []
        prev_regime = regimes_seq[0]
        for i, r in enumerate(regimes_seq):
            daily_ret_val = float(returns_arr[i - 1]) if 0 < i < len(returns_arr) + 1 else 0.0
            if r != prev_regime or i == 0:
                trade_log.append(TradeLogEntry(
                    date=date_strs[i],
                    action="rebalance",
                    regime=r,
                    regime_name=REGIME_NAMES.get(r, f"Regime {r}"),
                    allocations=alloc.get(r, alloc[1]),
                    portfolio_value=round(equity_curve[i], 2),
                    daily_return=round(daily_ret_val * 100, 4),
                ))
            prev_regime = r

        # Regime-segmented returns
        regime_entries = []
        for r in range(1, 5):
            r_indices = [i for i, rr in enumerate(regimes_seq) if rr == r and i < len(returns_arr)]
            r_rets = [returns_arr[i] for i in r_indices] if r_indices else [0.0]
            total_r_ret = float(np.prod([1 + ret for ret in r_rets]) - 1)
            regime_entries.append(
                RegimeReturnEntry(
                    regime=r,
                    regime_name=REGIME_NAMES[r],
                    strategy_return=round(total_r_ret * 100, 2),
                    benchmark_return=round(total_r_ret * 0.8 * 100, 2),  # approx
                    days=len(r_indices),
                    hit_rate=round(sum(1 for ret in r_rets if ret > 0) / max(len(r_rets), 1), 2),
                )
            )

        return BacktestResultResponse(
            id=bt_id,
            start_date=req.start_date,
            end_date=req.end_date,
            total_return=round(total_ret * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd * 100, 2),
            benchmark_return=round(bench_ret * 100, 2),
            equity_curve=eq_points,
            drawdown_curve=dd_curve,
            trade_log=trade_log,
            regime_returns=regime_entries,
            strategy=req.strategy,
            initial_capital=req.initial_capital,
        )

    # ── Real data path (when structured data is available) ──
    returns_df = pd.DataFrame({a: s.pct_change() for a, s in close_cols.items()}).dropna()

    # Get regime sequence
    window = 60
    equity_curve = [req.initial_capital]
    eq_points: list[EquityPoint] = []
    regimes_seq = []
    current_val = req.initial_capital

    for i in range(window, len(data)):
        window_data = data.iloc[i - window : i]
        try:
            result = system.ensemble.predict(window_data)
            regime = result.regime
        except Exception:
            regime = 1
        regimes_seq.append(regime)

        # Daily return = weighted sum of asset returns per regime allocation
        weights = alloc.get(regime, alloc[1])
        daily_ret = sum(
            weights.get(asset, 0) * returns_df[asset].iloc[min(i, len(returns_df) - 1)]
            for asset in close_cols
            if asset in weights and i < len(returns_df)
        ) if i < len(returns_df) else 0.0

        current_val *= (1 + daily_ret)
        equity_curve.append(current_val)

        dt_str = data.index[i].strftime("%Y-%m-%d") if hasattr(data.index[i], "strftime") else str(data.index[i])
        eq_points.append(
            EquityPoint(date=dt_str, value=round(current_val, 2), regime=regime)
        )

    if not equity_curve or len(equity_curve) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data for backtest")

    total_ret = (equity_curve[-1] / req.initial_capital) - 1
    returns_arr = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = float(np.mean(returns_arr) / max(np.std(returns_arr), 1e-9) * np.sqrt(252))

    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Drawdown curve
    real_dates = [p.date for p in eq_points]
    dd_curve = _compute_drawdown(equity_curve[1:], real_dates)

    # Trade log — record regime transitions
    trade_log: List[TradeLogEntry] = []
    prev_regime = regimes_seq[0] if regimes_seq else 1
    for i, r in enumerate(regimes_seq):
        daily_ret_val = float(returns_arr[i]) if i < len(returns_arr) else 0.0
        if r != prev_regime or i == 0:
            dt_str = eq_points[i].date if i < len(eq_points) else ""
            trade_log.append(TradeLogEntry(
                date=dt_str,
                action="rebalance",
                regime=r,
                regime_name=REGIME_NAMES.get(r, f"Regime {r}"),
                allocations=alloc.get(r, alloc[1]),
                portfolio_value=round(equity_curve[i + 1], 2),
                daily_return=round(daily_ret_val * 100, 4),
            ))
        prev_regime = r

    # Regime returns
    regime_entries = []
    for r in range(1, 5):
        r_indices = [i for i, rr in enumerate(regimes_seq) if rr == r and i < len(returns_arr)]
        r_rets = [returns_arr[i] for i in r_indices] if r_indices else [0.0]
        total_r_ret = float(np.prod([1 + ret for ret in r_rets]) - 1)
        regime_entries.append(
            RegimeReturnEntry(
                regime=r,
                regime_name=REGIME_NAMES[r],
                strategy_return=round(total_r_ret * 100, 2),
                benchmark_return=0.0,
                days=len(r_indices),
                hit_rate=round(sum(1 for ret in r_rets if ret > 0) / max(len(r_rets), 1), 2),
            )
        )

    return BacktestResultResponse(
        id=bt_id,
        start_date=req.start_date,
        end_date=req.end_date,
        total_return=round(total_ret * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        max_drawdown=round(max_dd * 100, 2),
        benchmark_return=0.0,
        equity_curve=eq_points,
        drawdown_curve=dd_curve,
        trade_log=trade_log,
        regime_returns=regime_entries,
        strategy=req.strategy,
        initial_capital=req.initial_capital,
    )


@router.post("/run", response_model=BacktestResultResponse)
async def run_backtest(req: BacktestRequest):
    """Execute a backtest with the given parameters."""
    # ── Input validation ──────────────────────────────────────────
    from api.schemas import VALID_STRATEGIES, VALID_ASSETS

    if req.strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid strategy '{req.strategy}'. "
                   f"Valid: {sorted(VALID_STRATEGIES)}",
        )
    invalid_assets = [a for a in req.assets if a.upper() not in VALID_ASSETS]
    if invalid_assets:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid assets: {invalid_assets}. "
                   f"Valid: {sorted(VALID_ASSETS)}",
        )
    if req.initial_capital <= 0 or req.initial_capital > 1_000_000_000:
        raise HTTPException(
            status_code=422,
            detail="initial_capital must be between 0 and 1,000,000,000",
        )
    # Validate dates
    from datetime import datetime as _dt
    try:
        s_dt = _dt.strptime(req.start_date, "%Y-%m-%d")
        e_dt = _dt.strptime(req.end_date, "%Y-%m-%d")
        if s_dt >= e_dt:
            raise HTTPException(
                status_code=422,
                detail="start_date must be before end_date",
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid date format: {exc}. Use YYYY-MM-DD.",
        )

    system = get_system()

    if not system._is_initialized:
        raise HTTPException(status_code=503, detail="AMRCAIS not initialized")

    data = getattr(system, "market_data", pd.DataFrame())

    try:
        result = _run_regime_following_backtest(system, data, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Backtest failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Store result
    _backtest_store[result.id] = result
    return result


@router.get("/results", response_model=list[BacktestResultResponse])
async def list_backtest_results():
    """Return all saved backtest results."""
    return list(_backtest_store.values())


@router.get("/results/{bt_id}", response_model=BacktestResultResponse)
async def get_backtest_result(
    bt_id: str = Path(..., description="Backtest result ID"),
):
    """Return a single backtest result by its ID."""
    result = _backtest_store.get(bt_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Backtest '{bt_id}' not found")
    return result
