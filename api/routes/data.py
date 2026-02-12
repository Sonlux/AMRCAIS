"""
Data API routes.

Endpoints:
    GET /api/data/assets                      — Available asset list
    GET /api/data/prices/{asset}              — OHLCV price data
    GET /api/data/correlations                — Correlation matrix
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from api.dependencies import get_system
from api.schemas import (
    CorrelationMatrixResponse,
    PricePoint,
    PriceResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Assets the system is designed to track
ALL_ASSETS = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"]


@router.get("/assets", response_model=list[str])
async def list_assets():
    """Return the list of tracked assets."""
    return ALL_ASSETS


@router.get("/prices/{asset}", response_model=PriceResponse)
async def get_prices(
    asset: str = Path(..., description="Asset symbol (e.g. SPX)"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
):
    """Return OHLCV price data for a single asset."""
    asset_upper = asset.upper()
    if asset_upper not in ALL_ASSETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown asset '{asset}'. Valid: {ALL_ASSETS}",
        )

    system = get_system()
    data = getattr(system, "market_data", None)
    if data is None or len(data) == 0:
        return PriceResponse(asset=asset_upper, prices=[], total_points=0)

    # Apply date filters
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    start_dt = (
        datetime.strptime(start, "%Y-%m-%d")
        if start
        else end_dt - timedelta(days=365)
    )

    # Handle both multi-asset DataFrame (with MultiIndex columns) and flat DataFrame
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # yfinance-style MultiIndex: (field, ticker)
            if asset_upper in data.columns.get_level_values(1):
                asset_df = data.xs(asset_upper, level=1, axis=1)
            else:
                asset_df = pd.DataFrame()
        else:
            # Flat columns — look for Close_{asset} or just Close
            close_col = f"Close_{asset_upper}" if f"Close_{asset_upper}" in data.columns else "Close"
            if close_col not in data.columns:
                close_col = None

            if close_col:
                asset_df = pd.DataFrame({"Close": data[close_col]})
            else:
                asset_df = pd.DataFrame()
    except Exception:
        asset_df = pd.DataFrame()

    if asset_df.empty:
        return PriceResponse(asset=asset_upper, prices=[], total_points=0)

    # Filter by date
    mask = (asset_df.index >= pd.Timestamp(start_dt)) & (
        asset_df.index <= pd.Timestamp(end_dt)
    )
    asset_df = asset_df.loc[mask]

    points = []
    for idx, row in asset_df.iterrows():
        dt_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        points.append(
            PricePoint(
                date=dt_str,
                open=float(row["Open"]) if "Open" in row and pd.notna(row.get("Open")) else None,
                high=float(row["High"]) if "High" in row and pd.notna(row.get("High")) else None,
                low=float(row["Low"]) if "Low" in row and pd.notna(row.get("Low")) else None,
                close=float(row["Close"]) if "Close" in row and pd.notna(row["Close"]) else 0.0,
                volume=float(row["Volume"]) if "Volume" in row and pd.notna(row.get("Volume")) else None,
            )
        )

    return PriceResponse(
        asset=asset_upper,
        prices=points,
        total_points=len(points),
    )


@router.get("/correlations", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix(
    window: int = Query(60, ge=10, le=252, description="Rolling window in days"),
):
    """Return cross-asset correlation matrix.

    Computes rolling return correlations for the tracked assets over the
    specified window.
    """
    system = get_system()
    data = getattr(system, "market_data", None)
    if data is None or len(data) == 0:
        return CorrelationMatrixResponse(
            assets=ALL_ASSETS,
            matrix=[[0.0] * len(ALL_ASSETS) for _ in ALL_ASSETS],
            window=window,
        )

    # Try to extract per-asset close columns
    closes: dict[str, pd.Series] = {}

    for asset in ALL_ASSETS:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if asset in data.columns.get_level_values(1):
                    series = data.xs(asset, level=1, axis=1).get("Close")
                    if series is not None:
                        closes[asset] = series
            else:
                col_name = f"Close_{asset}" if f"Close_{asset}" in data.columns else None
                if col_name:
                    closes[asset] = data[col_name]
        except Exception:
            continue

    if len(closes) < 2:
        return CorrelationMatrixResponse(
            assets=list(closes.keys()) or ALL_ASSETS,
            matrix=[[0.0] * len(ALL_ASSETS) for _ in ALL_ASSETS],
            window=window,
        )

    close_df = pd.DataFrame(closes)
    returns_df = close_df.pct_change().dropna()
    tail = returns_df.tail(window)
    corr = tail.corr()

    # Ensure consistent ordering
    ordered_assets = [a for a in ALL_ASSETS if a in corr.columns]
    corr = corr.reindex(index=ordered_assets, columns=ordered_assets).fillna(0.0)

    matrix = corr.values.tolist()
    # Round for cleaner JSON
    matrix = [[round(v, 4) for v in row] for row in matrix]

    return CorrelationMatrixResponse(
        assets=ordered_assets,
        matrix=matrix,
        window=window,
    )
