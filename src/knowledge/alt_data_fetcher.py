"""
Live Alternative Data Fetcher for AMRCAIS (Phase 5.3).

Extends the AltDataIntegrator with live data fetching from public APIs,
replacing the manual set_signal_value() workflow with automated retrieval.

Data Sources:
    - FRED API      → Fed Funds Rate, TIPS Breakevens, HY Spreads
    - yfinance      → MOVE Index proxy, Copper/Gold ratio, VIX
    - CBOE DataShop → SKEW Index

Configuration:
    FRED_API_KEY    — FRED API key (https://fred.stlouisfed.org/docs/api/)
    ALT_DATA_CACHE_MINUTES — Cache TTL (default: 30)

Classes:
    LiveAltDataFetcher: Fetches live alternative data to populate the integrator.

Example:
    >>> fetcher = LiveAltDataFetcher()
    >>> fetcher.refresh_all()
    >>> integrator.get_all_signals(regime=2)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LiveAltDataFetcher:
    """Fetches live alternative data signals from public APIs.

    Populates an AltDataIntegrator instance with current market data
    from FRED, yfinance, and other sources.

    Args:
        fred_api_key: FRED API key (or FRED_API_KEY env var).
        cache_minutes: How long to cache fetched values.

    Example:
        >>> from src.knowledge.alt_data import AltDataIntegrator
        >>> integrator = AltDataIntegrator()
        >>> fetcher = LiveAltDataFetcher()
        >>> fetcher.refresh_all(integrator)
    """

    # FRED series IDs for alternative data
    FRED_SERIES = {
        "fed_funds_futures": "DFF",           # Fed Funds Effective Rate
        "tips_breakeven": "T10YIE",           # 10Y Breakeven Inflation
        "hy_spreads": "BAMLH0A0HYM2",        # ICE BofA HY OAS
    }

    # yfinance tickers for market data proxies
    YFINANCE_TICKERS = {
        "move_index": "^MOVE",                # MOVE Index (if available)
        "copper": "HG=F",                     # Copper futures
        "gold": "GC=F",                       # Gold futures
        "skew_index": "^SKEW",                # SKEW Index
    }

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_minutes: int = 30,
    ) -> None:
        self._fred_key = fred_api_key or os.getenv("FRED_API_KEY", "")
        self._cache_minutes = cache_minutes
        self._cache: Dict[str, tuple] = {}  # signal_name → (value, fetch_time)
        self._errors: Dict[str, str] = {}

    def refresh_all(self, integrator: Any = None) -> Dict[str, float]:
        """Fetch all available alternative data signals.

        Args:
            integrator: AltDataIntegrator to populate (optional).

        Returns:
            Dict of signal_name → current value.
        """
        values: Dict[str, float] = {}

        # FRED data
        fred_values = self._fetch_fred_data()
        values.update(fred_values)

        # Market data (yfinance)
        market_values = self._fetch_market_data()
        values.update(market_values)

        # Compute derived signals
        if "copper" in market_values and "gold" in market_values:
            copper = market_values["copper"]
            gold = market_values["gold"]
            if gold > 0:
                values["copper_gold_ratio"] = copper / gold

        # Populate integrator if provided
        if integrator is not None:
            for name, val in values.items():
                try:
                    integrator.set_signal_value(name, val)
                except Exception:
                    pass  # skip signals not in integrator's enum

        logger.info(f"Alt data refresh complete: {len(values)} signals fetched")
        return values

    # ── FRED Fetching ─────────────────────────────────────────

    def _fetch_fred_data(self) -> Dict[str, float]:
        """Fetch alternative data from FRED API.

        Returns:
            Dict of signal_name → latest value.
        """
        if not self._fred_key:
            logger.debug("FRED API key not configured — skipping FRED data")
            return {}

        values: Dict[str, float] = {}

        for signal_name, series_id in self.FRED_SERIES.items():
            # Check cache
            cached = self._get_cached(signal_name)
            if cached is not None:
                values[signal_name] = cached
                continue

            try:
                value = self._fetch_fred_series(series_id)
                if value is not None:
                    values[signal_name] = value
                    self._set_cached(signal_name, value)
                    logger.debug(f"FRED {series_id}: {value}")
            except Exception as exc:
                self._errors[signal_name] = str(exc)
                logger.warning(f"FRED fetch failed for {series_id}: {exc}")

        return values

    def _fetch_fred_series(self, series_id: str) -> Optional[float]:
        """Fetch the latest value from a FRED time series.

        Args:
            series_id: FRED series identifier.

        Returns:
            Latest non-null value, or None.
        """
        import urllib.request
        import json

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&api_key={self._fred_key}"
            f"&file_type=json"
            f"&sort_order=desc"
            f"&limit=5"
        )

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        observations = data.get("observations", [])
        for obs in observations:
            try:
                value = float(obs["value"])
                return value
            except (ValueError, KeyError):
                continue

        return None

    # ── Market Data Fetching ──────────────────────────────────

    def _fetch_market_data(self) -> Dict[str, float]:
        """Fetch market data via yfinance.

        Returns:
            Dict of signal_name → latest price.
        """
        values: Dict[str, float] = {}

        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance not installed — skipping market data")
            return values

        for signal_name, ticker in self.YFINANCE_TICKERS.items():
            # Check cache
            cached = self._get_cached(signal_name)
            if cached is not None:
                values[signal_name] = cached
                continue

            try:
                data = yf.download(
                    ticker,
                    period="5d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if not data.empty:
                    value = float(data["Close"].iloc[-1])
                    values[signal_name] = value
                    self._set_cached(signal_name, value)
                    logger.debug(f"yfinance {ticker}: {value}")
            except Exception as exc:
                self._errors[signal_name] = str(exc)
                logger.debug(f"yfinance fetch failed for {ticker}: {exc}")

        return values

    # ── Cache ─────────────────────────────────────────────────

    def _get_cached(self, key: str) -> Optional[float]:
        """Get value from cache if still valid.

        Args:
            key: Signal name.

        Returns:
            Cached value, or None if expired/missing.
        """
        if key not in self._cache:
            return None

        value, fetch_time = self._cache[key]
        age_minutes = (time.time() - fetch_time) / 60

        if age_minutes > self._cache_minutes:
            del self._cache[key]
            return None

        return value

    def _set_cached(self, key: str, value: float) -> None:
        """Store value in cache.

        Args:
            key: Signal name.
            value: Fetched value.
        """
        self._cache[key] = (value, time.time())

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get fetcher status.

        Returns:
            Dict with configuration, cache state, and errors.
        """
        return {
            "fred_configured": bool(self._fred_key),
            "cache_minutes": self._cache_minutes,
            "cached_signals": list(self._cache.keys()),
            "errors": dict(self._errors),
            "fred_series": dict(self.FRED_SERIES),
            "yfinance_tickers": dict(self.YFINANCE_TICKERS),
        }

    def clear_cache(self) -> None:
        """Clear the value cache, forcing fresh fetches."""
        self._cache.clear()
        self._errors.clear()
        logger.info("Alt data cache cleared")
