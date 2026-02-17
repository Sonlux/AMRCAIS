"""
Data Fetchers for AMRCAIS.

This module provides fetcher classes for multiple data sources with automatic
fallback logic. Priority order: FRED API → yfinance → Alpha Vantage → cached data.

Classes:
    BaseFetcher: Abstract base class for all data fetchers
    FREDFetcher: Fetches data from Federal Reserve Economic Data API
    YFinanceFetcher: Fetches data from Yahoo Finance
    AlphaVantageFetcher: Fetches data from Alpha Vantage API

Example:
    >>> fetcher = YFinanceFetcher()
    >>> data = fetcher.fetch("SPX", start_date="2020-01-01", end_date="2024-12-31")
    >>> print(data.head())
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import os
import time

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseFetcher(ABC):
    """Abstract base class for all data fetchers.
    
    Defines the interface that all data fetchers must implement.
    Provides common functionality for rate limiting and error handling.
    
    Attributes:
        rate_limit_per_minute: Maximum API calls allowed per minute
        last_request_time: Timestamp of the last API request
        request_count: Number of requests made in current minute window
    """
    
    def __init__(self, rate_limit_per_minute: int = 60):
        """Initialize the fetcher with rate limiting.
        
        Args:
            rate_limit_per_minute: Maximum number of API calls per minute
        """
        self.rate_limit_per_minute = rate_limit_per_minute
        self.last_request_time: Optional[datetime] = None
        self.request_count = 0
        self._minute_start: Optional[datetime] = None
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting.
        
        Sleeps if necessary to avoid exceeding rate limits.
        """
        now = datetime.now()
        
        if self._minute_start is None:
            self._minute_start = now
            self.request_count = 0
        
        # Reset counter if a minute has passed
        if (now - self._minute_start).seconds >= 60:
            self._minute_start = now
            self.request_count = 0
        
        # If we've hit the limit, wait until the minute is up
        if self.request_count >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self._minute_start).seconds
            if sleep_time > 0:
                logger.warning(
                    f"Rate limit reached. Sleeping for {sleep_time} seconds."
                )
                time.sleep(sleep_time)
                self._minute_start = datetime.now()
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = now
    
    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Fetch data for a given symbol and date range.
        
        Args:
            symbol: The asset symbol to fetch (e.g., "SPX", "TLT")
            start_date: Start date for the data range
            end_date: End date for the data range
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            
        Raises:
            ValueError: If symbol is invalid or dates are malformed
            ConnectionError: If API request fails
        """
        pass
    
    @abstractmethod
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of asset symbols to fetch
            start_date: Start date for the data range
            end_date: End date for the data range
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        pass
    
    @staticmethod
    def _parse_date(date: Union[str, datetime]) -> datetime:
        """Convert string date to datetime object.
        
        Args:
            date: Date as string (YYYY-MM-DD) or datetime object
            
        Returns:
            datetime object
            
        Raises:
            ValueError: If date string is malformed
        """
        if isinstance(date, datetime):
            return date
        try:
            return datetime.strptime(date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid date format: {date}. Expected YYYY-MM-DD."
            ) from e


class FREDFetcher(BaseFetcher):
    """Fetcher for Federal Reserve Economic Data (FRED) API.
    
    Provides access to macroeconomic data including:
    - Treasury yields (DGS1, DGS2, DGS5, DGS10, DGS30)
    - S&P 500 (SP500)
    - VIX (VIXCLS)
    - Oil prices (DCOILWTICO)
    - Dollar index (DTWEXBGS)
    
    Requires FRED_API_KEY environment variable to be set.
    
    Example:
        >>> fetcher = FREDFetcher()
        >>> vix_data = fetcher.fetch("VIXCLS", "2020-01-01", "2024-12-31")
    """
    
    # Mapping from AMRCAIS symbols to FRED series IDs
    SYMBOL_MAPPING = {
        "SPX": "SP500",
        "VIX": "VIXCLS",
        "WTI": "DCOILWTICO",
        "DXY": "DTWEXBGS",
        # Yield curve
        "DGS1MO": "DGS1MO",
        "DGS3MO": "DGS3MO",
        "DGS6MO": "DGS6MO",
        "DGS1": "DGS1",
        "DGS2": "DGS2",
        "DGS3": "DGS3",
        "DGS5": "DGS5",
        "DGS7": "DGS7",
        "DGS10": "DGS10",
        "DGS20": "DGS20",
        "DGS30": "DGS30",
        # Macro data
        "NFP": "PAYEMS",
        "UNRATE": "UNRATE",
        "CPI": "CPIAUCSL",
        "CORE_PCE": "PCEPILFE",
        "GDP": "GDPC1",
        "FED_FUNDS": "FEDFUNDS",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize FRED fetcher with API key.
        
        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
            
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        super().__init__(rate_limit_per_minute=120)
        
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not found. FRED fetcher will be unavailable."
            )
        
        self._fred = None
    
    def _get_fred_client(self):
        """Lazily initialize FRED client."""
        if self._fred is None and self.api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except ImportError:
                logger.error("fredapi package not installed. Run: pip install fredapi")
                raise
        return self._fred
    
    def fetch(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Fetch data from FRED API.
        
        Args:
            symbol: AMRCAIS symbol or FRED series ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with Date index and value column
            
        Raises:
            ValueError: If symbol not found or API key missing
            ConnectionError: If API request fails
        """
        if not self.api_key:
            raise ValueError("FRED API key not configured")
        
        self._check_rate_limit()
        
        # Map symbol to FRED series ID
        series_id = self.SYMBOL_MAPPING.get(symbol, symbol)
        
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        
        try:
            fred = self._get_fred_client()
            data = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            
            df = pd.DataFrame(data, columns=["Close"])
            df.index.name = "Date"
            df = df.dropna()
            
            logger.info(
                f"FRED: Fetched {len(df)} observations for {symbol} ({series_id})"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"FRED fetch failed for {symbol}: {e}")
            raise ConnectionError(f"Failed to fetch {symbol} from FRED: {e}") from e
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple series from FRED.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, start_date, end_date)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results


class YFinanceFetcher(BaseFetcher):
    """Fetcher for Yahoo Finance data via yfinance library.
    
    Primary data source for:
    - Equity indices (^GSPC for S&P 500)
    - Bond ETFs (TLT, IEF)
    - Commodity ETFs (GLD, USO)
    - Volatility (^VIX)
    - Currency ETFs (UUP for dollar)
    
    Does not require API key.
    
    Example:
        >>> fetcher = YFinanceFetcher()
        >>> spy_data = fetcher.fetch("SPX", "2020-01-01", "2024-12-31")
    """
    
    # Mapping from AMRCAIS symbols to Yahoo Finance tickers
    SYMBOL_MAPPING = {
        "SPX": "^GSPC",
        "TLT": "TLT",
        "IEF": "IEF",
        "GLD": "GLD",
        "DXY": "DX-Y.NYB",
        "WTI": "CL=F",
        "VIX": "^VIX",
    }
    
    def __init__(self):
        """Initialize yfinance fetcher."""
        super().__init__(rate_limit_per_minute=2000)
        self._yf = None
    
    def _get_yf(self):
        """Lazily import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.error("yfinance package not installed. Run: pip install yfinance")
                raise
        return self._yf
    
    def fetch(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: AMRCAIS symbol or Yahoo Finance ticker
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV columns and Date index
            
        Raises:
            ValueError: If symbol not found
            ConnectionError: If API request fails
        """
        self._check_rate_limit()
        
        # Map symbol to Yahoo Finance ticker
        ticker = self.SYMBOL_MAPPING.get(symbol, symbol)
        
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        
        try:
            yf = self._get_yf()
            
            # Add one day to end date since yfinance excludes end date
            end_adjusted = end + timedelta(days=1)
            
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end_adjusted.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,  # Adjust for splits and dividends
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol} ({ticker})")
            
            # Handle MultiIndex columns from yfinance (v1.x returns MultiIndex
            # even for single-ticker downloads)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Standardize column names to Title case
            data.columns = [col.title() if isinstance(col, str) else str(col)
                            for col in data.columns]
            
            data.index.name = "Date"
            
            logger.info(
                f"yfinance: Fetched {len(data)} observations for {symbol} ({ticker})"
            )
            
            return data
            
        except Exception as e:
            logger.error(f"yfinance fetch failed for {symbol}: {e}")
            raise ConnectionError(
                f"Failed to fetch {symbol} from Yahoo Finance: {e}"
            ) from e
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple symbols from Yahoo Finance.
        
        Uses batch download for efficiency.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        self._check_rate_limit()
        
        # Map all symbols to Yahoo tickers
        tickers = [self.SYMBOL_MAPPING.get(s, s) for s in symbols]
        ticker_to_symbol = {
            self.SYMBOL_MAPPING.get(s, s): s for s in symbols
        }
        
        start = self._parse_date(start_date)
        end = self._parse_date(end_date) + timedelta(days=1)
        
        try:
            yf = self._get_yf()
            
            data = yf.download(
                tickers,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            
            results = {}
            for ticker in tickers:
                symbol = ticker_to_symbol[ticker]
                try:
                    if len(tickers) == 1:
                        # Single ticker returns non-MultiIndex
                        df = data.copy()
                    else:
                        df = data[ticker].copy()
                    
                    # Flatten MultiIndex columns from yfinance v1.x
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    df.columns = [col.title() if isinstance(col, str) else str(col)
                                  for col in df.columns]
                    df.index.name = "Date"
                    df = df.dropna(how="all")
                    results[symbol] = df
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {symbol}: {e}")
                    results[symbol] = pd.DataFrame()
            
            return results
            
        except Exception as e:
            logger.error(f"yfinance batch fetch failed: {e}")
            # Fall back to individual fetches
            return {
                symbol: self.fetch(symbol, start_date, end_date)
                for symbol in symbols
            }


class AlphaVantageFetcher(BaseFetcher):
    """Fetcher for Alpha Vantage API.
    
    Tertiary data source used when FRED and yfinance fail.
    Free tier has severe rate limits (5 calls/minute, 500 calls/day).
    
    Requires ALPHAVANTAGE_API_KEY environment variable.
    
    Example:
        >>> fetcher = AlphaVantageFetcher()
        >>> data = fetcher.fetch("SPY", "2020-01-01", "2024-12-31")
    """
    
    # Mapping from AMRCAIS symbols to Alpha Vantage symbols
    SYMBOL_MAPPING = {
        "SPX": "SPY",  # Use SPY as proxy
        "TLT": "TLT",
        "IEF": "IEF",
        "GLD": "GLD",
        "DXY": "UUP",  # Use UUP as proxy
        "WTI": "USO",  # Use USO as proxy
        "VIX": "VIXY",  # Use VIXY as proxy
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage fetcher.
        
        Args:
            api_key: Alpha Vantage API key. If None, reads from env var.
        """
        super().__init__(rate_limit_per_minute=5)
        
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            logger.warning(
                "ALPHAVANTAGE_API_KEY not found. Alpha Vantage fetcher unavailable."
            )
        
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage.
        
        Args:
            symbol: AMRCAIS symbol or Alpha Vantage symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If API key missing or symbol invalid
            ConnectionError: If API request fails
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not configured")
        
        self._check_rate_limit()
        
        import requests
        
        # Map symbol
        av_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": av_symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            
            if "Time Series (Daily)" not in data:
                raise ValueError(f"Unexpected response format for {symbol}")
            
            ts = data["Time Series (Daily)"]
            
            df = pd.DataFrame.from_dict(ts, orient="index")
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"
            
            # Rename columns
            df.columns = [
                col.split(". ")[1].title() if ". " in col else col
                for col in df.columns
            ]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Filter date range
            df = df[(df.index >= start) & (df.index <= end)]
            df = df.sort_index()
            
            logger.info(
                f"Alpha Vantage: Fetched {len(df)} observations for {symbol}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}")
            raise ConnectionError(
                f"Failed to fetch {symbol} from Alpha Vantage: {e}"
            ) from e
    
    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple symbols from Alpha Vantage.
        
        Note: Due to severe rate limits, this will be slow.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, start_date, end_date)
            except Exception as e:
                logger.warning(f"Alpha Vantage failed for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results
