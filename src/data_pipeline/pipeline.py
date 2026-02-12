"""
Data Pipeline for AMRCAIS.

This module orchestrates data fetching, validation, and storage with
automatic fallback logic across multiple data sources.

Priority order: FRED API → yfinance → Alpha Vantage → cached data (≤7 days old)

Classes:
    DataPipeline: Main orchestrator for all data operations

Example:
    >>> pipeline = DataPipeline()
    >>> data = pipeline.fetch_market_data(
    ...     assets=["SPX", "TLT", "GLD", "VIX"],
    ...     start_date="2010-01-01",
    ...     end_date="2024-12-31"
    ... )
    >>> print(data["SPX"].head())
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import os

import pandas as pd
import numpy as np
import yaml

from src.data_pipeline.fetchers import (
    FREDFetcher,
    YFinanceFetcher,
    AlphaVantageFetcher,
)
from src.data_pipeline.validators import DataValidator, validate_multi_asset
from src.data_pipeline.storage import DatabaseStorage

logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates data fetching, validation, and storage for AMRCAIS.
    
    Provides a unified interface for:
    - Fetching market data from multiple sources with fallback
    - Validating data quality
    - Storing and retrieving from local database
    - Managing data freshness and caching
    
    Attributes:
        storage: Database storage instance
        validator: Data validator instance
        config: Configuration dictionary from YAML
    
    Example:
        >>> pipeline = DataPipeline(config_path="config/data_sources.yaml")
        >>> 
        >>> # Fetch required assets for regime detection
        >>> data = pipeline.fetch_market_data(
        ...     assets=["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"],
        ...     start_date="2010-01-01"
        ... )
        >>> 
        >>> # Get returns matrix for HMM
        >>> returns = pipeline.calculate_returns(data)
    """
    
    # Required assets for regime detection
    REQUIRED_ASSETS = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"]
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        db_path: Optional[str] = None,
        cache_max_age_days: int = 7,
    ):
        """Initialize the data pipeline.
        
        Args:
            config_path: Path to data_sources.yaml configuration
            db_path: Path to SQLite database (overrides config)
            cache_max_age_days: Maximum age for cached data before refresh
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize database storage
        if db_path is None:
            db_path = self.config.get("pipeline", {}).get(
                "database", {}
            ).get("path", "data/amrcais.db")
        
        self.storage = DatabaseStorage(db_path)
        
        # Initialize validator
        self.validator = DataValidator(
            max_missing_pct=self.config.get("pipeline", {}).get(
                "validation", {}
            ).get("max_missing_pct", 5.0)
        )
        
        # Initialize fetchers (lazy loading)
        self._fred_fetcher: Optional[FREDFetcher] = None
        self._yf_fetcher: Optional[YFinanceFetcher] = None
        self._av_fetcher: Optional[AlphaVantageFetcher] = None
        
        self.cache_max_age_days = cache_max_age_days
        
        logger.info("DataPipeline initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for default config locations
            default_paths = [
                "config/data_sources.yaml",
                "../config/data_sources.yaml",
                Path(__file__).parent.parent.parent / "config" / "data_sources.yaml",
            ]
            for path in default_paths:
                if Path(path).exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        
        logger.warning("No configuration file found, using defaults")
        return {}
    
    @property
    def fred_fetcher(self) -> FREDFetcher:
        """Lazily initialize FRED fetcher."""
        if self._fred_fetcher is None:
            self._fred_fetcher = FREDFetcher()
        return self._fred_fetcher
    
    @property
    def yf_fetcher(self) -> YFinanceFetcher:
        """Lazily initialize yfinance fetcher."""
        if self._yf_fetcher is None:
            self._yf_fetcher = YFinanceFetcher()
        return self._yf_fetcher
    
    @property
    def av_fetcher(self) -> AlphaVantageFetcher:
        """Lazily initialize Alpha Vantage fetcher."""
        if self._av_fetcher is None:
            self._av_fetcher = AlphaVantageFetcher()
        return self._av_fetcher
    
    def fetch_market_data(
        self,
        assets: Optional[List[str]] = None,
        start_date: Union[str, datetime] = "2010-01-01",
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        validate: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for specified assets.
        
        Implements fallback logic:
        1. Check local database cache (if fresh enough)
        2. Try yfinance (no API key needed, fast)
        3. Try FRED (for supported series)
        4. Try Alpha Vantage (rate limited)
        
        Args:
            assets: List of asset symbols. If None, fetches REQUIRED_ASSETS
            start_date: Start date for data range
            end_date: End date (defaults to today)
            use_cache: If True, use cached data if fresh enough
            validate: If True, validate data quality
            
        Returns:
            Dictionary mapping asset symbol to validated DataFrame
            
        Raises:
            ValueError: If no data can be retrieved for required assets
        """
        if assets is None:
            assets = self.REQUIRED_ASSETS
        
        if end_date is None:
            end_date = datetime.now()
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        logger.info(
            f"Fetching data for {assets} from {start_date.date()} to {end_date.date()}"
        )
        
        results = {}
        
        for asset in assets:
            df = self._fetch_single_asset(
                asset, start_date, end_date, use_cache
            )
            
            if df is not None and not df.empty:
                results[asset] = df
            else:
                logger.error(f"Failed to fetch data for {asset}")
        
        # Validate all data
        if validate and results:
            results, reports = validate_multi_asset(results, self.validator)
            
            for asset, report in reports.items():
                if not report.is_valid:
                    logger.warning(f"Validation issues for {asset}:\n{report}")
        
        # Save to database
        for asset, df in results.items():
            try:
                self.storage.save_market_data(asset, df)
            except Exception as e:
                logger.warning(f"Failed to save {asset} to database: {e}")
        
        return results
    
    def _fetch_single_asset(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single asset with fallback logic."""
        
        # Step 1: Check cache
        if use_cache:
            cached = self._get_from_cache(asset, start_date, end_date)
            if cached is not None:
                logger.info(f"Using cached data for {asset}")
                return cached
        
        # Step 2: Try yfinance (primary for most assets)
        try:
            df = self.yf_fetcher.fetch(asset, start_date, end_date)
            if df is not None and not df.empty:
                logger.info(f"Fetched {asset} from yfinance")
                return df
        except Exception as e:
            logger.debug(f"yfinance failed for {asset}: {e}")
        
        # Step 3: Try FRED (for supported series)
        if asset in FREDFetcher.SYMBOL_MAPPING and os.getenv("FRED_API_KEY"):
            try:
                df = self.fred_fetcher.fetch(asset, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(f"Fetched {asset} from FRED")
                    return df
            except Exception as e:
                logger.debug(f"FRED failed for {asset}: {e}")
        
        # Step 4: Try Alpha Vantage (rate limited fallback)
        if os.getenv("ALPHAVANTAGE_API_KEY"):
            try:
                df = self.av_fetcher.fetch(asset, start_date, end_date)
                if df is not None and not df.empty:
                    logger.info(f"Fetched {asset} from Alpha Vantage")
                    return df
            except Exception as e:
                logger.debug(f"Alpha Vantage failed for {asset}: {e}")
        
        # Step 5: Fall back to stale cache
        stale_cached = self.storage.load_market_data(asset, start_date, end_date)
        if not stale_cached.empty:
            logger.warning(f"Using stale cached data for {asset}")
            return stale_cached
        
        return None
    
    def _get_from_cache(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get data from cache if fresh enough."""
        if not self.storage.is_data_fresh(asset, self.cache_max_age_days):
            return None
        
        date_range = self.storage.get_date_range(asset)
        if date_range is None:
            return None
        
        cached_start, cached_end = date_range
        
        # Check if cache covers requested range
        if cached_start <= start_date and cached_end >= end_date - timedelta(days=3):
            return self.storage.load_market_data(asset, start_date, end_date)
        
        return None
    
    def fetch_yield_curve(
        self,
        start_date: Union[str, datetime] = "2010-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Fetch Treasury yield curve data from FRED.
        
        Args:
            start_date: Start date for data range
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with columns for each tenor (1M, 3M, 6M, 1Y, ..., 30Y)
        """
        tenors = [
            "DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3",
            "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"
        ]
        
        if not os.getenv("FRED_API_KEY"):
            logger.error("FRED_API_KEY required for yield curve data")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now()
        
        data = {}
        for tenor in tenors:
            try:
                df = self.fred_fetcher.fetch(tenor, start_date, end_date)
                if not df.empty:
                    data[tenor] = df["Close"]
            except Exception as e:
                logger.warning(f"Failed to fetch {tenor}: {e}")
        
        if not data:
            return pd.DataFrame()
        
        yield_curve = pd.DataFrame(data)
        yield_curve = yield_curve.ffill()  # Forward fill missing values
        
        logger.info(f"Fetched yield curve with {len(yield_curve)} observations")
        return yield_curve
    
    def fetch_macro_data(
        self,
        series: List[str],
        start_date: Union[str, datetime] = "2010-01-01",
        end_date: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch macroeconomic data from FRED.
        
        Args:
            series: List of macro series (e.g., ["NFP", "CPI", "UNRATE"])
            start_date: Start date for data range
            end_date: End date (defaults to today)
            
        Returns:
            Dictionary mapping series name to DataFrame
        """
        if not os.getenv("FRED_API_KEY"):
            logger.error("FRED_API_KEY required for macro data")
            return {}
        
        if end_date is None:
            end_date = datetime.now()
        
        results = {}
        for s in series:
            try:
                df = self.fred_fetcher.fetch(s, start_date, end_date)
                if not df.empty:
                    results[s] = df
                    self.storage.save_macro_data(s, df)
            except Exception as e:
                logger.warning(f"Failed to fetch {s}: {e}")
        
        return results
    
    def calculate_returns(
        self,
        data: Dict[str, pd.DataFrame],
        method: str = "log",
    ) -> pd.DataFrame:
        """Calculate returns matrix from price data.
        
        Args:
            data: Dictionary mapping asset to price DataFrame
            method: "log" for log returns, "simple" for simple returns
            
        Returns:
            DataFrame with returns for each asset, aligned by date
        """
        returns = {}
        
        for asset, df in data.items():
            if "Close" not in df.columns:
                logger.warning(f"No Close column for {asset}, skipping")
                continue
            
            prices = df["Close"]
            
            if method == "log":
                ret = np.log(prices / prices.shift(1))
            else:
                ret = prices.pct_change()
            
            returns[asset] = ret
        
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df.dropna()
        
        logger.info(
            f"Calculated {method} returns: {len(returns_df)} observations, "
            f"{len(returns_df.columns)} assets"
        )
        
        return returns_df
    
    def prepare_regime_features(
        self,
        data: Dict[str, pd.DataFrame],
        lookback: int = 20,
    ) -> pd.DataFrame:
        """Prepare feature matrix for regime classification.
        
        Creates features including:
        - Asset returns
        - Rolling volatility
        - Cross-asset correlations
        - VIX level and percentile
        
        Args:
            data: Dictionary mapping asset to price DataFrame
            lookback: Rolling window for feature calculation
            
        Returns:
            DataFrame with features aligned by date
        """
        # Calculate returns
        returns = self.calculate_returns(data, method="log")
        
        features = pd.DataFrame(index=returns.index)
        
        # 1. Returns (already calculated)
        for asset in returns.columns:
            features[f"{asset}_return"] = returns[asset]
        
        # 2. Rolling volatility
        for asset in returns.columns:
            features[f"{asset}_vol_{lookback}d"] = (
                returns[asset].rolling(lookback).std() * np.sqrt(252)
            )
        
        # 3. VIX level and percentile (if available)
        if "VIX" in data:
            vix = data["VIX"]["Close"]
            features["VIX_level"] = vix
            features["VIX_percentile"] = vix.rolling(252).rank(pct=True)
        
        # 4. Key correlations
        if "SPX" in returns.columns and "TLT" in returns.columns:
            features["SPX_TLT_corr_30d"] = (
                returns["SPX"].rolling(30).corr(returns["TLT"])
            )
        
        if "SPX" in returns.columns and "GLD" in returns.columns:
            features["SPX_GLD_corr_30d"] = (
                returns["SPX"].rolling(30).corr(returns["GLD"])
            )
        
        # 5. Momentum indicators
        for asset in ["SPX", "TLT", "GLD"]:
            if asset in data:
                prices = data[asset]["Close"]
                features[f"{asset}_mom_20d"] = prices / prices.shift(20) - 1
                features[f"{asset}_mom_60d"] = prices / prices.shift(60) - 1
        
        # Drop NaN rows from rolling calculations
        features = features.dropna()
        
        logger.info(
            f"Prepared {len(features.columns)} features with "
            f"{len(features)} observations"
        )
        
        return features
    
    def get_data_summary(self) -> Dict:
        """Get summary of all available data in the database.
        
        Returns:
            Dictionary with data availability information
        """
        assets = self.storage.get_available_assets()
        
        summary = {
            "available_assets": assets,
            "asset_details": {},
        }
        
        for asset in assets:
            date_range = self.storage.get_date_range(asset)
            if date_range:
                start, end = date_range
                summary["asset_details"][asset] = {
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"),
                    "is_fresh": self.storage.is_data_fresh(asset, 1),
                }
        
        return summary
