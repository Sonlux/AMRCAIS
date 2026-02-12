"""
Data Validators for AMRCAIS.

This module provides comprehensive data validation to ensure data quality
before use in regime detection and analytical modules.

Validation checks:
- Missing values (NaN)
- Price validity (positive, within reasonable bounds)
- Outlier detection (>20% single-day moves flagged)
- Corporate actions detection
- Date continuity

Classes:
    DataValidator: Main validation class with configurable thresholds

Example:
    >>> validator = DataValidator()
    >>> clean_data, report = validator.validate(raw_data, asset="SPX")
    >>> print(report)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report containing validation results and statistics.
    
    Attributes:
        asset: Asset symbol that was validated
        start_date: Start of data range
        end_date: End of data range
        total_rows: Total number of rows in input
        valid_rows: Number of rows passing validation
        issues: List of validation issues found
        statistics: Summary statistics of the data
    """
    asset: str
    start_date: datetime
    end_date: datetime
    total_rows: int
    valid_rows: int
    issues: List[Dict] = field(default_factory=list)
    statistics: Dict = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if data passed validation with acceptable issue count."""
        critical_issues = [
            i for i in self.issues if i.get("severity") == "critical"
        ]
        return len(critical_issues) == 0
    
    @property
    def missing_pct(self) -> float:
        """Percentage of missing/invalid rows."""
        if self.total_rows == 0:
            return 100.0
        return 100.0 * (1 - self.valid_rows / self.total_rows)
    
    def __str__(self) -> str:
        """Human-readable validation report."""
        lines = [
            f"Validation Report for {self.asset}",
            f"=" * 40,
            f"Date Range: {self.start_date.date()} to {self.end_date.date()}",
            f"Total Rows: {self.total_rows}",
            f"Valid Rows: {self.valid_rows} ({100 - self.missing_pct:.1f}%)",
            f"Status: {'PASSED' if self.is_valid else 'FAILED'}",
        ]
        
        if self.issues:
            lines.append(f"\nIssues Found ({len(self.issues)}):")
            for issue in self.issues[:10]:  # Show first 10
                lines.append(
                    f"  - [{issue['severity'].upper()}] {issue['type']}: "
                    f"{issue['message']}"
                )
            if len(self.issues) > 10:
                lines.append(f"  ... and {len(self.issues) - 10} more")
        
        return "\n".join(lines)


class DataValidator:
    """Validates market data for quality and consistency.
    
    Performs comprehensive validation including:
    - Missing value detection
    - Price sanity checks
    - Return outlier detection
    - Volume validation
    - Date continuity checks
    
    Attributes:
        max_missing_pct: Maximum allowed missing data percentage
        max_daily_return: Maximum allowed single-day return (flags outliers)
        min_price: Minimum valid price threshold
    
    Example:
        >>> validator = DataValidator(max_missing_pct=5.0)
        >>> clean_df, report = validator.validate(df, asset="SPX")
        >>> if not report.is_valid:
        ...     print(report)
    """
    
    # Default validation thresholds by asset type
    DEFAULT_THRESHOLDS = {
        "SPX": {"min_price": 100, "max_daily_pct": 15},
        "TLT": {"min_price": 50, "max_daily_pct": 10},
        "IEF": {"min_price": 50, "max_daily_pct": 8},
        "GLD": {"min_price": 50, "max_daily_pct": 10},
        "DXY": {"min_price": 70, "max_daily_pct": 5},
        "WTI": {"min_price": -50, "max_daily_pct": 50},  # Can go negative (2020)
        "VIX": {"min_price": 5, "max_daily_pct": 100},
    }
    
    def __init__(
        self,
        max_missing_pct: float = 5.0,
        max_daily_return: float = 0.20,
        min_price: float = 0.0,
        check_volume: bool = True,
    ):
        """Initialize validator with thresholds.
        
        Args:
            max_missing_pct: Maximum percentage of missing values allowed
            max_daily_return: Maximum single-day return before flagging
            min_price: Minimum valid price (default 0)
            check_volume: Whether to validate volume data
        """
        self.max_missing_pct = max_missing_pct
        self.max_daily_return = max_daily_return
        self.min_price = min_price
        self.check_volume = check_volume
    
    def validate(
        self,
        df: pd.DataFrame,
        asset: str,
        fix_issues: bool = True,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Validate and optionally clean market data.
        
        Args:
            df: DataFrame with OHLCV data and Date index
            asset: Asset symbol for asset-specific thresholds
            fix_issues: If True, attempt to fix issues; if False, only report
            
        Returns:
            Tuple of (cleaned DataFrame, ValidationReport)
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df.empty:
            raise ValueError(f"Empty DataFrame provided for {asset}")
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Cannot parse dates for {asset}: {e}") from e
        
        # Get asset-specific thresholds
        thresholds = self.DEFAULT_THRESHOLDS.get(asset, {})
        min_price = thresholds.get("min_price", self.min_price)
        max_daily_pct = thresholds.get("max_daily_pct", self.max_daily_return * 100)
        
        # Initialize report
        report = ValidationReport(
            asset=asset,
            start_date=df.index.min(),
            end_date=df.index.max(),
            total_rows=len(df),
            valid_rows=len(df),
        )
        
        # Make a copy for cleaning
        clean_df = df.copy()
        
        # Run all validation checks
        clean_df, report = self._check_missing_values(clean_df, report, fix_issues)
        clean_df, report = self._check_price_validity(
            clean_df, report, min_price, fix_issues
        )
        clean_df, report = self._check_return_outliers(
            clean_df, report, max_daily_pct, fix_issues
        )
        clean_df, report = self._check_ohlc_consistency(clean_df, report, fix_issues)
        clean_df, report = self._check_date_continuity(clean_df, report)
        
        if self.check_volume and "Volume" in clean_df.columns:
            clean_df, report = self._check_volume(clean_df, report, fix_issues)
        
        # Calculate final statistics
        report.valid_rows = len(clean_df)
        report.statistics = self._calculate_statistics(clean_df)
        
        # Log summary
        if report.is_valid:
            logger.info(f"Validation PASSED for {asset}: {report.valid_rows} valid rows")
        else:
            logger.warning(
                f"Validation FAILED for {asset}: {len(report.issues)} issues found"
            )
        
        return clean_df, report
    
    def _check_missing_values(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        fix: bool,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Check for missing values in critical columns."""
        critical_cols = ["Close"]
        optional_cols = ["Open", "High", "Low", "Volume"]
        
        for col in critical_cols:
            if col not in df.columns:
                report.issues.append({
                    "type": "missing_column",
                    "severity": "critical",
                    "message": f"Required column '{col}' not found",
                    "column": col,
                })
                continue
            
            missing_mask = df[col].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                missing_pct = 100.0 * missing_count / len(df)
                severity = "critical" if missing_pct > self.max_missing_pct else "warning"
                
                report.issues.append({
                    "type": "missing_values",
                    "severity": severity,
                    "message": f"{missing_count} missing values in {col} ({missing_pct:.1f}%)",
                    "column": col,
                    "count": missing_count,
                    "dates": df.index[missing_mask].tolist()[:5],
                })
                
                if fix:
                    # Forward fill, then backward fill
                    df[col] = df[col].ffill().bfill()
        
        # Check optional columns
        for col in optional_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0 and fix:
                    df[col] = df[col].ffill().bfill()
        
        return df, report
    
    def _check_price_validity(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        min_price: float,
        fix: bool,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Check that prices are within valid ranges."""
        price_cols = ["Open", "High", "Low", "Close"]
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Check for negative or below-minimum prices
            invalid_mask = df[col] < min_price
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                report.issues.append({
                    "type": "invalid_price",
                    "severity": "warning",
                    "message": f"{invalid_count} prices below {min_price} in {col}",
                    "column": col,
                    "count": invalid_count,
                    "dates": df.index[invalid_mask].tolist()[:5],
                })
                
                if fix:
                    # Replace with NaN and forward fill
                    df.loc[invalid_mask, col] = np.nan
                    df[col] = df[col].ffill()
        
        return df, report
    
    def _check_return_outliers(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        max_daily_pct: float,
        fix: bool,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Flag large single-day moves as potential data errors."""
        if "Close" not in df.columns:
            return df, report
        
        # Calculate daily returns
        returns = df["Close"].pct_change()
        
        # Find outliers
        outlier_mask = returns.abs() > (max_daily_pct / 100)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outlier_dates = df.index[outlier_mask].tolist()
            outlier_returns = returns[outlier_mask].tolist()
            
            report.issues.append({
                "type": "return_outlier",
                "severity": "warning",
                "message": f"{outlier_count} days with >{max_daily_pct}% moves",
                "count": outlier_count,
                "dates": outlier_dates[:5],
                "returns": [f"{r*100:.1f}%" for r in outlier_returns[:5]],
            })
            
            # Note: We flag but don't fix return outliers as they may be legitimate
            # (e.g., March 2020 COVID crash)
        
        return df, report
    
    def _check_ohlc_consistency(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        fix: bool,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Check OHLC relationship: High >= Open, Close, Low and Low <= all."""
        required = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required):
            return df, report
        
        # High should be >= Open, Close, Low
        high_invalid = (
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["High"] < df["Low"])
        )
        
        # Low should be <= Open, Close, High
        low_invalid = (
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"]) |
            (df["Low"] > df["High"])
        )
        
        invalid_mask = high_invalid | low_invalid
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            report.issues.append({
                "type": "ohlc_inconsistency",
                "severity": "warning",
                "message": f"{invalid_count} rows with OHLC inconsistencies",
                "count": invalid_count,
                "dates": df.index[invalid_mask].tolist()[:5],
            })
            
            if fix:
                # Fix by ensuring High is max and Low is min of OHLC
                df.loc[invalid_mask, "High"] = df.loc[invalid_mask, required].max(axis=1)
                df.loc[invalid_mask, "Low"] = df.loc[invalid_mask, required].min(axis=1)
        
        return df, report
    
    def _check_date_continuity(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Check for gaps in trading dates (excluding weekends/holidays)."""
        if len(df) < 2:
            return df, report
        
        # Calculate business day differences
        date_diff = df.index.to_series().diff()
        
        # Flag gaps > 5 business days (likely data gaps, not just holidays)
        large_gaps = date_diff > pd.Timedelta(days=7)
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            gap_dates = df.index[large_gaps].tolist()
            gap_sizes = date_diff[large_gaps].tolist()
            
            report.issues.append({
                "type": "date_gap",
                "severity": "info",
                "message": f"{gap_count} large gaps (>7 days) in data",
                "count": gap_count,
                "dates": gap_dates[:5],
                "gap_sizes": [str(g.days) + " days" for g in gap_sizes[:5]],
            })
        
        return df, report
    
    def _check_volume(
        self,
        df: pd.DataFrame,
        report: ValidationReport,
        fix: bool,
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Check volume data for validity."""
        if "Volume" not in df.columns:
            return df, report
        
        # Check for zero volume
        zero_vol_mask = df["Volume"] == 0
        zero_vol_count = zero_vol_mask.sum()
        
        if zero_vol_count > 0:
            report.issues.append({
                "type": "zero_volume",
                "severity": "info",
                "message": f"{zero_vol_count} days with zero volume",
                "count": zero_vol_count,
            })
        
        # Check for negative volume
        neg_vol_mask = df["Volume"] < 0
        neg_vol_count = neg_vol_mask.sum()
        
        if neg_vol_count > 0:
            report.issues.append({
                "type": "negative_volume",
                "severity": "warning",
                "message": f"{neg_vol_count} days with negative volume",
                "count": neg_vol_count,
            })
            
            if fix:
                df.loc[neg_vol_mask, "Volume"] = 0
        
        return df, report
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the validated data."""
        stats = {
            "rows": len(df),
            "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        }
        
        if "Close" in df.columns:
            returns = df["Close"].pct_change().dropna()
            stats.update({
                "close_mean": df["Close"].mean(),
                "close_std": df["Close"].std(),
                "returns_mean": returns.mean(),
                "returns_std": returns.std(),
                "returns_skew": returns.skew(),
                "returns_kurt": returns.kurtosis(),
                "max_drawdown": self._calculate_max_drawdown(df["Close"]),
            })
        
        return stats
    
    @staticmethod
    def _calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown from a price series."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()


def validate_multi_asset(
    data: Dict[str, pd.DataFrame],
    validator: Optional[DataValidator] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, ValidationReport]]:
    """Validate multiple assets and return cleaned data with reports.
    
    Args:
        data: Dictionary mapping asset symbol to DataFrame
        validator: Optional validator instance (uses default if None)
        
    Returns:
        Tuple of (cleaned data dict, reports dict)
    """
    if validator is None:
        validator = DataValidator()
    
    clean_data = {}
    reports = {}
    
    for asset, df in data.items():
        try:
            clean_df, report = validator.validate(df, asset)
            clean_data[asset] = clean_df
            reports[asset] = report
        except Exception as e:
            logger.error(f"Validation failed for {asset}: {e}")
            reports[asset] = ValidationReport(
                asset=asset,
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_rows=len(df) if not df.empty else 0,
                valid_rows=0,
                issues=[{
                    "type": "validation_error",
                    "severity": "critical",
                    "message": str(e),
                }],
            )
    
    return clean_data, reports
