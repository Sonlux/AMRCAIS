"""
Database Storage for AMRCAIS.

This module provides SQLite/PostgreSQL storage for market data with
efficient indexing and retrieval capabilities.

Classes:
    DatabaseStorage: Main storage class for persisting market data

Example:
    >>> storage = DatabaseStorage("data/amrcais.db")
    >>> storage.save_market_data("SPX", df)
    >>> retrieved = storage.load_market_data("SPX", "2020-01-01", "2024-12-31")
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import os

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Integer,
    Index,
    MetaData,
    Table,
    select,
    delete,
    and_,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()


class MarketData(Base):
    """SQLAlchemy model for market data storage."""
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    adjusted_close = Column(Float)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index("idx_asset_date", "asset", "date", unique=True),
    )


class MacroData(Base):
    """SQLAlchemy model for macroeconomic data storage."""
    
    __tablename__ = "macro_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    series = Column(String(50), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    
    __table_args__ = (
        Index("idx_series_date", "series", "date", unique=True),
    )


class RegimeHistory(Base):
    """SQLAlchemy model for storing regime classifications."""
    
    __tablename__ = "regime_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True, unique=True)
    regime = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    disagreement = Column(Float, nullable=False)
    hmm_vote = Column(Integer)
    ml_vote = Column(Integer)
    correlation_vote = Column(Integer)
    volatility_vote = Column(Integer)


class DatabaseStorage:
    """Storage manager for AMRCAIS data persistence.
    
    Provides efficient storage and retrieval of:
    - Market price data (OHLCV)
    - Macroeconomic data
    - Regime classifications
    
    Supports SQLite for local development and PostgreSQL for production.
    
    Attributes:
        engine: SQLAlchemy database engine
        Session: Session factory for database operations
    
    Example:
        >>> storage = DatabaseStorage("data/amrcais.db")
        >>> storage.save_market_data("SPX", price_df)
        >>> df = storage.load_market_data("SPX", "2020-01-01", "2024-12-31")
    """
    
    def __init__(
        self,
        db_path: str = "data/amrcais.db",
        echo: bool = False,
    ):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database or PostgreSQL connection string
            echo: If True, log all SQL statements
        """
        # Ensure directory exists
        if not db_path.startswith("postgresql"):
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
        else:
            connection_string = db_path
        
        self.engine = create_engine(connection_string, echo=echo)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Database initialized: {db_path}")
    
    def save_market_data(
        self,
        asset: str,
        df: pd.DataFrame,
        replace: bool = True,
    ) -> int:
        """Save market data to database.
        
        Args:
            asset: Asset symbol (e.g., "SPX", "TLT")
            df: DataFrame with Date index and OHLCV columns
            replace: If True, replace existing data for overlapping dates
            
        Returns:
            Number of rows saved
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
        """
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame for {asset}")
        
        if "Close" not in df.columns:
            raise ValueError(f"DataFrame must have 'Close' column for {asset}")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        session = self.Session()
        rows_saved = 0
        
        try:
            if replace:
                # Delete existing data for this asset in date range
                min_date = df.index.min()
                max_date = df.index.max()
                
                delete_stmt = delete(MarketData).where(
                    and_(
                        MarketData.asset == asset,
                        MarketData.date >= min_date,
                        MarketData.date <= max_date,
                    )
                )
                session.execute(delete_stmt)
            
            # Insert new data
            for date, row in df.iterrows():
                record = MarketData(
                    asset=asset,
                    date=date,
                    open=row.get("Open"),
                    high=row.get("High"),
                    low=row.get("Low"),
                    close=row["Close"],
                    volume=row.get("Volume"),
                    adjusted_close=row.get("Adj Close", row.get("Adjusted_close")),
                )
                session.merge(record)  # Upsert
                rows_saved += 1
            
            session.commit()
            logger.info(f"Saved {rows_saved} rows for {asset}")
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error saving {asset}: {e}")
            raise
        finally:
            session.close()
        
        return rows_saved
    
    def load_market_data(
        self,
        asset: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Load market data from database.
        
        Args:
            asset: Asset symbol to load
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            
        Returns:
            DataFrame with OHLCV data and Date index
        """
        session = self.Session()
        
        try:
            query = select(MarketData).where(MarketData.asset == asset)
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                query = query.where(MarketData.date >= start_date)
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                query = query.where(MarketData.date <= end_date)
            
            query = query.order_by(MarketData.date)
            
            result = session.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                logger.warning(f"No data found for {asset}")
                return pd.DataFrame()
            
            data = []
            for row in rows:
                data.append({
                    "Date": row.date,
                    "Open": row.open,
                    "High": row.high,
                    "Low": row.low,
                    "Close": row.close,
                    "Volume": row.volume,
                })
            
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            df = df.dropna(subset=["Close"])
            
            logger.debug(f"Loaded {len(df)} rows for {asset}")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database error loading {asset}: {e}")
            raise
        finally:
            session.close()
    
    def get_available_assets(self) -> List[str]:
        """Get list of all assets with data in database.
        
        Returns:
            List of asset symbols
        """
        session = self.Session()
        
        try:
            from sqlalchemy import distinct
            query = select(distinct(MarketData.asset))
            result = session.execute(query)
            return [row[0] for row in result]
        finally:
            session.close()
    
    def get_date_range(self, asset: str) -> Optional[tuple]:
        """Get the date range of data available for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Tuple of (min_date, max_date) or None if no data
        """
        session = self.Session()
        
        try:
            from sqlalchemy import func
            
            query = select(
                func.min(MarketData.date),
                func.max(MarketData.date),
            ).where(MarketData.asset == asset)
            
            result = session.execute(query).fetchone()
            
            if result[0] is None:
                return None
            
            return (result[0], result[1])
            
        finally:
            session.close()
    
    def save_macro_data(
        self,
        series: str,
        df: pd.DataFrame,
        replace: bool = True,
    ) -> int:
        """Save macroeconomic data to database.
        
        Args:
            series: Series identifier (e.g., "NFP", "CPI")
            df: DataFrame with Date index and value column
            replace: If True, replace existing data
            
        Returns:
            Number of rows saved
        """
        if df.empty:
            raise ValueError(f"Cannot save empty DataFrame for {series}")
        
        # Get the value column (first non-Date column)
        value_col = df.columns[0] if isinstance(df.columns[0], str) else "Close"
        
        session = self.Session()
        rows_saved = 0
        
        try:
            if replace:
                min_date = df.index.min()
                max_date = df.index.max()
                
                delete_stmt = delete(MacroData).where(
                    and_(
                        MacroData.series == series,
                        MacroData.date >= min_date,
                        MacroData.date <= max_date,
                    )
                )
                session.execute(delete_stmt)
            
            for date, row in df.iterrows():
                value = row[value_col] if isinstance(row, pd.Series) else row
                
                record = MacroData(
                    series=series,
                    date=date,
                    value=float(value),
                )
                session.merge(record)
                rows_saved += 1
            
            session.commit()
            logger.info(f"Saved {rows_saved} rows for macro series {series}")
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error saving {series}: {e}")
            raise
        finally:
            session.close()
        
        return rows_saved
    
    def load_macro_data(
        self,
        series: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Load macroeconomic data from database.
        
        Args:
            series: Series identifier
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            DataFrame with Date index and value column
        """
        session = self.Session()
        
        try:
            query = select(MacroData).where(MacroData.series == series)
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                query = query.where(MacroData.date >= start_date)
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                query = query.where(MacroData.date <= end_date)
            
            query = query.order_by(MacroData.date)
            
            result = session.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                return pd.DataFrame()
            
            data = [{"Date": r.date, "Value": r.value} for r in rows]
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            
            return df
            
        finally:
            session.close()
    
    def save_regime(
        self,
        date: datetime,
        regime: int,
        confidence: float,
        disagreement: float,
        votes: Optional[Dict[str, int]] = None,
    ) -> None:
        """Save a single regime classification.
        
        Args:
            date: Date of classification
            regime: Regime ID (1-4)
            confidence: Confidence score (0-1)
            disagreement: Disagreement index (0-1)
            votes: Optional dict of individual classifier votes
        """
        session = self.Session()
        
        try:
            record = RegimeHistory(
                date=date,
                regime=regime,
                confidence=confidence,
                disagreement=disagreement,
                hmm_vote=votes.get("hmm") if votes else None,
                ml_vote=votes.get("ml") if votes else None,
                correlation_vote=votes.get("correlation") if votes else None,
                volatility_vote=votes.get("volatility") if votes else None,
            )
            session.merge(record)
            session.commit()
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error saving regime: {e}")
            raise
        finally:
            session.close()
    
    def load_regime_history(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """Load regime classification history.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            DataFrame with regime classifications
        """
        session = self.Session()
        
        try:
            query = select(RegimeHistory)
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, "%Y-%m-%d")
                query = query.where(RegimeHistory.date >= start_date)
            
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, "%Y-%m-%d")
                query = query.where(RegimeHistory.date <= end_date)
            
            query = query.order_by(RegimeHistory.date)
            
            result = session.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                return pd.DataFrame()
            
            data = []
            for r in rows:
                data.append({
                    "Date": r.date,
                    "Regime": r.regime,
                    "Confidence": r.confidence,
                    "Disagreement": r.disagreement,
                    "HMM_Vote": r.hmm_vote,
                    "ML_Vote": r.ml_vote,
                    "Correlation_Vote": r.correlation_vote,
                    "Volatility_Vote": r.volatility_vote,
                })
            
            df = pd.DataFrame(data)
            df.set_index("Date", inplace=True)
            
            return df
            
        finally:
            session.close()
    
    def is_data_fresh(
        self,
        asset: str,
        max_age_days: int = 1,
    ) -> bool:
        """Check if data for an asset is recent enough.
        
        Args:
            asset: Asset symbol
            max_age_days: Maximum allowed age in days
            
        Returns:
            True if data is fresh enough, False otherwise
        """
        date_range = self.get_date_range(asset)
        
        if date_range is None:
            return False
        
        _, max_date = date_range
        age = datetime.now() - max_date
        
        return age.days <= max_age_days
    
    def clear_all_data(self) -> None:
        """Clear all data from all tables. USE WITH CAUTION."""
        session = self.Session()
        
        try:
            session.execute(delete(MarketData))
            session.execute(delete(MacroData))
            session.execute(delete(RegimeHistory))
            session.commit()
            logger.warning("All database tables cleared")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            session.close()
