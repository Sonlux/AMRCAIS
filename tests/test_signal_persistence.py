"""
Tests for module signal persistence pipeline.

Validates:
    - DatabaseStorage.save_module_signal() writes to SQLite
    - DatabaseStorage.load_module_signals() retrieves history
    - Signal data round-trips correctly (signal, strength, confidence, regime)
    - Module name filtering works
    - Limit parameter is respected
    - Empty database returns empty DataFrame
    - Metadata JSON serialization
"""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data_pipeline.storage import DatabaseStorage


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_signals.db")
        storage = DatabaseStorage(db_path=db_path)
        yield storage
        # Dispose engine before tmpdir cleanup to release SQLite lock (Windows)
        storage.engine.dispose()



@pytest.fixture
def populated_db(temp_db):
    """Database pre-populated with sample signal data."""
    base_date = datetime(2026, 1, 1, 12, 0, 0)

    # Insert signals for multiple modules across multiple days
    sample_signals = [
        ("macro", "bullish", 0.8, 0.9, "GDP strong", "Risk-on", 1),
        ("macro", "bearish", 0.6, 0.7, "CPI rising", "Stagflation", 3),
        ("macro", "neutral", 0.3, 0.5, "Mixed data", "Transition", 1),
        ("yield_curve", "cautious", 0.5, 0.6, "Curve flat", "Late cycle", 1),
        ("yield_curve", "bearish", 0.7, 0.8, "Inverted", "Risk-off", 2),
        ("options", "neutral", 0.2, 0.5, "Vol normal", "Calm", 4),
        ("factors", "bullish", 0.9, 0.85, "Quality leads", "Growth", 1),
        ("correlations", "cautious", 0.4, 0.55, "Rising corr", "Uncertainty", 3),
    ]

    for i, (name, signal, strength, conf, expl, ctx, regime) in enumerate(sample_signals):
        temp_db.save_module_signal(
            module_name=name,
            signal=signal,
            strength=strength,
            confidence=conf,
            explanation=expl,
            regime_context=ctx,
            regime_id=regime,
            metadata={"test_index": i},
            date=base_date + timedelta(hours=i),
        )

    return temp_db


# ─── Basic Save/Load Tests ───────────────────────────────────────


class TestSignalSaveLoad:
    """Test basic signal persistence."""

    def test_save_single_signal(self, temp_db):
        """Saving a signal should not raise."""
        temp_db.save_module_signal(
            module_name="macro",
            signal="bullish",
            strength=0.8,
            confidence=0.9,
            explanation="GDP strong",
            regime_context="Risk-on",
            regime_id=1,
        )
        # Should not raise

    def test_load_after_save(self, temp_db):
        """A saved signal should be retrievable."""
        temp_db.save_module_signal(
            module_name="macro",
            signal="bullish",
            strength=0.8,
            confidence=0.9,
            explanation="GDP strong",
            regime_id=1,
        )
        df = temp_db.load_module_signals(module_name="macro")
        assert not df.empty
        assert len(df) == 1

    def test_signal_round_trip(self, temp_db):
        """Signal fields should survive the save/load round trip."""
        temp_db.save_module_signal(
            module_name="yield_curve",
            signal="bearish",
            strength=0.75,
            confidence=0.82,
            explanation="Curve inverted",
            regime_context="Risk-off crisis",
            regime_id=2,
            date=datetime(2026, 2, 1, 10, 0, 0),
        )
        df = temp_db.load_module_signals(module_name="yield_curve")
        row = df.iloc[0]

        assert row["Module"] == "yield_curve"
        assert row["Signal"] == "bearish"
        assert abs(row["Strength"] - 0.75) < 0.01
        assert abs(row["Confidence"] - 0.82) < 0.01
        assert row["Regime_ID"] == 2

    def test_empty_database_returns_empty(self, temp_db):
        """Loading from empty DB returns empty DataFrame."""
        df = temp_db.load_module_signals(module_name="macro")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ─── Filtering Tests ─────────────────────────────────────────────


class TestSignalFiltering:
    """Test module name filtering and limits."""

    def test_filter_by_module_name(self, populated_db):
        """Filtering by module name returns only that module's signals."""
        df_macro = populated_db.load_module_signals(module_name="macro")
        assert len(df_macro) == 3
        assert all(df_macro["Module"] == "macro")

    def test_filter_different_module(self, populated_db):
        """Different module returns different signals."""
        df_yc = populated_db.load_module_signals(module_name="yield_curve")
        assert len(df_yc) == 2
        assert all(df_yc["Module"] == "yield_curve")

    def test_filter_single_entry_module(self, populated_db):
        """Module with only one entry works correctly."""
        df_opt = populated_db.load_module_signals(module_name="options")
        assert len(df_opt) == 1
        assert df_opt.iloc[0]["Signal"] == "neutral"

    def test_load_all_modules(self, populated_db):
        """Loading without filter returns all signals."""
        df_all = populated_db.load_module_signals()
        assert len(df_all) == 8  # Total sample signals

    def test_limit_parameter(self, populated_db):
        """Limit parameter caps returned rows."""
        df = populated_db.load_module_signals(limit=3)
        assert len(df) <= 3

    def test_nonexistent_module_returns_empty(self, populated_db):
        """Querying a module that has no data returns empty."""
        df = populated_db.load_module_signals(module_name="nonexistent")
        assert df.empty


# ─── Metadata Tests ──────────────────────────────────────────────


class TestSignalMetadata:
    """Test JSON metadata serialization."""

    def test_metadata_saved(self, temp_db):
        """Metadata dict should be serialized as JSON."""
        meta = {"slope_2_10": -0.2, "curvature": 0.15}
        temp_db.save_module_signal(
            module_name="yield_curve",
            signal="neutral",
            strength=0.3,
            confidence=0.5,
            metadata=meta,
        )
        # Verify by loading raw — we check the ORM column
        from sqlalchemy import select
        from src.data_pipeline.storage import ModuleSignalHistory

        session = temp_db.Session()
        try:
            row = session.execute(
                select(ModuleSignalHistory)
            ).scalars().first()
            assert row is not None
            parsed = json.loads(row.metadata_json)
            assert parsed["slope_2_10"] == -0.2
            assert parsed["curvature"] == 0.15
        finally:
            session.close()

    def test_none_metadata(self, temp_db):
        """None metadata should be stored as NULL."""
        temp_db.save_module_signal(
            module_name="macro",
            signal="bullish",
            strength=0.5,
            confidence=0.5,
            metadata=None,
        )
        from sqlalchemy import select
        from src.data_pipeline.storage import ModuleSignalHistory

        session = temp_db.Session()
        try:
            row = session.execute(
                select(ModuleSignalHistory)
            ).scalars().first()
            assert row.metadata_json is None
        finally:
            session.close()


# ─── Date Range Tests ────────────────────────────────────────────


class TestSignalDateRange:
    """Test date-based filtering."""

    def test_filter_by_start_date(self, populated_db):
        """Start date filter should exclude earlier signals."""
        df = populated_db.load_module_signals(
            start_date=datetime(2026, 1, 1, 16, 0, 0),
        )
        # Signals at hours 16:00-19:00 (indices 4-7)
        assert len(df) <= 4

    def test_filter_by_end_date(self, populated_db):
        """End date filter should exclude later signals."""
        df = populated_db.load_module_signals(
            end_date=datetime(2026, 1, 1, 14, 0, 0),
        )
        # Signals at hours 12:00-14:00 (indices 0-2)
        assert len(df) <= 3

    def test_filter_by_date_range(self, populated_db):
        """Combined date range should narrow results."""
        df = populated_db.load_module_signals(
            start_date=datetime(2026, 1, 1, 13, 0, 0),
            end_date=datetime(2026, 1, 1, 16, 0, 0),
        )
        assert len(df) >= 1
        assert len(df) <= 4
