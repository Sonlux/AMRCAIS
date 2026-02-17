"""
Tests for Phase 5: Network Effects + Moat.

Covers:
- KnowledgeBase (knowledge_base.py): transitions, anomalies, pattern matching, persistence
- AltDataIntegrator (alt_data.py): signal interpretation, regime voting, z-scores
- ResearchPublisher (research_publisher.py): case studies, backtests, factor analysis
- UserManager (user_manager.py): CRUD, RBAC, authentication, annotations
- Phase 5 API endpoints (routes/phase5.py)
"""

import json
import math
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Phase 5 Test Client Fixture ──────────────────────────────────


def _add_phase5_components(mock_system, tmp_path):
    """Attach real Phase 5 component instances to mock system."""
    from src.knowledge.knowledge_base import KnowledgeBase
    from src.knowledge.alt_data import AltDataIntegrator
    from src.knowledge.research_publisher import ResearchPublisher
    from src.knowledge.user_manager import UserManager

    kb = KnowledgeBase(storage_path=str(tmp_path / "kb.json"))
    adi = AltDataIntegrator()
    mgr = UserManager(storage_path=str(tmp_path / "users.json"))
    pub = ResearchPublisher(
        knowledge_base=kb,
        output_dir=str(tmp_path / "reports"),
    )

    mock_system.knowledge_base = kb
    mock_system.alt_data_integrator = adi
    mock_system.research_publisher = pub
    mock_system.user_manager = mgr
    return mock_system


@pytest.fixture
def phase5_client(mock_system, tmp_path):
    """FastAPI TestClient with Phase 5 real component instances."""
    import api.dependencies as deps
    from contextlib import asynccontextmanager

    _add_phase5_components(mock_system, tmp_path)

    orig_system = deps._system
    orig_time = deps._startup_time
    orig_analysis = deps._last_analysis

    deps._system = mock_system
    deps._startup_time = time.time()
    deps._last_analysis = None

    from api.main import app

    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    saved_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan

    from api.middleware import CSRFMiddleware, RateLimitMiddleware

    _orig_dispatch = RateLimitMiddleware.dispatch
    _orig_csrf_dispatch = CSRFMiddleware.dispatch

    async def _passthrough(self, request, call_next):
        return await call_next(request)

    RateLimitMiddleware.dispatch = _passthrough
    CSRFMiddleware.dispatch = _passthrough

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        yield client

    RateLimitMiddleware.dispatch = _orig_dispatch
    CSRFMiddleware.dispatch = _orig_csrf_dispatch
    app.router.lifespan_context = saved_lifespan
    deps._system = orig_system
    deps._startup_time = orig_time
    deps._last_analysis = orig_analysis


# ═══════════════════════════════════════════════════════════════════
#  Knowledge Base Tests (5.1)
# ═══════════════════════════════════════════════════════════════════


class TestRegimeTransitionRecord:
    """RegimeTransitionRecord dataclass tests."""

    def test_creation_defaults(self):
        from src.knowledge.knowledge_base import RegimeTransitionRecord

        r = RegimeTransitionRecord()
        assert r.from_regime == 0
        assert r.to_regime == 0
        assert r.confidence == 0.0
        assert len(r.transition_id) == 12
        assert isinstance(r.timestamp, str)

    def test_creation_with_values(self):
        from src.knowledge.knowledge_base import RegimeTransitionRecord

        r = RegimeTransitionRecord(
            from_regime=1,
            to_regime=2,
            confidence=0.85,
            disagreement=0.6,
            leading_indicators={"vix": 30.0, "spread": 0.05},
        )
        assert r.from_regime == 1
        assert r.to_regime == 2
        assert r.confidence == 0.85
        assert r.leading_indicators["vix"] == 30.0

    def test_to_dict(self):
        from src.knowledge.knowledge_base import RegimeTransitionRecord

        r = RegimeTransitionRecord(from_regime=1, to_regime=3, confidence=0.7)
        d = r.to_dict()
        assert d["from_regime"] == 1
        assert d["to_regime"] == 3
        assert d["confidence"] == 0.7
        assert "transition_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        from src.knowledge.knowledge_base import RegimeTransitionRecord

        data = {
            "transition_id": "abc123",
            "from_regime": 2,
            "to_regime": 4,
            "confidence": 0.9,
            "disagreement": 0.3,
            "leading_indicators": {"test": 1.0},
        }
        r = RegimeTransitionRecord.from_dict(data)
        assert r.transition_id == "abc123"
        assert r.from_regime == 2
        assert r.to_regime == 4
        assert r.confidence == 0.9

    def test_roundtrip_serialization(self):
        from src.knowledge.knowledge_base import RegimeTransitionRecord

        original = RegimeTransitionRecord(
            from_regime=1, to_regime=2, confidence=0.75,
            leading_indicators={"a": 1.0, "b": 2.0},
            classifier_accuracy={"hmm": True, "ml": False},
        )
        d = original.to_dict()
        restored = RegimeTransitionRecord.from_dict(d)
        assert restored.from_regime == original.from_regime
        assert restored.confidence == original.confidence
        assert restored.leading_indicators == original.leading_indicators


class TestAnomalyRecord:
    """AnomalyRecord dataclass tests."""

    def test_creation(self):
        from src.knowledge.knowledge_base import AnomalyRecord

        a = AnomalyRecord(
            anomaly_type="correlation_spike",
            asset_pair="SPX_TLT",
            regime=2,
            z_score=3.5,
        )
        assert a.anomaly_type == "correlation_spike"
        assert a.z_score == 3.5
        assert a.reversion_days is None

    def test_to_dict(self):
        from src.knowledge.knowledge_base import AnomalyRecord

        a = AnomalyRecord(
            anomaly_type="vol_divergence",
            asset_pair="GLD_DXY",
            regime=3,
            z_score=2.1,
            expected_value=0.3,
            actual_value=0.7,
        )
        d = a.to_dict()
        assert d["anomaly_type"] == "vol_divergence"
        assert d["z_score"] == 2.1
        assert d["expected_value"] == 0.3

    def test_from_dict(self):
        from src.knowledge.knowledge_base import AnomalyRecord

        data = {"anomaly_type": "test", "z_score": 1.5, "regime": 1}
        a = AnomalyRecord.from_dict(data)
        assert a.anomaly_type == "test"
        assert a.z_score == 1.5


class TestPatternMatch:
    """PatternMatch dataclass tests."""

    def test_creation(self):
        from src.knowledge.knowledge_base import PatternMatch, RegimeTransitionRecord

        record = RegimeTransitionRecord(from_regime=1, to_regime=2)
        pm = PatternMatch(record=record, similarity=0.87, days_ago=30.5)
        assert pm.similarity == 0.87
        assert pm.days_ago == 30.5

    def test_to_dict(self):
        from src.knowledge.knowledge_base import PatternMatch, RegimeTransitionRecord

        record = RegimeTransitionRecord(from_regime=1, to_regime=2)
        pm = PatternMatch(record=record, similarity=0.9, outcome_summary="Good")
        d = pm.to_dict()
        assert d["similarity"] == 0.9
        assert "transition" in d
        assert d["outcome_summary"] == "Good"


class TestKnowledgeBase:
    """KnowledgeBase core tests."""

    @pytest.fixture
    def kb(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase
        path = str(tmp_path / "kb_test.json")
        return KnowledgeBase(storage_path=path)

    def test_initialization(self, kb):
        assert kb._transitions == []
        assert kb._anomalies == []

    def test_record_transition(self, kb):
        r = kb.record_transition(from_regime=1, to_regime=2, confidence=0.8)
        assert r.from_regime == 1
        assert r.to_regime == 2
        assert len(kb._transitions) == 1

    def test_record_multiple_transitions(self, kb):
        kb.record_transition(1, 2, confidence=0.8)
        kb.record_transition(2, 3, confidence=0.7)
        kb.record_transition(3, 1, confidence=0.9)
        assert len(kb._transitions) == 3

    def test_get_transitions_unfiltered(self, kb):
        kb.record_transition(1, 2)
        kb.record_transition(2, 3)
        results = kb.get_transitions()
        assert len(results) == 2

    def test_get_transitions_filtered(self, kb):
        kb.record_transition(1, 2)
        kb.record_transition(2, 3)
        kb.record_transition(1, 3)
        results = kb.get_transitions(from_regime=1)
        assert len(results) == 2
        assert all(r.from_regime == 1 for r in results)

    def test_get_transitions_to_filter(self, kb):
        kb.record_transition(1, 2)
        kb.record_transition(2, 3)
        kb.record_transition(3, 2)
        results = kb.get_transitions(to_regime=2)
        assert len(results) == 2

    def test_get_transitions_limit(self, kb):
        for i in range(10):
            kb.record_transition(1, 2)
        results = kb.get_transitions(limit=3)
        assert len(results) == 3

    def test_max_transitions_cap(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(
            storage_path=str(tmp_path / "kb.json"),
            max_transitions=5,
        )
        for i in range(10):
            kb.record_transition(1, 2)
        assert len(kb._transitions) == 5

    def test_find_similar_transitions_empty(self, kb):
        matches = kb.find_similar_transitions({"vix": 30.0})
        assert matches == []

    def test_find_similar_transitions(self, kb):
        kb.record_transition(
            1, 2, confidence=0.8,
            leading_indicators={"vix": 30.0, "spread": 0.05},
        )
        kb.record_transition(
            1, 3, confidence=0.6,
            leading_indicators={"vix": 35.0, "spread": 0.1},
        )
        matches = kb.find_similar_transitions({"vix": 31.0, "spread": 0.06})
        assert len(matches) > 0
        assert matches[0].similarity > 0.0

    def test_find_similar_transitions_no_indicators(self, kb):
        kb.record_transition(1, 2, confidence=0.8)
        matches = kb.find_similar_transitions({"vix": 30.0})
        assert len(matches) == 0

    def test_find_similar_transitions_min_similarity(self, kb):
        kb.record_transition(
            1, 2, leading_indicators={"vix": 30.0, "spread": 0.05}
        )
        matches = kb.find_similar_transitions(
            {"vix": -100.0, "spread": -100.0}, min_similarity=0.99
        )
        assert len(matches) == 0

    def test_update_transition_outcome(self, kb):
        r = kb.record_transition(1, 2, confidence=0.8)
        result = kb.update_transition_outcome(
            r.transition_id, {"SPX": 3.2, "TLT": -1.5}
        )
        assert result is True
        t = kb.get_transitions()[0]
        assert t.post_transition_performance["SPX"] == 3.2

    def test_update_nonexistent_transition(self, kb):
        assert kb.update_transition_outcome("nonexistent", {"SPX": 1.0}) is False

    def test_record_anomaly(self, kb):
        a = kb.record_anomaly("correlation_spike", "SPX_TLT", 2, z_score=3.5)
        assert a.anomaly_type == "correlation_spike"
        assert len(kb._anomalies) == 1

    def test_get_anomalies_filtered(self, kb):
        kb.record_anomaly("type_a", "SPX_TLT", 1, z_score=2.0)
        kb.record_anomaly("type_b", "GLD_DXY", 2, z_score=3.0)
        kb.record_anomaly("type_a", "SPX_GLD", 1, z_score=1.5)
        results = kb.get_anomalies(anomaly_type="type_a")
        assert len(results) == 2

    def test_get_anomalies_by_pair(self, kb):
        kb.record_anomaly("type_a", "SPX_TLT", 1, z_score=2.0)
        kb.record_anomaly("type_b", "SPX_TLT", 2, z_score=3.0)
        results = kb.get_anomalies(asset_pair="SPX_TLT")
        assert len(results) == 2

    def test_get_anomalies_by_regime(self, kb):
        kb.record_anomaly("type_a", "SPX_TLT", 1, z_score=2.0)
        kb.record_anomaly("type_b", "GLD_DXY", 2, z_score=3.0)
        results = kb.get_anomalies(regime=2)
        assert len(results) == 1

    def test_get_anomaly_stats(self, kb):
        kb.record_anomaly("type_a", "SPX_TLT", 1, z_score=2.0)
        kb.record_anomaly("type_a", "GLD_DXY", 2, z_score=4.0)
        stats = kb.get_anomaly_stats("type_a")
        assert stats["total"] == 2
        assert stats["avg_z_score"] == 3.0

    def test_get_anomaly_stats_empty(self, kb):
        stats = kb.get_anomaly_stats()
        assert stats["total"] == 0

    def test_update_anomaly_outcome(self, kb):
        a = kb.record_anomaly("test", "SPX_TLT", 1, z_score=2.0)
        result = kb.update_anomaly_outcome(a.anomaly_id, reversion_days=5.0, outcome="Reverted")
        assert result is True
        updated = kb.get_anomalies()[0]
        assert updated.reversion_days == 5.0
        assert updated.outcome == "Reverted"

    def test_update_nonexistent_anomaly(self, kb):
        assert kb.update_anomaly_outcome("fake", reversion_days=1.0) is False

    def test_max_anomalies_cap(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(
            storage_path=str(tmp_path / "kb.json"),
            max_anomalies=3,
        )
        for i in range(5):
            kb.record_anomaly("t", "p", 1, z_score=float(i))
        assert len(kb._anomalies) == 3

    def test_record_macro_impact(self, kb):
        kb.record_macro_impact("NFP", 1, 0.3)
        kb.record_macro_impact("NFP", 2, -0.8)
        kb.record_macro_impact("CPI", 3, -1.2)
        stats = kb.get_macro_impact_stats()
        assert "NFP" in stats
        assert "CPI" in stats

    def test_macro_impact_stats_filtered(self, kb):
        kb.record_macro_impact("NFP", 1, 0.3)
        kb.record_macro_impact("CPI", 2, -0.5)
        stats = kb.get_macro_impact_stats("NFP")
        assert "NFP" in stats
        assert "CPI" not in stats

    def test_macro_impact_avg(self, kb):
        kb.record_macro_impact("NFP", 1, 0.2)
        kb.record_macro_impact("NFP", 1, 0.4)
        stats = kb.get_macro_impact_stats("NFP")
        assert stats["NFP"][1]["avg_impact"] == 0.3
        assert stats["NFP"][1]["count"] == 2

    def test_get_summary(self, kb):
        kb.record_transition(1, 2, confidence=0.8)
        kb.record_transition(2, 3, confidence=0.7)
        kb.record_anomaly("test", "SPX_TLT", 1, z_score=2.0)
        kb.record_macro_impact("NFP", 1, 0.3)
        summary = kb.get_summary()
        assert summary["total_transitions"] == 2
        assert summary["unique_transition_types"] == 2
        assert summary["total_anomalies"] == 1
        assert summary["macro_indicators_tracked"] == 1

    def test_persistence_save_load(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase

        path = str(tmp_path / "persist.json")

        # Save
        kb1 = KnowledgeBase(storage_path=path)
        kb1.record_transition(1, 2, confidence=0.8,
                              leading_indicators={"vix": 30.0})
        kb1.record_anomaly("test", "SPX_TLT", 1, z_score=2.5)
        kb1.record_macro_impact("NFP", 1, 0.3)

        # Load in new instance
        kb2 = KnowledgeBase(storage_path=path)
        assert len(kb2._transitions) == 1
        assert kb2._transitions[0].confidence == 0.8
        assert len(kb2._anomalies) == 1
        assert "NFP" in kb2._macro_impacts

    def test_cosine_similarity(self, kb):
        sim = kb._cosine_similarity(
            {"a": 1.0, "b": 0.0},
            {"a": 1.0, "b": 0.0},
        )
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self, kb):
        sim = kb._cosine_similarity(
            {"a": 1.0, "b": 0.0},
            {"a": 0.0, "b": 1.0},
        )
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_no_overlap(self, kb):
        sim = kb._cosine_similarity({"a": 1.0}, {"b": 1.0})
        assert sim == 0.0

    def test_cosine_similarity_empty(self, kb):
        assert kb._cosine_similarity({}, {"a": 1.0}) == 0.0

    def test_load_nonexistent_file(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(storage_path=str(tmp_path / "no_file.json"))
        assert len(kb._transitions) == 0


# ═══════════════════════════════════════════════════════════════════
#  Alternative Data Integration Tests (5.3)
# ═══════════════════════════════════════════════════════════════════


class TestAltDataSignal:
    """AltDataSignal dataclass tests."""

    def test_creation(self):
        from src.knowledge.alt_data import AltDataSignal

        s = AltDataSignal(
            name="move_index", value=130.0, z_score=1.33,
            regime_signal="elevated", regime_weight=0.14,
        )
        assert s.name == "move_index"
        assert s.value == 130.0

    def test_to_dict(self):
        from src.knowledge.alt_data import AltDataSignal

        s = AltDataSignal(name="test", value=100.0, z_score=0.5)
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["value"] == 100.0
        assert "timestamp" in d


class TestAltSignalType:
    """AltSignalType enum tests."""

    def test_all_types_exist(self):
        from src.knowledge.alt_data import AltSignalType

        expected = [
            "fed_funds_futures", "move_index", "skew_index",
            "cdx_index", "copper_gold_ratio", "hy_spreads",
            "tips_breakeven",
        ]
        for name in expected:
            assert AltSignalType(name) is not None

    def test_count(self):
        from src.knowledge.alt_data import AltSignalType
        assert len(AltSignalType) == 7


class TestAltDataIntegrator:
    """AltDataIntegrator core tests."""

    @pytest.fixture
    def adi(self):
        from src.knowledge.alt_data import AltDataIntegrator
        return AltDataIntegrator()

    def test_initialization(self, adi):
        status = adi.get_status()
        assert status["total_configured"] == 7
        assert status["total_available"] == 0

    def test_set_signal_value(self, adi):
        adi.set_signal_value("move_index", 130.0)
        assert adi._last_update is not None
        status = adi.get_status()
        assert "move_index" in status["available_signals"]

    def test_set_invalid_signal(self, adi):
        adi.set_signal_value("nonexistent_signal", 100.0)
        assert len(adi._current_values) == 0

    def test_set_all_values(self, adi):
        adi.set_all_values({
            "move_index": 130.0,
            "skew_index": 140.0,
            "hy_spreads": 500.0,
        })
        status = adi.get_status()
        assert status["total_available"] == 3

    def test_get_signal_risk_on(self, adi):
        adi.set_signal_value("move_index", 130.0)
        signal = adi.get_signal("move_index", regime=1)
        assert signal is not None
        assert signal.name == "move_index"
        assert signal.value == 130.0
        assert signal.z_score == pytest.approx((130.0 - 90.0) / 30.0, rel=0.01)
        assert signal.regime_signal in ["bond_stress", "elevated", "calm"]

    def test_get_signal_risk_off(self, adi):
        adi.set_signal_value("hy_spreads", 700.0)
        signal = adi.get_signal("hy_spreads", regime=2)
        assert signal is not None
        assert signal.regime_signal in ["blowout", "stress", "elevated"]

    def test_get_signal_stagflation(self, adi):
        adi.set_signal_value("tips_breakeven", 3.5)
        signal = adi.get_signal("tips_breakeven", regime=3)
        assert signal is not None
        assert signal.regime_signal in ["inflation_embedded", "rising", "mixed"]

    def test_get_signal_disinflationary(self, adi):
        adi.set_signal_value("copper_gold_ratio", 0.007)
        signal = adi.get_signal("copper_gold_ratio", regime=4)
        assert signal is not None
        assert signal.regime_signal in ["disinflation_signal", "mixed"]

    def test_get_signal_no_data(self, adi):
        signal = adi.get_signal("move_index", regime=1)
        assert signal is None

    def test_get_signal_invalid_name(self, adi):
        signal = adi.get_signal("fake_signal", regime=1)
        assert signal is None

    def test_get_all_signals(self, adi):
        adi.set_all_values({
            "move_index": 100.0,
            "skew_index": 130.0,
        })
        signals = adi.get_all_signals(regime=1)
        assert len(signals) == 2
        names = {s.name for s in signals}
        assert "move_index" in names
        assert "skew_index" in names

    def test_get_all_signals_empty(self, adi):
        signals = adi.get_all_signals(regime=1)
        assert len(signals) == 0

    def test_get_regime_vote_empty(self, adi):
        vote = adi.get_regime_vote(regime=1)
        assert vote["vote"] == "abstain"
        assert vote["signal_count"] == 0

    def test_get_regime_vote_with_signals(self, adi):
        adi.set_all_values({
            "move_index": 90.0,
            "skew_index": 125.0,
            "hy_spreads": 400.0,
            "tips_breakeven": 2.2,
            "copper_gold_ratio": 0.0055,
        })
        vote = adi.get_regime_vote(regime=1)
        assert vote["signal_count"] == 5
        assert vote["total_weight"] > 0
        assert vote["vote"] in ["confirm", "contradict", "neutral"]
        assert len(vote["details"]) == 5

    def test_get_status(self, adi):
        adi.set_signal_value("move_index", 100.0)
        status = adi.get_status()
        assert "move_index" in status["available_signals"]
        assert "skew_index" in status["missing_signals"]
        assert status["total_configured"] == 7
        assert status["total_available"] == 1

    def test_get_signal_types(self, adi):
        types = adi.get_signal_types()
        assert len(types) == 7
        assert "move_index" in types

    def test_custom_baselines(self):
        from src.knowledge.alt_data import AltDataIntegrator
        adi = AltDataIntegrator(
            baselines={"move_index": (100.0, 20.0, "custom")}
        )
        adi.set_signal_value("move_index", 140.0)
        sig = adi.get_signal("move_index", regime=1)
        assert sig.z_score == pytest.approx(2.0, rel=0.01)

    def test_all_regime_interpretations(self, adi):
        """Ensure all signal×regime combos produce valid output."""
        from src.knowledge.alt_data import AltSignalType
        adi.set_all_values({
            "fed_funds_futures": 3.0,
            "move_index": 120.0,
            "skew_index": 140.0,
            "cdx_index": 100.0,
            "copper_gold_ratio": 0.004,
            "hy_spreads": 600.0,
            "tips_breakeven": 3.0,
        })
        for regime in [1, 2, 3, 4]:
            signals = adi.get_all_signals(regime=regime)
            assert len(signals) == 7
            for s in signals:
                assert isinstance(s.regime_signal, str)
                assert s.regime_weight > 0
                assert 0 <= s.confidence <= 1


# ═══════════════════════════════════════════════════════════════════
#  Research Publisher Tests (5.4)
# ═══════════════════════════════════════════════════════════════════


class TestResearchReport:
    """ResearchReport dataclass tests."""

    def test_creation(self):
        from src.knowledge.research_publisher import ResearchReport, ReportSection

        r = ResearchReport(
            report_id="test123",
            title="Test Report",
            report_type="case_study",
            sections=[ReportSection(heading="Intro", content="Hello")],
            summary="A test",
        )
        assert r.title == "Test Report"
        assert len(r.sections) == 1

    def test_to_dict(self):
        from src.knowledge.research_publisher import ResearchReport

        r = ResearchReport(
            report_id="id1", title="Title", report_type="backtest"
        )
        d = r.to_dict()
        assert d["report_id"] == "id1"
        assert d["title"] == "Title"

    def test_to_markdown(self):
        from src.knowledge.research_publisher import ResearchReport, ReportSection

        r = ResearchReport(
            report_id="md1",
            title="MD Test",
            report_type="test",
            sections=[
                ReportSection(heading="S1", content="Content here"),
                ReportSection(heading="S2", content="More content", data={"key": 1}),
            ],
            summary="Summary text",
        )
        md = r.to_markdown()
        assert "# MD Test" in md
        assert "## Executive Summary" in md
        assert "## S1" in md
        assert "Content here" in md
        assert '"key": 1' in md


class TestReportSection:
    """ReportSection dataclass tests."""

    def test_creation(self):
        from src.knowledge.research_publisher import ReportSection

        s = ReportSection(heading="Test", content="Body")
        assert s.heading == "Test"
        assert s.data == {}

    def test_to_dict(self):
        from src.knowledge.research_publisher import ReportSection

        s = ReportSection(heading="H", content="C", data={"x": 1})
        d = s.to_dict()
        assert d["heading"] == "H"
        assert d["data"]["x"] == 1


class TestResearchPublisher:
    """ResearchPublisher core tests."""

    @pytest.fixture
    def kb(self, tmp_path):
        from src.knowledge.knowledge_base import KnowledgeBase
        return KnowledgeBase(storage_path=str(tmp_path / "kb.json"))

    @pytest.fixture
    def publisher(self, kb, tmp_path):
        from src.knowledge.research_publisher import ResearchPublisher
        return ResearchPublisher(
            knowledge_base=kb,
            output_dir=str(tmp_path / "reports"),
        )

    def test_initialization(self, publisher):
        summary = publisher.get_summary()
        assert summary["total_reports"] == 0

    def test_generate_case_study_empty(self, publisher):
        report = publisher.generate_transition_case_study(from_regime=1, to_regime=2)
        assert report.report_type == "transition_case_study"
        assert "Risk-On Growth" in report.title
        assert "Risk-Off Crisis" in report.title
        assert len(report.sections) == 5

    def test_generate_case_study_with_data(self, publisher, kb):
        kb.record_transition(1, 2, confidence=0.8,
                             leading_indicators={"vix": 30.0},
                             classifier_accuracy={"hmm": True, "ml": False},
                             detection_latency_days=1.5)
        kb.record_transition(1, 2, confidence=0.7,
                             leading_indicators={"vix": 35.0},
                             classifier_accuracy={"hmm": True, "ml": True},
                             detection_latency_days=2.0)
        report = publisher.generate_transition_case_study(from_regime=1, to_regime=2)
        assert report.metadata["sample_size"] == 2
        assert len(report.sections) >= 5

    def test_generate_case_study_any_regime(self, publisher):
        report = publisher.generate_transition_case_study()
        assert "Any" in report.title

    def test_generate_backtest_report(self, publisher):
        results = {
            "periods": 100,
            "total_return": 15.5,
            "accuracy": 0.72,
            "regime_alpha": 0.08,
            "market_beta": 0.05,
        }
        report = publisher.generate_backtest_report(backtest_results=results)
        assert report.report_type == "backtest_report"
        assert "72.0%" in report.summary
        assert len(report.sections) == 3

    def test_generate_backtest_report_empty(self, publisher):
        report = publisher.generate_backtest_report()
        assert report.report_type == "backtest_report"

    def test_generate_factor_analysis(self, publisher):
        data = {
            "exposures": {"momentum": 0.3, "value": -0.1},
            "returns_by_regime": {"1": 0.05, "2": -0.02},
            "disagreement_analysis": {"leading": True},
        }
        report = publisher.generate_factor_analysis(factor_data=data)
        assert report.report_type == "factor_analysis"
        assert len(report.sections) == 3

    def test_generate_factor_analysis_empty(self, publisher):
        report = publisher.generate_factor_analysis()
        assert report.report_type == "factor_analysis"

    def test_get_reports(self, publisher):
        publisher.generate_backtest_report()
        publisher.generate_factor_analysis()
        reports = publisher.get_reports()
        assert len(reports) == 2

    def test_get_reports_filtered(self, publisher):
        publisher.generate_backtest_report()
        publisher.generate_factor_analysis()
        reports = publisher.get_reports(report_type="backtest_report")
        assert len(reports) == 1
        assert reports[0].report_type == "backtest_report"

    def test_save_report(self, publisher, tmp_path):
        report = publisher.generate_backtest_report()
        path = publisher.save_report(report)
        assert os.path.exists(path)
        content = Path(path).read_text()
        assert "AMRCAIS Backtest Report" in content

    def test_get_summary(self, publisher):
        publisher.generate_backtest_report()
        publisher.generate_case_study_report_type = True  # coverage
        summary = publisher.get_summary()
        assert summary["total_reports"] == 1
        assert "backtest_report" in summary["by_type"]


# ═══════════════════════════════════════════════════════════════════
#  User Manager Tests (5.2)
# ═══════════════════════════════════════════════════════════════════


class TestUserRole:
    """UserRole enum tests."""

    def test_all_roles_exist(self):
        from src.knowledge.user_manager import UserRole

        assert UserRole.RESEARCHER == "researcher"
        assert UserRole.PM == "pm"
        assert UserRole.RISK_MANAGER == "risk_manager"
        assert UserRole.CIO == "cio"

    def test_count(self):
        from src.knowledge.user_manager import UserRole
        assert len(UserRole) == 4


class TestUser:
    """User dataclass tests."""

    def test_creation(self):
        from src.knowledge.user_manager import User, UserRole

        u = User(name="Alice", email="alice@fund.com", role=UserRole.RESEARCHER)
        assert u.name == "Alice"
        assert u.role == UserRole.RESEARCHER
        assert u.is_active is True

    def test_to_dict(self):
        from src.knowledge.user_manager import User, UserRole

        u = User(name="Bob", email="bob@fund.com", role=UserRole.PM)
        d = u.to_dict()
        assert d["name"] == "Bob"
        assert d["role"] == "pm"
        assert "api_key_hash" not in d

    def test_to_dict_sensitive(self):
        from src.knowledge.user_manager import User, UserRole

        u = User(name="Eve", email="eve@fund.com", api_key_hash="secret123")
        d = u.to_dict(include_sensitive=True)
        assert d["api_key_hash"] == "secret123"

    def test_from_dict(self):
        from src.knowledge.user_manager import User

        data = {"name": "Test", "email": "test@test.com", "role": "cio"}
        u = User.from_dict(data)
        assert u.name == "Test"
        assert u.role.value == "cio"


class TestUserManager:
    """UserManager core tests."""

    @pytest.fixture
    def mgr(self, tmp_path):
        from src.knowledge.user_manager import UserManager
        return UserManager(storage_path=str(tmp_path / "users.json"))

    def test_initialization(self, mgr):
        summary = mgr.get_summary()
        assert summary["total_users"] == 0

    def test_create_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, api_key = mgr.create_user("Alice", "alice@fund.com", UserRole.RESEARCHER)
        assert user.name == "Alice"
        assert user.role == UserRole.RESEARCHER
        assert len(api_key) > 0
        assert "write_regime_definition" in user.permissions

    def test_create_user_duplicate_email(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("Alice", "alice@fund.com", UserRole.RESEARCHER)
        with pytest.raises(ValueError, match="already registered"):
            mgr.create_user("Alice2", "alice@fund.com", UserRole.PM)

    def test_get_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("Bob", "bob@fund.com", UserRole.PM)
        found = mgr.get_user(user.user_id)
        assert found is not None
        assert found.name == "Bob"

    def test_get_user_not_found(self, mgr):
        assert mgr.get_user("nonexistent") is None

    def test_get_user_by_email(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("Carol", "carol@fund.com", UserRole.CIO)
        found = mgr.get_user_by_email("carol@fund.com")
        assert found is not None
        assert found.name == "Carol"

    def test_get_user_by_email_not_found(self, mgr):
        assert mgr.get_user_by_email("nobody@nowhere.com") is None

    def test_list_users(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("A", "a@t.com", UserRole.RESEARCHER)
        mgr.create_user("B", "b@t.com", UserRole.PM)
        mgr.create_user("C", "c@t.com", UserRole.CIO)
        users = mgr.list_users()
        assert len(users) == 3

    def test_list_users_by_role(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("A", "a@t.com", UserRole.RESEARCHER)
        mgr.create_user("B", "b@t.com", UserRole.PM)
        mgr.create_user("C", "c@t.com", UserRole.PM)
        users = mgr.list_users(role=UserRole.PM)
        assert len(users) == 2

    def test_list_users_active_only(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("D", "d@t.com", UserRole.PM)
        mgr.update_user(user.user_id, is_active=False)
        users = mgr.list_users(active_only=True)
        assert len(users) == 0
        users = mgr.list_users(active_only=False)
        assert len(users) == 1

    def test_update_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("E", "e@t.com", UserRole.PM)
        updated = mgr.update_user(user.user_id, name="Eunice", role=UserRole.CIO)
        assert updated.name == "Eunice"
        assert updated.role == UserRole.CIO
        assert "admin_dashboard" in updated.permissions

    def test_update_user_not_found(self, mgr):
        assert mgr.update_user("ghost") is None

    def test_update_user_preferences(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("F", "f@t.com", UserRole.PM)
        mgr.update_user(user.user_id, preferences={"theme": "dark"})
        found = mgr.get_user(user.user_id)
        assert found.preferences["theme"] == "dark"

    def test_delete_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("G", "g@t.com", UserRole.PM)
        assert mgr.delete_user(user.user_id) is True
        assert mgr.get_user(user.user_id) is None

    def test_delete_user_not_found(self, mgr):
        assert mgr.delete_user("ghost") is False

    def test_authenticate_success(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, api_key = mgr.create_user("H", "h@t.com", UserRole.RESEARCHER)
        authed = mgr.authenticate(api_key)
        assert authed is not None
        assert authed.user_id == user.user_id
        assert authed.last_login is not None

    def test_authenticate_wrong_key(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("I", "i@t.com", UserRole.PM)
        assert mgr.authenticate("wrong_key") is None

    def test_authenticate_inactive_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, api_key = mgr.create_user("J", "j@t.com", UserRole.PM)
        mgr.update_user(user.user_id, is_active=False)
        assert mgr.authenticate(api_key) is None

    def test_authorize(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("K", "k@t.com", UserRole.RESEARCHER)
        assert mgr.authorize(user.user_id, "write_regime_definition") is True
        assert mgr.authorize(user.user_id, "admin_dashboard") is False

    def test_authorize_inactive(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("L", "l@t.com", UserRole.PM)
        mgr.update_user(user.user_id, is_active=False)
        assert mgr.authorize(user.user_id, "read_regime") is False

    def test_authorize_nonexistent(self, mgr):
        assert mgr.authorize("ghost", "read_regime") is False

    def test_get_permissions(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("M", "m@t.com", UserRole.RISK_MANAGER)
        perms = mgr.get_permissions(user.user_id)
        assert "read_risk" in perms
        assert "read_var" in perms
        assert "write_alerts" in perms

    def test_get_permissions_nonexistent(self, mgr):
        assert mgr.get_permissions("ghost") == set()

    def test_grant_permission(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("N", "n@t.com", UserRole.PM)
        assert mgr.grant_permission(user.user_id, "custom_perm") is True
        perms = mgr.get_permissions(user.user_id)
        assert "custom_perm" in perms

    def test_grant_permission_nonexistent(self, mgr):
        assert mgr.grant_permission("ghost", "perm") is False

    def test_revoke_permission(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("O", "o@t.com", UserRole.PM)
        mgr.revoke_permission(user.user_id, "read_regime")
        perms = mgr.get_permissions(user.user_id)
        assert "read_regime" not in perms

    def test_revoke_permission_nonexistent(self, mgr):
        assert mgr.revoke_permission("ghost", "perm") is False

    def test_add_annotation(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("P", "p@t.com", UserRole.RESEARCHER)
        note = mgr.add_annotation(
            user.user_id, "I think this is a false signal",
            context={"regime": 2, "date": "2024-01-15"},
        )
        assert note is not None
        assert note["content"] == "I think this is a false signal"
        assert note["author"] == "P"

    def test_add_annotation_nonexistent_user(self, mgr):
        assert mgr.add_annotation("ghost", "test") is None

    def test_get_annotations_all(self, mgr):
        from src.knowledge.user_manager import UserRole

        u1, _ = mgr.create_user("Q", "q@t.com", UserRole.RESEARCHER)
        u2, _ = mgr.create_user("R", "r@t.com", UserRole.PM)
        mgr.add_annotation(u1.user_id, "Note 1")
        mgr.add_annotation(u2.user_id, "Note 2")
        mgr.add_annotation(u1.user_id, "Note 3")
        notes = mgr.get_annotations()
        assert len(notes) == 3

    def test_get_annotations_by_user(self, mgr):
        from src.knowledge.user_manager import UserRole

        u1, _ = mgr.create_user("S", "s@t.com", UserRole.RESEARCHER)
        u2, _ = mgr.create_user("T", "t@t.com", UserRole.PM)
        mgr.add_annotation(u1.user_id, "Note from S")
        mgr.add_annotation(u2.user_id, "Note from T")
        notes = mgr.get_annotations(user_id=u1.user_id)
        assert len(notes) == 1
        assert notes[0]["author"] == "S"

    def test_get_summary(self, mgr):
        from src.knowledge.user_manager import UserRole

        mgr.create_user("U", "u@t.com", UserRole.RESEARCHER)
        mgr.create_user("V", "v@t.com", UserRole.PM)
        summary = mgr.get_summary()
        assert summary["total_users"] == 2
        assert summary["active_users"] == 2
        assert len(summary["roles_available"]) == 4

    def test_persistence(self, tmp_path):
        from src.knowledge.user_manager import UserManager, UserRole

        path = str(tmp_path / "persist_users.json")

        mgr1 = UserManager(storage_path=path)
        user, api_key = mgr1.create_user("W", "w@t.com", UserRole.CIO)
        mgr1.add_annotation(user.user_id, "Persisted note")

        mgr2 = UserManager(storage_path=path)
        assert len(mgr2._users) == 1
        u = mgr2.get_user(user.user_id)
        assert u.name == "W"
        assert len(u.annotations) == 1

    def test_role_permissions_researcher(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("Res", "res@t.com", UserRole.RESEARCHER)
        perms = user.permissions
        assert "write_regime_definition" in perms
        assert "write_research" in perms
        assert "read_knowledge" in perms

    def test_role_permissions_pm(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("PM", "pm@t.com", UserRole.PM)
        perms = user.permissions
        assert "read_allocations" in perms
        assert "write_regime_definition" not in perms

    def test_role_permissions_risk_manager(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("Risk", "risk@t.com", UserRole.RISK_MANAGER)
        perms = user.permissions
        assert "write_alerts" in perms
        assert "read_var" in perms

    def test_role_permissions_cio(self, mgr):
        from src.knowledge.user_manager import UserRole

        user, _ = mgr.create_user("CIO", "cio@t.com", UserRole.CIO)
        perms = user.permissions
        assert "admin_dashboard" in perms
        assert "read_narrative" in perms
        assert "read_users" in perms


# ═══════════════════════════════════════════════════════════════════
#  Phase 5 API Endpoint Tests
# ═══════════════════════════════════════════════════════════════════


class TestPhase5StatusAPI:
    """Phase 5 status endpoint tests."""

    def test_status_returns_200(self, phase5_client):
        resp = phase5_client.get("/api/phase5/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge_base" in data
        assert "alt_data" in data
        assert "research_publisher" in data
        assert "user_manager" in data


class TestKnowledgeBaseAPI:
    """Knowledge base API endpoint tests."""

    def test_knowledge_summary(self, phase5_client):
        resp = phase5_client.get("/api/phase5/knowledge/summary")
        assert resp.status_code == 200

    def test_get_transitions(self, phase5_client):
        resp = phase5_client.get("/api/phase5/transitions")
        assert resp.status_code == 200
        data = resp.json()
        assert "transitions" in data
        assert "total" in data

    def test_record_transition(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/transitions",
            json={
                "from_regime": 1, "to_regime": 2,
                "confidence": 0.8, "disagreement": 0.3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["from_regime"] == 1
        assert data["to_regime"] == 2

    def test_search_transitions(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/transitions/search",
            json={
                "current_indicators": {"vix": 30.0, "spread": 0.05},
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "matches" in data

    def test_get_anomalies(self, phase5_client):
        resp = phase5_client.get("/api/phase5/anomalies")
        assert resp.status_code == 200

    def test_record_anomaly(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/anomalies",
            json={
                "anomaly_type": "test", "asset_pair": "SPX_TLT",
                "regime": 1, "z_score": 2.5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["anomaly_type"] == "test"

    def test_anomaly_stats(self, phase5_client):
        resp = phase5_client.get("/api/phase5/anomalies/stats")
        assert resp.status_code == 200

    def test_record_macro_impact(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/macro-impact",
            json={"indicator": "NFP", "regime": 1, "impact_pct": 0.3},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_macro_impact_stats(self, phase5_client):
        resp = phase5_client.get("/api/phase5/macro-impact/stats")
        assert resp.status_code == 200


class TestAltDataAPI:
    """Alternative data API endpoint tests."""

    def test_alt_data_status(self, phase5_client):
        resp = phase5_client.get("/api/phase5/alt-data/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_configured" in data

    def test_alt_data_types(self, phase5_client):
        resp = phase5_client.get("/api/phase5/alt-data/types")
        assert resp.status_code == 200
        data = resp.json()
        assert "signal_types" in data
        assert len(data["signal_types"]) == 7

    def test_set_alt_data_values(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/alt-data/values",
            json={"values": {"move_index": 100.0, "skew_index": 130.0}},
        )
        assert resp.status_code == 200
        assert resp.json()["signals_set"] == 2

    def test_get_alt_data_signals(self, phase5_client):
        # Set values first
        phase5_client.post(
            "/api/phase5/alt-data/values",
            json={"values": {"move_index": 100.0}},
        )
        resp = phase5_client.get("/api/phase5/alt-data/signals?regime=1")
        assert resp.status_code == 200

    def test_alt_data_vote(self, phase5_client):
        resp = phase5_client.get("/api/phase5/alt-data/vote?regime=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "vote" in data


class TestResearchAPI:
    """Research publisher API endpoint tests."""

    def test_research_summary(self, phase5_client):
        resp = phase5_client.get("/api/phase5/research/summary")
        assert resp.status_code == 200

    def test_list_reports(self, phase5_client):
        resp = phase5_client.get("/api/phase5/research/reports")
        assert resp.status_code == 200
        data = resp.json()
        assert "reports" in data

    def test_generate_case_study(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/research/case-study",
            json={"from_regime": 1, "to_regime": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "transition_case_study"

    def test_generate_backtest_report(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/research/backtest",
            json={
                "backtest_results": {
                    "periods": 100,
                    "accuracy": 0.72,
                    "regime_alpha": 0.08,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "backtest_report"

    def test_generate_factor_analysis(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/research/factors",
            json={
                "factor_data": {
                    "exposures": {"momentum": 0.3},
                    "returns_by_regime": {},
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "factor_analysis"


class TestUserManagerAPI:
    """User manager API endpoint tests."""

    def test_user_manager_summary(self, phase5_client):
        resp = phase5_client.get("/api/phase5/users/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_users" in data
        assert "roles_available" in data

    def test_create_user(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/users",
            json={"name": "TestUser", "email": "test@test.com", "role": "researcher"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "TestUser"
        assert data["role"] == "researcher"

    def test_create_user_invalid_role(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/users",
            json={"name": "Bad", "email": "bad@t.com", "role": "superadmin"},
        )
        assert resp.status_code == 400

    def test_list_users(self, phase5_client):
        resp = phase5_client.get("/api/phase5/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "users" in data

    def test_list_users_by_role(self, phase5_client):
        resp = phase5_client.get("/api/phase5/users?role=pm")
        assert resp.status_code == 200

    def test_list_users_invalid_role(self, phase5_client):
        resp = phase5_client.get("/api/phase5/users?role=invalid")
        assert resp.status_code == 400

    def test_get_user_not_found(self, phase5_client):
        resp = phase5_client.get("/api/phase5/users/nonexistent")
        assert resp.status_code == 404

    def test_update_user_not_found(self, phase5_client):
        resp = phase5_client.put(
            "/api/phase5/users/nonexistent",
            json={"name": "Updated"},
        )
        assert resp.status_code == 404

    def test_delete_user_not_found(self, phase5_client):
        resp = phase5_client.delete("/api/phase5/users/nonexistent")
        assert resp.status_code == 404

    def test_list_annotations(self, phase5_client):
        resp = phase5_client.get("/api/phase5/annotations")
        assert resp.status_code == 200
        data = resp.json()
        assert "annotations" in data

    def test_create_annotation_user_not_found(self, phase5_client):
        resp = phase5_client.post(
            "/api/phase5/annotations",
            json={
                "user_id": "ghost",
                "content": "A note",
            },
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════
#  Phase 5 Integration with AMRCAIS Main Tests
# ═══════════════════════════════════════════════════════════════════


class TestPhase5Integration:
    """Test Phase 5 components integrate with AMRCAIS main."""

    def test_knowledge_base_import(self):
        from src.knowledge.knowledge_base import (
            KnowledgeBase,
            RegimeTransitionRecord,
            AnomalyRecord,
            PatternMatch,
        )
        assert KnowledgeBase is not None
        assert RegimeTransitionRecord is not None

    def test_alt_data_import(self):
        from src.knowledge.alt_data import AltDataIntegrator, AltDataSignal
        assert AltDataIntegrator is not None

    def test_research_import(self):
        from src.knowledge.research_publisher import ResearchPublisher, ResearchReport
        assert ResearchPublisher is not None

    def test_user_manager_import(self):
        from src.knowledge.user_manager import UserManager, User, UserRole
        assert UserManager is not None
        assert len(UserRole) == 4

    def test_package_init_imports(self):
        from src.knowledge import (
            KnowledgeBase,
            AltDataIntegrator,
            ResearchPublisher,
            UserManager,
            UserRole,
        )
        assert KnowledgeBase is not None
        assert AltDataIntegrator is not None
        assert ResearchPublisher is not None
        assert UserManager is not None
        assert UserRole is not None

    def test_amrcais_has_phase5_attributes(self):
        """Verify AMRCAIS class has Phase 5 attribute declarations."""
        from src.main import AMRCAIS
        system = AMRCAIS.__new__(AMRCAIS)
        system.__init__()
        assert hasattr(system, "knowledge_base")
        assert hasattr(system, "alt_data_integrator")
        assert hasattr(system, "research_publisher")
        assert hasattr(system, "user_manager")
