"""
Tests for Phase 4: Real-Time + Execution.

Covers:
- EventBus (event_bus.py): pub/sub, history, wildcard, async
- AnalysisScheduler (scheduler.py): lifecycle, market hours, events
- AlertEngine (alert_engine.py): alert generation, cooldown, config
- StreamManager (stream_manager.py): SSE formatting, client mgmt
- PaperTradingEngine (paper_trading.py): orders, P&L, attribution
- Phase 4 API endpoints (routes/phase4.py)
"""

import asyncio
import math
import os
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════
#  EventBus Tests
# ═══════════════════════════════════════════════════════════════════


class TestEventType:
    """EventType enum tests."""

    def test_all_types_exist(self):
        from src.realtime.event_bus import EventType

        expected = [
            "regime_change", "regime_update", "transition_warning",
            "correlation_anomaly", "vix_spike", "macro_event",
            "recalibration_needed", "analysis_complete", "data_updated",
            "rebalance_trigger", "alpha_signal", "order_filled",
            "system_error", "system_health",
        ]
        for name in expected:
            assert EventType(name) is not None

    def test_string_value(self):
        from src.realtime.event_bus import EventType
        assert EventType.REGIME_CHANGE == "regime_change"
        assert EventType.VIX_SPIKE.value == "vix_spike"


class TestEvent:
    """Event dataclass tests."""

    def test_creation(self):
        from datetime import datetime

        from src.realtime.event_bus import Event, EventType

        ev = Event(EventType.REGIME_CHANGE, {"new": 2}, source="test")
        assert ev.event_type == EventType.REGIME_CHANGE
        assert ev.data == {"new": 2}
        assert ev.source == "test"
        assert len(ev.event_id) == 12
        assert isinstance(ev.timestamp, datetime)

    def test_frozen(self):
        from src.realtime.event_bus import Event, EventType

        ev = Event(EventType.REGIME_CHANGE, {})
        with pytest.raises(AttributeError):
            ev.source = "other"

    def test_to_dict(self):
        from src.realtime.event_bus import Event, EventType

        ev = Event(EventType.VIX_SPIKE, {"level": 35.0}, source="s")
        d = ev.to_dict()
        assert d["event_type"] == "vix_spike"
        assert d["data"]["level"] == 35.0
        assert "event_id" in d
        assert "timestamp" in d


class TestEventBus:
    """EventBus pub/sub tests."""

    def test_subscribe_and_publish(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []

        bus.subscribe(EventType.REGIME_CHANGE, lambda e: received.append(e))
        bus.publish(Event(EventType.REGIME_CHANGE, {"r": 1}))

        assert len(received) == 1
        assert received[0].data["r"] == 1

    def test_no_cross_type(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []

        bus.subscribe(EventType.VIX_SPIKE, lambda e: received.append(e))
        bus.publish(Event(EventType.REGIME_CHANGE, {}))

        assert len(received) == 0

    def test_subscribe_all(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []

        bus.subscribe_all(lambda e: received.append(e))
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.publish(Event(EventType.VIX_SPIKE, {}))

        assert len(received) == 2

    def test_unsubscribe(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)

        bus.subscribe(EventType.REGIME_CHANGE, handler)
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.unsubscribe(EventType.REGIME_CHANGE, handler)
        bus.publish(Event(EventType.REGIME_CHANGE, {}))

        assert len(received) == 1

    def test_unsubscribe_all(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)

        bus.subscribe_all(handler)
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.unsubscribe_all(handler)
        bus.publish(Event(EventType.REGIME_CHANGE, {}))

        assert len(received) == 1

    def test_history(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.publish(Event(EventType.REGIME_CHANGE, {"n": 1}))
        bus.publish(Event(EventType.VIX_SPIKE, {"n": 2}))
        bus.publish(Event(EventType.REGIME_CHANGE, {"n": 3}))

        history = bus.get_history(EventType.REGIME_CHANGE)
        assert len(history) == 2

        all_history = bus.get_history()
        assert len(all_history) == 3

    def test_history_limit(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        for i in range(10):
            bus.publish(Event(EventType.REGIME_CHANGE, {"i": i}))

        history = bus.get_history(limit=3)
        assert len(history) == 3

    def test_max_history_trimming(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus(max_history=5)
        for i in range(10):
            bus.publish(Event(EventType.REGIME_CHANGE, {"i": i}))

        # event_count tracks total published, history is trimmed
        assert bus.event_count == 10
        assert len(bus.get_history(limit=100)) == 5

    def test_event_count_and_subscriber_count(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: None)
        bus.subscribe(EventType.VIX_SPIKE, lambda e: None)
        bus.subscribe_all(lambda e: None)

        assert bus.subscriber_count == 3

        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        assert bus.event_count == 1

    def test_publish_returns_handler_count(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: None)
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: None)
        bus.subscribe_all(lambda e: None)

        count = bus.publish(Event(EventType.REGIME_CHANGE, {}))
        assert count == 3

    def test_clear_history(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.clear_history()
        # clear_history clears the history list but event_count is cumulative
        assert len(bus.get_history(limit=100)) == 0
        assert bus.event_count == 1

    def test_reset(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: None)
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.reset()
        assert bus.event_count == 0
        assert bus.subscriber_count == 0

    def test_handler_exception_doesnt_crash(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: 1 / 0)
        # Should not raise
        bus.publish(Event(EventType.REGIME_CHANGE, {}))

    @pytest.mark.asyncio
    async def test_publish_async(self):
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe(EventType.REGIME_CHANGE, handler)
        count = await bus.publish_async(Event(EventType.REGIME_CHANGE, {"a": 1}))

        assert count == 1
        assert len(received) == 1


# ═══════════════════════════════════════════════════════════════════
#  AnalysisScheduler Tests
# ═══════════════════════════════════════════════════════════════════


class TestAnalysisScheduler:
    """AnalysisScheduler lifecycle and event publishing tests."""

    def _make_scheduler(self, market_hours=False):
        from src.realtime.event_bus import EventBus
        from src.realtime.scheduler import AnalysisScheduler

        bus = EventBus()
        system = MagicMock()
        system.analyze.return_value = {
            "regime": {
                "id": 1,
                "name": "Risk-On Growth",
                "confidence": 0.85,
                "disagreement": 0.15,
                "transition_warning": False,
            },
            "modules": {"correlations": {"signal": {"strength": 0.3}}},
        }
        scheduler = AnalysisScheduler(
            system, bus, interval_seconds=1, market_hours_only=market_hours
        )
        return scheduler, bus, system

    def test_initial_state(self):
        s, _, _ = self._make_scheduler()
        assert not s.is_running
        assert s.run_count == 0
        assert s.error_count == 0
        assert s.last_analysis is None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        s, _, _ = self._make_scheduler()
        await s.start()
        assert s.is_running
        await s.stop()
        assert not s.is_running

    @pytest.mark.asyncio
    async def test_double_start_ignored(self):
        s, _, _ = self._make_scheduler()
        await s.start()
        await s.start()  # Should not error
        assert s.is_running
        await s.stop()

    @pytest.mark.asyncio
    async def test_trigger_now(self):
        s, bus, system = self._make_scheduler()
        result = await s.trigger_now()
        assert result is not None
        assert s.run_count == 1
        assert s.last_run_time is not None
        assert result["regime"]["id"] == 1

    @pytest.mark.asyncio
    async def test_regime_change_event(self):
        from src.realtime.event_bus import EventType

        s, bus, system = self._make_scheduler()
        events = []
        bus.subscribe(EventType.REGIME_CHANGE, lambda e: events.append(e))

        # First run — sets baseline
        await s.trigger_now()
        assert len(events) == 0

        # Second run with different regime
        system.analyze.return_value = {
            "regime": {
                "id": 2,
                "name": "Risk-Off Crisis",
                "confidence": 0.9,
                "disagreement": 0.1,
                "transition_warning": False,
            },
            "modules": {},
        }
        await s.trigger_now()
        assert len(events) == 1
        assert events[0].data["new_regime"] == 2
        assert events[0].data["previous_regime"] == 1

    @pytest.mark.asyncio
    async def test_transition_warning_event(self):
        from src.realtime.event_bus import EventType

        s, bus, system = self._make_scheduler()
        system.analyze.return_value = {
            "regime": {
                "id": 1,
                "name": "Risk-On Growth",
                "confidence": 0.5,
                "disagreement": 0.75,
                "transition_warning": True,
            },
            "modules": {},
        }
        events = []
        bus.subscribe(EventType.TRANSITION_WARNING, lambda e: events.append(e))

        await s.trigger_now()
        assert len(events) == 1
        assert events[0].data["disagreement"] == 0.75

    @pytest.mark.asyncio
    async def test_analysis_complete_event(self):
        from src.realtime.event_bus import EventType

        s, bus, system = self._make_scheduler()
        events = []
        bus.subscribe(EventType.ANALYSIS_COMPLETE, lambda e: events.append(e))

        await s.trigger_now()
        assert len(events) == 1
        assert events[0].data["run_number"] == 1

    @pytest.mark.asyncio
    async def test_error_publishes_system_error(self):
        from src.realtime.event_bus import EventType

        s, bus, system = self._make_scheduler()
        system.analyze.side_effect = RuntimeError("Boom")
        errors = []
        bus.subscribe(EventType.SYSTEM_ERROR, lambda e: errors.append(e))

        await s.trigger_now()
        assert len(errors) == 1
        assert "Boom" in errors[0].data["error"]
        assert s.error_count == 1

    @pytest.mark.asyncio
    async def test_correlation_anomaly_event(self):
        from src.realtime.event_bus import EventType

        s, bus, system = self._make_scheduler()
        system.analyze.return_value = {
            "regime": {"id": 1, "name": "X", "confidence": 0.8, "disagreement": 0.1},
            "modules": {
                "correlations": {
                    "signal": {"strength": 0.7},
                    "details": {"pairs": 3},
                }
            },
        }
        events = []
        bus.subscribe(EventType.CORRELATION_ANOMALY, lambda e: events.append(e))

        await s.trigger_now()
        assert len(events) == 1
        assert events[0].data["strength"] == 0.7

    def test_get_status(self):
        s, _, _ = self._make_scheduler()
        status = s.get_status()
        assert "is_running" in status
        assert "run_count" in status
        assert "interval_seconds" in status

    def test_should_run_market_hours(self):
        s, _, _ = self._make_scheduler(market_hours=True)
        # Depends on current time — just ensure it returns bool
        result = s._should_run()
        assert isinstance(result, bool)

    def test_should_run_no_market_hours(self):
        s, _, _ = self._make_scheduler(market_hours=False)
        assert s._should_run() is True


# ═══════════════════════════════════════════════════════════════════
#  AlertEngine Tests
# ═══════════════════════════════════════════════════════════════════


class TestAlertTypes:
    """AlertType and AlertSeverity enum tests."""

    def test_all_alert_types(self):
        from src.realtime.alert_engine import AlertType

        expected = [
            "regime_change", "transition_warning", "correlation_anomaly",
            "recalibration", "macro_event", "vix_spike", "system_error",
        ]
        for name in expected:
            assert AlertType(name) is not None

    def test_all_severities(self):
        from src.realtime.alert_engine import AlertSeverity

        for s in ["critical", "high", "medium", "low", "info"]:
            assert AlertSeverity(s) is not None


class TestAlert:
    """Alert dataclass tests."""

    def test_creation(self):
        from src.realtime.alert_engine import Alert, AlertSeverity, AlertType

        alert = Alert(
            alert_type=AlertType.REGIME_CHANGE,
            severity=AlertSeverity.CRITICAL,
            title="Regime Changed",
            message="Risk-on → risk-off",
        )
        assert alert.alert_type == AlertType.REGIME_CHANGE
        assert alert.acknowledged is False
        assert len(alert.alert_id) == 12

    def test_to_dict(self):
        from src.realtime.alert_engine import Alert, AlertSeverity, AlertType

        alert = Alert(
            alert_type=AlertType.VIX_SPIKE,
            severity=AlertSeverity.HIGH,
            title="VIX 35",
            message="VIX spiked",
        )
        d = alert.to_dict()
        assert d["alert_type"] == "vix_spike"
        assert d["severity"] == "high"
        assert d["acknowledged"] is False


class TestAlertEngine:
    """AlertEngine event processing and fatigue management tests."""

    def _make_engine(self):
        from src.realtime.alert_engine import AlertEngine
        from src.realtime.event_bus import EventBus

        bus = EventBus()
        engine = AlertEngine(bus)
        engine.start()
        return engine, bus

    def test_start_stop(self):
        engine, bus = self._make_engine()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_regime_change_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.REGIME_CHANGE, {
            "previous_regime": 1,
            "new_regime": 2,
            "regime_name": "Risk-Off Crisis",
            "confidence": 0.9,
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "regime_change"
        assert "severity" in alerts[0]

    def test_transition_warning_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.TRANSITION_WARNING, {
            "disagreement": 0.8,
            "regime": 1,
            "confidence": 0.5,
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "transition_warning"

    def test_transition_below_threshold_skipped(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.TRANSITION_WARNING, {
            "disagreement": 0.3,  # Below default 0.6 threshold
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 0

    def test_vix_spike_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.VIX_SPIKE, {"level": 40.0}))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"

    def test_vix_below_threshold_skipped(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.VIX_SPIKE, {"level": 18.0}))

        alerts = engine.get_alerts()
        assert len(alerts) == 0

    def test_correlation_anomaly_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.CORRELATION_ANOMALY, {
            "strength": 0.8,
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "correlation_anomaly"

    def test_recalibration_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.RECALIBRATION_NEEDED, {
            "reason": "performance_degraded",
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "recalibration"

    def test_macro_event_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.MACRO_EVENT, {
            "indicator": "CPI",
            "surprise_magnitude": 2.5,
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert "CPI" in alerts[0]["title"]

    def test_system_error_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.SYSTEM_ERROR, {
            "error": "Connection timeout",
            "component": "data_pipeline",
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1

    def test_cooldown_suppresses_duplicate(self):
        from src.realtime.alert_engine import AlertConfig, AlertType
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        from src.realtime.alert_engine import AlertEngine
        engine = AlertEngine(bus, configs={
            AlertType.REGIME_CHANGE: AlertConfig(
                enabled=True, cooldown_seconds=3600
            ),
        })
        engine.start()

        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))
        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 3, "confidence": 0.8}))

        # Second should be suppressed
        assert engine.alert_count == 1

    def test_disabled_alert_skipped(self):
        from src.realtime.alert_engine import AlertConfig, AlertEngine, AlertType
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        engine = AlertEngine(bus, configs={
            AlertType.REGIME_CHANGE: AlertConfig(enabled=False),
        })
        engine.start()

        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))
        assert engine.alert_count == 0

    def test_acknowledge_alert(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))

        alerts = engine.get_alerts()
        alert_id = alerts[0]["alert_id"]

        assert engine.acknowledge_alert(alert_id)
        acked = engine.get_alerts()
        assert acked[0]["acknowledged"] is True

    def test_acknowledge_nonexistent(self):
        engine, bus = self._make_engine()
        assert engine.acknowledge_alert("nonexistent") is False

    def test_filter_by_type(self):
        from src.realtime.alert_engine import AlertType
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))
        bus.publish(Event(EventType.SYSTEM_ERROR, {"error": "x", "component": "y"}))

        regime_alerts = engine.get_alerts(alert_type=AlertType.REGIME_CHANGE)
        assert len(regime_alerts) == 1
        assert regime_alerts[0]["alert_type"] == "regime_change"

    def test_filter_by_severity(self):
        from src.realtime.alert_engine import AlertSeverity
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.VIX_SPIKE, {"level": 40.0}))
        bus.publish(Event(EventType.MACRO_EVENT, {"indicator": "X", "surprise_magnitude": 0.5}))

        critical = engine.get_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) == 1

    def test_unacknowledged_only(self):
        from src.realtime.event_bus import Event, EventType

        engine, bus = self._make_engine()
        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))

        alerts = engine.get_alerts()
        engine.acknowledge_alert(alerts[0]["alert_id"])

        unacked = engine.get_alerts(unacknowledged_only=True)
        assert len(unacked) == 0

    def test_update_config(self):
        from src.realtime.alert_engine import AlertConfig, AlertType

        engine, bus = self._make_engine()
        engine.update_config(
            AlertType.VIX_SPIKE,
            AlertConfig(enabled=True, cooldown_seconds=60, threshold=30.0),
        )

        cfg = engine.get_config(AlertType.VIX_SPIKE)
        assert cfg["vix_spike"]["threshold"] == 30.0

    def test_get_all_config(self):
        engine, bus = self._make_engine()
        cfg = engine.get_config()
        assert "regime_change" in cfg
        assert "vix_spike" in cfg

    def test_get_status(self):
        engine, bus = self._make_engine()
        status = engine.get_status()
        assert "is_running" in status
        assert "total_alerts" in status
        assert "by_type" in status
        assert "by_severity" in status

    def test_delivery_callback(self):
        from src.realtime.alert_engine import AlertEngine
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        delivered = []
        engine = AlertEngine(bus, delivery_callbacks=[
            lambda a: delivered.append(a),
        ])
        engine.start()

        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))
        assert len(delivered) == 1

    def test_max_history_trimming(self):
        from src.realtime.alert_engine import AlertConfig, AlertEngine, AlertType
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        engine = AlertEngine(bus, max_history=3, configs={
            AlertType.SYSTEM_ERROR: AlertConfig(
                enabled=True, cooldown_seconds=0
            ),
        })
        engine.start()

        for i in range(5):
            bus.publish(Event(EventType.SYSTEM_ERROR, {
                "error": f"err{i}", "component": "test",
            }))

        assert engine.alert_count == 3


# ═══════════════════════════════════════════════════════════════════
#  StreamManager Tests
# ═══════════════════════════════════════════════════════════════════


class TestStreamManager:
    """StreamManager SSE broadcasting tests."""

    def test_start_stop(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus)
        mgr.start()
        assert mgr.is_running
        mgr.stop()
        assert not mgr.is_running

    def test_format_sse(self):
        from src.realtime.event_bus import Event, EventType
        from src.realtime.stream_manager import StreamManager

        ev = Event(EventType.REGIME_CHANGE, {"new": 2}, source="test")
        sse = StreamManager._format_sse(ev)

        assert sse.startswith("event: regime_change\n")
        assert f"id: {ev.event_id}\n" in sse
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_connected_clients(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus)
        assert mgr.connected_clients == 0

    def test_get_status(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus)
        mgr.start()

        status = mgr.get_status()
        assert status["is_running"] is True
        assert status["connected_clients"] == 0
        mgr.stop()

    @pytest.mark.asyncio
    async def test_subscribe_yields_connected_event(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus, heartbeat_interval=0.1)
        mgr.start()

        gen = mgr.subscribe(client_id="test-1")
        first = await gen.__anext__()
        assert "connected" in first
        assert "test-1" in first
        assert mgr.connected_clients == 1

        # Cleanup
        mgr.stop()

    @pytest.mark.asyncio
    async def test_broadcast_to_client(self):
        from src.realtime.event_bus import Event, EventBus, EventType
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus, heartbeat_interval=60)
        mgr.start()

        gen = mgr.subscribe(client_id="c1")
        # Consume connection event
        await gen.__anext__()

        # Broadcast an event
        mgr.broadcast(Event(EventType.VIX_SPIKE, {"level": 30}))

        # Next yield should be the event
        msg = await gen.__anext__()
        assert "vix_spike" in msg

        mgr.stop()

    def test_disconnect_client(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.stream_manager import StreamManager

        bus = EventBus()
        mgr = StreamManager(bus)
        mgr.start()

        # No client to disconnect
        assert mgr.disconnect_client("nonexistent") is False
        mgr.stop()


# ═══════════════════════════════════════════════════════════════════
#  PaperTradingEngine Tests
# ═══════════════════════════════════════════════════════════════════


class TestPaperTradingEngine:
    """Paper trading engine tests."""

    def _make_engine(self, capital=100_000.0):
        from src.realtime.event_bus import EventBus
        from src.realtime.paper_trading import PaperTradingEngine

        bus = EventBus()
        engine = PaperTradingEngine(bus, initial_capital=capital)
        return engine, bus

    def test_initial_state(self):
        engine, bus = self._make_engine()
        assert engine.total_equity == 100_000.0
        assert engine.cash == 100_000.0
        assert engine.positions_value == 0.0
        assert engine.total_return == 0.0
        assert len(engine.get_positions()) == 0

    def test_simple_rebalance(self):
        engine, bus = self._make_engine()

        orders = engine.execute_rebalance(
            target_weights={"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
            prices={"SPY": 450.0, "TLT": 95.0, "GLD": 185.0},
            regime=1,
        )

        assert len(orders) > 0
        assert engine.cash < 100_000.0
        assert engine.positions_value > 0
        assert len(engine.get_positions()) == 3

    def test_rebalance_publishes_events(self):
        from src.realtime.event_bus import EventType

        engine, bus = self._make_engine()
        events = []
        bus.subscribe(EventType.REBALANCE_TRIGGER, lambda e: events.append(e))
        bus.subscribe(EventType.ORDER_FILLED, lambda e: events.append(e))

        engine.execute_rebalance(
            target_weights={"SPY": 0.6},
            prices={"SPY": 450.0},
            regime=1,
        )

        # Should have order fill and rebalance events
        event_types = [e.event_type for e in events]
        assert EventType.ORDER_FILLED in event_types
        assert EventType.REBALANCE_TRIGGER in event_types

    def test_get_portfolio_summary(self):
        engine, bus = self._make_engine()
        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 450.0},
            regime=1,
        )

        summary = engine.get_portfolio_summary()
        assert "total_equity" in summary
        assert "cash" in summary
        assert "positions" in summary
        assert summary["num_positions"] == 1

    def test_get_orders(self):
        engine, bus = self._make_engine()
        engine.execute_rebalance(
            target_weights={"SPY": 0.5, "TLT": 0.3},
            prices={"SPY": 450.0, "TLT": 95.0},
            regime=1,
        )

        orders = engine.get_orders()
        assert len(orders) == 2

        # Filter by asset
        spy_orders = engine.get_orders(asset="SPY")
        assert len(spy_orders) == 1
        assert spy_orders[0]["asset"] == "SPY"

    def test_get_equity_curve(self):
        engine, bus = self._make_engine()
        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 450.0},
            regime=1,
        )

        curve = engine.get_equity_curve()
        assert len(curve) >= 2  # Initial + after rebalance

    def test_equity_curve_limit(self):
        engine, bus = self._make_engine()
        for _ in range(5):
            engine.execute_rebalance(
                target_weights={"SPY": 0.5},
                prices={"SPY": 450.0},
                regime=1,
            )
        curve = engine.get_equity_curve(limit=2)
        assert len(curve) == 2

    def test_performance_metrics(self):
        engine, bus = self._make_engine()
        engine.execute_rebalance(
            target_weights={"SPY": 0.6},
            prices={"SPY": 450.0},
            regime=1,
        )
        engine.execute_rebalance(
            target_weights={"SPY": 0.6},
            prices={"SPY": 460.0},
            regime=1,
        )

        metrics = engine.get_performance_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "num_trades" in metrics

    def test_performance_metrics_empty(self):
        engine, bus = self._make_engine()
        metrics = engine.get_performance_metrics()
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0

    def test_regime_attribution(self):
        engine, bus = self._make_engine()

        # Buy in regime 1
        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 450.0},
            regime=1,
        )
        # Sell in regime 2 at higher price → realized P&L
        engine.execute_rebalance(
            target_weights={"SPY": 0.0},
            prices={"SPY": 460.0},
            regime=2,
        )

        attr = engine.get_regime_attribution()
        assert "regime_pnl" in attr
        assert "total_realized" in attr

    def test_max_position_pct_enforced(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.paper_trading import PaperTradingEngine

        bus = EventBus()
        engine = PaperTradingEngine(
            bus, initial_capital=100_000, max_position_pct=0.30
        )

        engine.execute_rebalance(
            target_weights={"SPY": 0.8},  # Exceeds 30% limit
            prices={"SPY": 450.0},
            regime=1,
        )

        positions = engine.get_positions()
        total = engine.total_equity
        for p in positions:
            assert p["market_value"] / total <= 0.35  # Allow small float error

    def test_transaction_fees(self):
        from src.realtime.event_bus import EventBus
        from src.realtime.paper_trading import PaperTradingEngine

        bus = EventBus()
        # Use max_position_pct=1.0 so the 50% weight is not capped
        engine = PaperTradingEngine(
            bus, initial_capital=100_000.0, max_position_pct=1.0
        )
        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 100.0},
            regime=1,
        )

        # Cash should be less than half due to fees
        assert engine.cash < 50_000.0

    def test_sell_reduces_position(self):
        engine, bus = self._make_engine()

        # Buy
        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 100.0},
            regime=1,
        )
        first_qty = engine.get_positions()[0]["quantity"]

        # Reduce
        engine.execute_rebalance(
            target_weights={"SPY": 0.2},
            prices={"SPY": 100.0},
            regime=1,
        )
        second_qty = engine.get_positions()[0]["quantity"]
        assert second_qty < first_qty

    def test_full_liquidation(self):
        engine, bus = self._make_engine()

        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 100.0},
            regime=1,
        )
        engine.execute_rebalance(
            target_weights={"SPY": 0.0},
            prices={"SPY": 100.0},
            regime=1,
        )

        assert len(engine.get_positions()) == 0

    def test_get_status(self):
        engine, bus = self._make_engine()
        status = engine.get_status()
        assert "initial_capital" in status
        assert "total_equity" in status
        assert "num_orders" in status

    def test_order_to_dict(self):
        from src.realtime.paper_trading import OrderSide, OrderStatus, PaperOrder

        order = PaperOrder(
            order_id="abc123",
            asset="SPY",
            side=OrderSide.BUY,
            quantity=10.0,
            price=450.0,
        )
        d = order.to_dict()
        assert d["order_id"] == "abc123"
        assert d["side"] == "buy"
        assert d["notional"] == 4500.0

    def test_position_properties(self):
        from src.realtime.paper_trading import Position

        pos = Position(
            asset="SPY", quantity=10, avg_cost=100, current_price=110
        )
        assert pos.market_value == 1100
        assert pos.cost_basis == 1000
        assert pos.unrealized_pnl == 100
        assert abs(pos.unrealized_pnl_pct - 0.10) < 0.001

    def test_position_zero_cost(self):
        from src.realtime.paper_trading import Position

        pos = Position(asset="X", quantity=0, avg_cost=0, current_price=50)
        assert pos.unrealized_pnl_pct == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Phase 4 API Endpoint Tests
# ═══════════════════════════════════════════════════════════════════


def _add_phase4_mocks(mock_system):
    """Add Phase 4 component mocks to the standard mock system."""
    from src.realtime.alert_engine import AlertEngine
    from src.realtime.event_bus import EventBus
    from src.realtime.paper_trading import PaperTradingEngine
    from src.realtime.stream_manager import StreamManager

    bus = EventBus()
    alert_engine = AlertEngine(bus)
    alert_engine.start()
    stream_manager = StreamManager(bus)
    stream_manager.start()
    paper_trading = PaperTradingEngine(bus, initial_capital=100_000)

    mock_system.event_bus = bus
    mock_system.alert_engine = alert_engine
    mock_system.stream_manager = stream_manager
    mock_system.paper_trading = paper_trading
    mock_system.scheduler = MagicMock()
    mock_system.scheduler.get_status.return_value = {
        "is_running": False,
        "interval_seconds": 900,
        "market_hours_only": True,
        "run_count": 0,
        "error_count": 0,
        "last_run_time": None,
        "last_regime": None,
    }

    return mock_system


@pytest.fixture
def phase4_client(mock_system):
    """FastAPI TestClient with Phase 4 mocks."""
    import api.dependencies as deps
    from contextlib import asynccontextmanager

    _add_phase4_mocks(mock_system)

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


class TestPhase4StatusAPI:
    """Phase 4 status endpoint tests."""

    def test_status(self, phase4_client):
        resp = phase4_client.get("/api/phase4/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "event_bus" in data
        assert "scheduler" in data
        assert "alert_engine" in data
        assert "stream_manager" in data
        assert "paper_trading" in data


class TestPhase4EventsAPI:
    """Event bus API tests."""

    def test_get_events_empty(self, phase4_client):
        resp = phase4_client.get("/api/phase4/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []

    def test_get_events_with_data(self, phase4_client, mock_system):
        from src.realtime.event_bus import Event, EventType

        bus = mock_system.event_bus
        bus.publish(Event(EventType.REGIME_CHANGE, {"new": 2}))

        resp = phase4_client.get("/api/phase4/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_get_events_filter(self, phase4_client, mock_system):
        from src.realtime.event_bus import Event, EventType

        bus = mock_system.event_bus
        bus.publish(Event(EventType.REGIME_CHANGE, {}))
        bus.publish(Event(EventType.VIX_SPIKE, {"level": 30}))

        resp = phase4_client.get("/api/phase4/events?event_type=vix_spike")
        assert resp.status_code == 200
        data = resp.json()
        # Only vix_spike events returned
        for ev in data["events"]:
            assert ev["event_type"] == "vix_spike"

    def test_get_events_invalid_type(self, phase4_client):
        resp = phase4_client.get("/api/phase4/events?event_type=invalid_type")
        assert resp.status_code == 400


class TestPhase4AlertsAPI:
    """Alert engine API tests."""

    def test_get_alerts_empty(self, phase4_client):
        resp = phase4_client.get("/api/phase4/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alerts"] == []

    def test_get_alerts_with_data(self, phase4_client, mock_system):
        from src.realtime.event_bus import Event, EventType

        bus = mock_system.event_bus
        bus.publish(Event(EventType.REGIME_CHANGE, {
            "new_regime": 2, "confidence": 0.9
        }))

        resp = phase4_client.get("/api/phase4/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) >= 1

    def test_acknowledge_alert(self, phase4_client, mock_system):
        from src.realtime.event_bus import Event, EventType

        bus = mock_system.event_bus
        bus.publish(Event(EventType.REGIME_CHANGE, {
            "new_regime": 2, "confidence": 0.9
        }))

        alerts_resp = phase4_client.get("/api/phase4/alerts")
        alert_id = alerts_resp.json()["alerts"][0]["alert_id"]

        resp = phase4_client.post(
            f"/api/phase4/alerts/acknowledge?alert_id={alert_id}"
        )
        assert resp.status_code == 200

    def test_acknowledge_nonexistent(self, phase4_client):
        resp = phase4_client.post(
            "/api/phase4/alerts/acknowledge?alert_id=nonexistent"
        )
        assert resp.status_code == 404

    def test_get_alert_config(self, phase4_client):
        resp = phase4_client.get("/api/phase4/alerts/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "configs" in data
        assert "regime_change" in data["configs"]

    def test_update_alert_config(self, phase4_client):
        resp = phase4_client.post(
            "/api/phase4/alerts/config",
            json={
                "alert_type": "vix_spike",
                "cooldown_seconds": 120,
                "threshold": 30.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["threshold"] == 30.0

    def test_update_alert_config_invalid_type(self, phase4_client):
        resp = phase4_client.post(
            "/api/phase4/alerts/config",
            json={"alert_type": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_filter_alerts_by_type(self, phase4_client, mock_system):
        from src.realtime.event_bus import Event, EventType

        bus = mock_system.event_bus
        bus.publish(Event(EventType.REGIME_CHANGE, {"new_regime": 2, "confidence": 0.9}))
        bus.publish(Event(EventType.SYSTEM_ERROR, {"error": "x", "component": "y"}))

        resp = phase4_client.get("/api/phase4/alerts?alert_type=regime_change")
        assert resp.status_code == 200
        data = resp.json()
        for a in data["alerts"]:
            assert a["alert_type"] == "regime_change"


class TestPhase4StreamAPI:
    """SSE streaming API tests."""

    def test_stream_status(self, phase4_client):
        resp = phase4_client.get("/api/phase4/stream/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_running" in data
        assert "connected_clients" in data


class TestPhase4PortfolioAPI:
    """Paper trading portfolio API tests."""

    def test_get_portfolio_empty(self, phase4_client):
        resp = phase4_client.get("/api/phase4/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_equity"] == 100_000.0
        assert data["num_positions"] == 0

    def test_get_performance_metrics(self, phase4_client):
        resp = phase4_client.get("/api/phase4/portfolio/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_return" in data
        assert "sharpe_ratio" in data

    def test_get_equity_curve(self, phase4_client):
        resp = phase4_client.get("/api/phase4/portfolio/equity")
        assert resp.status_code == 200
        data = resp.json()
        assert "curve" in data

    def test_get_regime_attribution(self, phase4_client):
        resp = phase4_client.get("/api/phase4/portfolio/attribution")
        assert resp.status_code == 200
        data = resp.json()
        assert "regime_pnl" in data

    def test_rebalance(self, phase4_client):
        resp = phase4_client.post(
            "/api/phase4/rebalance",
            json={
                "target_weights": {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
                "prices": {"SPY": 450.0, "TLT": 95.0, "GLD": 185.0},
                "regime": 1,
                "reason": "test_rebalance",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_orders"] > 0
        assert data["total_equity"] > 0

    def test_rebalance_excessive_weights(self, phase4_client):
        resp = phase4_client.post(
            "/api/phase4/rebalance",
            json={
                "target_weights": {"SPY": 0.6, "TLT": 0.5, "GLD": 0.5},
                "prices": {"SPY": 450.0, "TLT": 95.0, "GLD": 185.0},
            },
        )
        assert resp.status_code == 400

    def test_get_trades_empty(self, phase4_client):
        resp = phase4_client.get("/api/phase4/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert data["orders"] == []

    def test_get_trades_after_rebalance(self, phase4_client):
        # Execute a rebalance first
        phase4_client.post(
            "/api/phase4/rebalance",
            json={
                "target_weights": {"SPY": 0.5},
                "prices": {"SPY": 450.0},
                "regime": 1,
            },
        )

        resp = phase4_client.get("/api/phase4/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["orders"]) > 0

    def test_portfolio_after_rebalance(self, phase4_client):
        phase4_client.post(
            "/api/phase4/rebalance",
            json={
                "target_weights": {"SPY": 0.5, "TLT": 0.3},
                "prices": {"SPY": 450.0, "TLT": 95.0},
                "regime": 1,
            },
        )

        resp = phase4_client.get("/api/phase4/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_positions"] == 2
        assert data["total_equity"] > 0


# ═══════════════════════════════════════════════════════════════════
#  Integration Tests
# ═══════════════════════════════════════════════════════════════════


class TestPhase4Integration:
    """End-to-end integration of Phase 4 components."""

    def test_event_bus_to_alert_engine(self):
        """Events published on bus trigger alerts in engine."""
        from src.realtime.alert_engine import AlertEngine
        from src.realtime.event_bus import Event, EventBus, EventType

        bus = EventBus()
        engine = AlertEngine(bus)
        engine.start()

        bus.publish(Event(EventType.REGIME_CHANGE, {
            "previous_regime": 1,
            "new_regime": 2,
            "regime_name": "Risk-Off Crisis",
            "confidence": 0.9,
        }))

        alerts = engine.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["title"].startswith("Regime Change")

    def test_paper_trading_publishes_events(self):
        """Paper trading engine publishes events that the bus can relay."""
        from src.realtime.event_bus import EventBus, EventType
        from src.realtime.paper_trading import PaperTradingEngine

        bus = EventBus()
        engine = PaperTradingEngine(bus)

        events = []
        bus.subscribe(EventType.ORDER_FILLED, lambda e: events.append(e))
        bus.subscribe(EventType.REBALANCE_TRIGGER, lambda e: events.append(e))

        engine.execute_rebalance(
            target_weights={"SPY": 0.5},
            prices={"SPY": 450.0},
            regime=1,
        )

        order_events = [e for e in events if e.event_type == EventType.ORDER_FILLED]
        rebal_events = [e for e in events if e.event_type == EventType.REBALANCE_TRIGGER]
        assert len(order_events) >= 1
        assert len(rebal_events) == 1

    def test_full_pipeline_bus_alerts_trading(self):
        """Full flow: event → alert + paper trading in same bus."""
        from src.realtime.alert_engine import AlertEngine
        from src.realtime.event_bus import Event, EventBus, EventType
        from src.realtime.paper_trading import PaperTradingEngine

        bus = EventBus()
        alert_engine = AlertEngine(bus)
        alert_engine.start()
        paper = PaperTradingEngine(bus)

        # Simulate regime change
        bus.publish(Event(EventType.REGIME_CHANGE, {
            "previous_regime": 1,
            "new_regime": 2,
            "regime_name": "Risk-Off",
            "confidence": 0.85,
        }))

        # Execute trades based on new regime
        paper.execute_rebalance(
            target_weights={"TLT": 0.6, "GLD": 0.3},
            prices={"TLT": 95.0, "GLD": 185.0},
            regime=2,
        )

        # Verify alerts from regime change
        alerts = alert_engine.get_alerts()
        assert any(a["alert_type"] == "regime_change" for a in alerts)

        # Verify positions
        positions = paper.get_positions()
        assert len(positions) == 2

        # Verify bus has event history
        history = bus.get_history()
        assert len(history) >= 3  # regime_change + orders + rebalance

    @pytest.mark.asyncio
    async def test_scheduler_triggers_analysis(self):
        """Scheduler runs analysis and publishes events."""
        from src.realtime.event_bus import EventBus, EventType
        from src.realtime.scheduler import AnalysisScheduler

        bus = EventBus()
        system = MagicMock()
        system.analyze.return_value = {
            "regime": {"id": 1, "name": "Growth", "confidence": 0.8, "disagreement": 0.2},
            "modules": {},
        }

        scheduler = AnalysisScheduler(system, bus, interval_seconds=1, market_hours_only=False)

        events = []
        bus.subscribe(EventType.ANALYSIS_COMPLETE, lambda e: events.append(e))

        await scheduler.trigger_now()
        assert len(events) == 1
        assert events[0].data["regime"] == 1
