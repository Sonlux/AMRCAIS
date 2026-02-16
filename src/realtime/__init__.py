"""
Phase 4: Real-Time + Execution package.

Provides event-driven architecture, alert engine, streaming,
and paper trading integration for AMRCAIS.

Modules:
    event_bus:      In-process publish/subscribe event bus
    scheduler:      Periodic regime re-analysis scheduler
    alert_engine:   Multi-channel alert system with fatigue management
    stream_manager: SSE streaming for live dashboard updates
    paper_trading:  Paper trading with regime-aware order generation
"""

from src.realtime.event_bus import EventBus, Event, EventType
from src.realtime.scheduler import AnalysisScheduler
from src.realtime.alert_engine import AlertEngine, Alert, AlertType, AlertSeverity
from src.realtime.stream_manager import StreamManager
from src.realtime.paper_trading import PaperTradingEngine

__all__ = [
    "EventBus",
    "Event",
    "EventType",
    "AnalysisScheduler",
    "AlertEngine",
    "Alert",
    "AlertType",
    "AlertSeverity",
    "StreamManager",
    "PaperTradingEngine",
]
