"""
Alert engine for AMRCAIS (Phase 4.2).

Subscribes to EventBus events and generates typed, severity-levelled
alerts with fatigue management (cooldown periods) and multi-channel
delivery stubs.

Alert Types (from PRD):
    🔴 REGIME_CHANGE      — Market regime transition detected
    🟡 TRANSITION_WARNING  — High classifier disagreement (>0.6)
    🟠 CORRELATION_ANOMALY — Unusual cross-asset correlations
    🔵 RECALIBRATION       — Meta-learner recommends recalibration
    ⚪ MACRO_EVENT         — Significant macro data surprise

Classes:
    AlertType: Enum of alert categories.
    AlertSeverity: Enum of severity levels.
    Alert: Immutable alert record.
    AlertConfig: Per-type configuration (threshold, cooldown, enabled).
    AlertEngine: Core engine that maps events → alerts.

Example:
    >>> engine = AlertEngine(event_bus)
    >>> engine.start()
    >>> alerts = engine.get_alerts(limit=10)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    """Categories of alerts."""

    REGIME_CHANGE = "regime_change"
    TRANSITION_WARNING = "transition_warning"
    CORRELATION_ANOMALY = "correlation_anomaly"
    RECALIBRATION = "recalibration"
    MACRO_EVENT = "macro_event"
    VIX_SPIKE = "vix_spike"
    SYSTEM_ERROR = "system_error"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class Alert:
    """Immutable alert record.

    Args:
        alert_type: Category of alert.
        severity: Severity level.
        title: Short human-readable title.
        message: Detailed description.
        data: Raw event data.
        timestamp: Creation time (epoch).
        alert_id: Unique identifier.
        source_event_id: ID of the triggering event.
        acknowledged: Whether the alert has been acknowledged.
    """

    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:12]
    )
    source_event_id: str = ""
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert for API / SSE delivery."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "source_event_id": self.source_event_id,
            "acknowledged": self.acknowledged,
        }


@dataclass
class AlertConfig:
    """Per-type alert configuration.

    Args:
        enabled: Whether this alert type fires.
        cooldown_seconds: Minimum gap between same-type alerts (fatigue).
        threshold: Numeric threshold triggering the alert.
    """

    enabled: bool = True
    cooldown_seconds: float = 300.0  # 5 minutes default
    threshold: float = 0.0


# ── Default configs (can be overridden via API) ────────────────

_DEFAULT_CONFIGS: Dict[AlertType, AlertConfig] = {
    AlertType.REGIME_CHANGE: AlertConfig(
        enabled=True, cooldown_seconds=600, threshold=0.0
    ),
    AlertType.TRANSITION_WARNING: AlertConfig(
        enabled=True, cooldown_seconds=300, threshold=0.6
    ),
    AlertType.CORRELATION_ANOMALY: AlertConfig(
        enabled=True, cooldown_seconds=600, threshold=0.5
    ),
    AlertType.RECALIBRATION: AlertConfig(
        enabled=True, cooldown_seconds=1800, threshold=0.0
    ),
    AlertType.MACRO_EVENT: AlertConfig(
        enabled=True, cooldown_seconds=300, threshold=0.0
    ),
    AlertType.VIX_SPIKE: AlertConfig(
        enabled=True, cooldown_seconds=900, threshold=25.0
    ),
    AlertType.SYSTEM_ERROR: AlertConfig(
        enabled=True, cooldown_seconds=60, threshold=0.0
    ),
}


class AlertEngine:
    """Maps EventBus events to typed, severity-levelled alerts.

    Subscribes to relevant event types and generates alerts with
    cooldown-based fatigue management.

    Args:
        event_bus: EventBus to subscribe to.
        configs: Optional per-type config overrides.
        max_history: Maximum number of alerts to retain.
        delivery_callbacks: Optional list of callables for alert delivery.

    Example:
        >>> engine = AlertEngine(bus)
        >>> engine.start()
        >>> engine.get_alerts(alert_type=AlertType.REGIME_CHANGE)
    """

    def __init__(
        self,
        event_bus: EventBus,
        configs: Optional[Dict[AlertType, AlertConfig]] = None,
        max_history: int = 500,
        delivery_callbacks: Optional[List[Callable]] = None,
    ) -> None:
        self._bus = event_bus
        self._configs: Dict[AlertType, AlertConfig] = {
            **_DEFAULT_CONFIGS,
            **(configs or {}),
        }
        self._max_history = max_history
        self._delivery_callbacks = delivery_callbacks or []

        self._alerts: List[Alert] = []
        self._last_fired: Dict[AlertType, float] = {}
        self._is_running = False

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to events and begin generating alerts."""
        if self._is_running:
            return
        self._bus.subscribe(
            EventType.REGIME_CHANGE, self._on_regime_change
        )
        self._bus.subscribe(
            EventType.TRANSITION_WARNING, self._on_transition_warning
        )
        self._bus.subscribe(
            EventType.CORRELATION_ANOMALY, self._on_correlation_anomaly
        )
        self._bus.subscribe(
            EventType.RECALIBRATION_NEEDED, self._on_recalibration
        )
        self._bus.subscribe(
            EventType.MACRO_EVENT, self._on_macro_event
        )
        self._bus.subscribe(
            EventType.VIX_SPIKE, self._on_vix_spike
        )
        self._bus.subscribe(
            EventType.SYSTEM_ERROR, self._on_system_error
        )
        self._is_running = True
        logger.info("Alert engine started")

    def stop(self) -> None:
        """Unsubscribe from all events."""
        self._bus.unsubscribe(
            EventType.REGIME_CHANGE, self._on_regime_change
        )
        self._bus.unsubscribe(
            EventType.TRANSITION_WARNING, self._on_transition_warning
        )
        self._bus.unsubscribe(
            EventType.CORRELATION_ANOMALY, self._on_correlation_anomaly
        )
        self._bus.unsubscribe(
            EventType.RECALIBRATION_NEEDED, self._on_recalibration
        )
        self._bus.unsubscribe(
            EventType.MACRO_EVENT, self._on_macro_event
        )
        self._bus.unsubscribe(
            EventType.VIX_SPIKE, self._on_vix_spike
        )
        self._bus.unsubscribe(
            EventType.SYSTEM_ERROR, self._on_system_error
        )
        self._is_running = False
        logger.info("Alert engine stopped")

    @property
    def is_running(self) -> bool:
        """Whether the engine is actively processing events."""
        return self._is_running

    # ── Event handlers ────────────────────────────────────────

    def _on_regime_change(self, event: Event) -> None:
        """Handle regime change events."""
        data = event.data
        prev = data.get("previous_regime", "?")
        new = data.get("new_regime", "?")
        name = data.get("regime_name", "")
        conf = data.get("confidence", 0.0)

        severity = AlertSeverity.CRITICAL if conf > 0.7 else AlertSeverity.HIGH

        self._create_alert(
            AlertType.REGIME_CHANGE,
            severity,
            f"Regime Change: {name}",
            f"Market regime shifted from {prev} to {new} "
            f"(confidence: {conf:.1%})",
            data,
            event.event_id,
        )

    def _on_transition_warning(self, event: Event) -> None:
        """Handle transition warning events."""
        data = event.data
        disagreement = data.get("disagreement", 0.0)
        cfg = self._configs[AlertType.TRANSITION_WARNING]

        if disagreement < cfg.threshold:
            return

        severity = (
            AlertSeverity.HIGH
            if disagreement > 0.8
            else AlertSeverity.MEDIUM
        )

        self._create_alert(
            AlertType.TRANSITION_WARNING,
            severity,
            "Regime Transition Warning",
            f"Classifier disagreement at {disagreement:.1%} — "
            f"potential regime transition imminent",
            data,
            event.event_id,
        )

    def _on_correlation_anomaly(self, event: Event) -> None:
        """Handle correlation anomaly events."""
        data = event.data
        strength = data.get("strength", 0.0)
        cfg = self._configs[AlertType.CORRELATION_ANOMALY]

        if strength < cfg.threshold:
            return

        severity = (
            AlertSeverity.HIGH
            if strength > 0.8
            else AlertSeverity.MEDIUM
        )

        self._create_alert(
            AlertType.CORRELATION_ANOMALY,
            severity,
            "Correlation Anomaly Detected",
            f"Cross-asset correlation anomaly (strength: {strength:.2f})",
            data,
            event.event_id,
        )

    def _on_recalibration(self, event: Event) -> None:
        """Handle recalibration needed events."""
        self._create_alert(
            AlertType.RECALIBRATION,
            AlertSeverity.MEDIUM,
            "Model Recalibration Recommended",
            "Meta-learner detected degraded classifier performance — "
            "recalibration advised",
            event.data,
            event.event_id,
        )

    def _on_macro_event(self, event: Event) -> None:
        """Handle macro event surprises."""
        data = event.data
        indicator = data.get("indicator", "Unknown")
        surprise = data.get("surprise_magnitude", 0.0)

        severity = (
            AlertSeverity.HIGH
            if abs(surprise) > 2.0
            else AlertSeverity.MEDIUM
        )

        self._create_alert(
            AlertType.MACRO_EVENT,
            severity,
            f"Macro Surprise: {indicator}",
            f"Macro data surprise on {indicator} "
            f"(magnitude: {surprise:+.2f}σ)",
            data,
            event.event_id,
        )

    def _on_vix_spike(self, event: Event) -> None:
        """Handle VIX spike events."""
        data = event.data
        level = data.get("level", 0.0)
        cfg = self._configs[AlertType.VIX_SPIKE]

        if level < cfg.threshold:
            return

        severity = (
            AlertSeverity.CRITICAL
            if level > 35
            else AlertSeverity.HIGH
        )

        self._create_alert(
            AlertType.VIX_SPIKE,
            severity,
            f"VIX Spike: {level:.1f}",
            f"VIX surged to {level:.1f} — elevated volatility regime",
            data,
            event.event_id,
        )

    def _on_system_error(self, event: Event) -> None:
        """Handle system error events."""
        error = event.data.get("error", "Unknown")
        component = event.data.get("component", "unknown")

        self._create_alert(
            AlertType.SYSTEM_ERROR,
            AlertSeverity.HIGH,
            f"System Error: {component}",
            f"Error in {component}: {error}",
            event.data,
            event.event_id,
        )

    # ── Alert creation with fatigue management ────────────────

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        data: Dict[str, Any],
        source_event_id: str,
    ) -> Optional[Alert]:
        """Create and store alert, respecting cooldown.

        Returns:
            The created Alert, or None if suppressed by cooldown.
        """
        cfg = self._configs.get(alert_type, AlertConfig())
        if not cfg.enabled:
            return None

        # Cooldown check
        now = time.time()
        last = self._last_fired.get(alert_type, 0.0)
        if now - last < cfg.cooldown_seconds:
            logger.debug(
                f"Alert {alert_type.value} suppressed (cooldown)"
            )
            return None

        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=data,
            source_event_id=source_event_id,
        )

        self._alerts.append(alert)
        self._last_fired[alert_type] = now

        # Trim history
        if len(self._alerts) > self._max_history:
            self._alerts = self._alerts[-self._max_history :]

        logger.info(
            f"Alert [{severity.value}] {title}"
        )

        # Deliver to callbacks
        for cb in self._delivery_callbacks:
            try:
                cb(alert)
            except Exception as exc:
                logger.error(f"Alert delivery callback failed: {exc}")

        return alert

    # ── Query interface ───────────────────────────────────────

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 50,
        unacknowledged_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query alerts with optional filters.

        Args:
            alert_type: Filter by alert category.
            severity: Filter by severity level.
            limit: Maximum number of alerts to return.
            unacknowledged_only: Only return unacknowledged alerts.

        Returns:
            List of alert dicts, newest first.
        """
        alerts = self._alerts

        if alert_type is not None:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return [a.to_dict() for a in reversed(alerts)][:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: ID of the alert to acknowledge.

        Returns:
            True if found and acknowledged, False otherwise.
        """
        for i, alert in enumerate(self._alerts):
            if alert.alert_id == alert_id:
                # Replace with acknowledged copy (frozen dataclass)
                self._alerts[i] = Alert(
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    title=alert.title,
                    message=alert.message,
                    data=alert.data,
                    timestamp=alert.timestamp,
                    alert_id=alert.alert_id,
                    source_event_id=alert.source_event_id,
                    acknowledged=True,
                )
                return True
        return False

    def update_config(
        self, alert_type: AlertType, config: AlertConfig
    ) -> None:
        """Update configuration for a specific alert type.

        Args:
            alert_type: Alert category to configure.
            config: New configuration.
        """
        self._configs[alert_type] = config
        logger.info(
            f"Alert config updated: {alert_type.value} "
            f"(enabled={config.enabled}, "
            f"cooldown={config.cooldown_seconds}s)"
        )

    def get_config(
        self, alert_type: Optional[AlertType] = None
    ) -> Dict[str, Any]:
        """Get current alert configs.

        Args:
            alert_type: Specific type, or None for all.

        Returns:
            Config dict.
        """
        if alert_type:
            cfg = self._configs.get(alert_type, AlertConfig())
            return {
                alert_type.value: {
                    "enabled": cfg.enabled,
                    "cooldown_seconds": cfg.cooldown_seconds,
                    "threshold": cfg.threshold,
                }
            }
        return {
            at.value: {
                "enabled": c.enabled,
                "cooldown_seconds": c.cooldown_seconds,
                "threshold": c.threshold,
            }
            for at, c in self._configs.items()
        }

    @property
    def alert_count(self) -> int:
        """Total alerts generated."""
        return len(self._alerts)

    def get_status(self) -> Dict[str, Any]:
        """Return engine status summary."""
        return {
            "is_running": self._is_running,
            "total_alerts": len(self._alerts),
            "unacknowledged": sum(
                1 for a in self._alerts if not a.acknowledged
            ),
            "by_type": {
                at.value: sum(
                    1 for a in self._alerts if a.alert_type == at
                )
                for at in AlertType
            },
            "by_severity": {
                s.value: sum(
                    1 for a in self._alerts if a.severity == s
                )
                for s in AlertSeverity
            },
        }
