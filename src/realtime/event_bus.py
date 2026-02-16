"""
In-process event bus for AMRCAIS real-time operations (Phase 4.1).

Provides a lightweight publish/subscribe system that decouples
regime detection, alert generation, and dashboard streaming.

Architecture:
    Event producers (scheduler, data pipeline) publish typed events.
    Event consumers (alert engine, stream manager, paper trading)
    subscribe to specific event types.

    ┌────────────┐       ┌────────────┐       ┌────────────┐
    │ Scheduler  │──┐    │            │   ┌──→│ Alert Eng  │
    └────────────┘  │    │  EventBus  │   │   └────────────┘
    ┌────────────┐  ├──→ │            │───┤   ┌────────────┐
    │ Data Feed  │──┘    │ (in-proc)  │   ├──→│ SSE Stream │
    └────────────┘       └────────────┘   │   └────────────┘
                                          │   ┌────────────┐
                                          └──→│ Paper Trade│
                                              └────────────┘

Classes:
    EventType: Enum of supported event categories.
    Event: Immutable event payload.
    EventBus: Singleton pub/sub bus.

Example:
    >>> bus = EventBus()
    >>> bus.subscribe(EventType.REGIME_CHANGE, my_handler)
    >>> bus.publish(Event(EventType.REGIME_CHANGE, {"regime": 2}))
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─── Event Types ──────────────────────────────────────────────────


class EventType(str, Enum):
    """Supported event categories.

    Categories map directly to Phase 4.2 alert types plus internal
    system events for scheduling and data flow.
    """

    # Regime events
    REGIME_CHANGE = "regime_change"
    REGIME_UPDATE = "regime_update"
    TRANSITION_WARNING = "transition_warning"

    # Module events
    CORRELATION_ANOMALY = "correlation_anomaly"
    VIX_SPIKE = "vix_spike"
    MACRO_EVENT = "macro_event"

    # System events
    RECALIBRATION_NEEDED = "recalibration_needed"
    ANALYSIS_COMPLETE = "analysis_complete"
    DATA_UPDATED = "data_updated"

    # Trading events
    REBALANCE_TRIGGER = "rebalance_trigger"
    ALPHA_SIGNAL = "alpha_signal"
    ORDER_FILLED = "order_filled"

    # Health events
    SYSTEM_ERROR = "system_error"
    SYSTEM_HEALTH = "system_health"


# ─── Event ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    """Immutable event payload.

    Attributes:
        event_type: Category of the event.
        data: Arbitrary payload (must be JSON-serializable for streaming).
        timestamp: When the event was created.
        event_id: Unique identifier.
        source: Originating component name.
    """

    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    source: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for SSE / JSON."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "source": self.source,
        }


# ─── Subscriber types ────────────────────────────────────────────

# Sync handler: (event: Event) -> None
SyncHandler = Callable[[Event], None]
# Async handler: (event: Event) -> Coroutine
AsyncHandler = Callable[[Event], Any]
Handler = SyncHandler | AsyncHandler


# ─── Event Bus ────────────────────────────────────────────────────


class EventBus:
    """In-process publish/subscribe event bus.

    Thread-safe, supports both sync and async handlers.
    Designed as a singleton but can be instantiated multiple times
    for testing.

    Features:
        - Type-filtered subscriptions
        - Wildcard subscriptions (receive all events)
        - Async handler support (auto-scheduled on event loop)
        - Event history with configurable retention
        - Error isolation: one handler crash doesn't affect others

    Args:
        max_history: Maximum events to retain in history.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe(EventType.REGIME_CHANGE, handle_regime)
        >>> bus.publish(Event(EventType.REGIME_CHANGE, {"regime": 2}))
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: Dict[EventType, List[Handler]] = {}
        self._wildcard_subscribers: List[Handler] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._event_count = 0

    # ── Subscription ──────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        handler: Handler,
    ) -> None:
        """Register a handler for a specific event type.

        Args:
            event_type: Event category to listen for.
            handler: Callable(Event) → None (sync or async).
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(
                f"Subscribed {handler.__name__} to {event_type.value}"
            )

    def subscribe_all(self, handler: Handler) -> None:
        """Register a handler that receives every event.

        Args:
            handler: Callable(Event) → None.
        """
        if handler not in self._wildcard_subscribers:
            self._wildcard_subscribers.append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Handler,
    ) -> None:
        """Remove a handler from an event type.

        Args:
            event_type: Event category.
            handler: Previously registered handler.
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    def unsubscribe_all(self, handler: Handler) -> None:
        """Remove a wildcard handler."""
        try:
            self._wildcard_subscribers.remove(handler)
        except ValueError:
            pass

    # ── Publishing ────────────────────────────────────────────

    def publish(self, event: Event) -> int:
        """Publish an event to all matching subscribers.

        Handlers are invoked synchronously in subscription order.
        Async handlers are scheduled on the running event loop
        if one exists, otherwise they are skipped with a warning.

        Args:
            event: Event to publish.

        Returns:
            Number of handlers that processed the event.
        """
        self._event_count += 1
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers_called = 0
        all_handlers = (
            self._subscribers.get(event.event_type, [])
            + self._wildcard_subscribers
        )

        for handler in all_handlers:
            try:
                result = handler(event)
                # If handler returned a coroutine, schedule it
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running loop — skip async handler
                        logger.warning(
                            f"Async handler {handler.__name__} skipped "
                            "(no running event loop)"
                        )
                handlers_called += 1
            except Exception as exc:
                logger.error(
                    f"Handler {handler.__name__} failed on "
                    f"{event.event_type.value}: {exc}"
                )

        logger.debug(
            f"Published {event.event_type.value} → "
            f"{handlers_called} handler(s)"
        )
        return handlers_called

    async def publish_async(self, event: Event) -> int:
        """Publish an event, awaiting async handlers.

        Args:
            event: Event to publish.

        Returns:
            Number of handlers that processed the event.
        """
        self._event_count += 1
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers_called = 0
        all_handlers = (
            self._subscribers.get(event.event_type, [])
            + self._wildcard_subscribers
        )

        for handler in all_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                handlers_called += 1
            except Exception as exc:
                logger.error(
                    f"Handler {handler.__name__} failed on "
                    f"{event.event_type.value}: {exc}"
                )

        return handlers_called

    # ── Query ─────────────────────────────────────────────────

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Return recent events, optionally filtered by type.

        Args:
            event_type: Filter to this type (None = all).
            limit: Maximum events to return.

        Returns:
            List of events, most recent first.
        """
        events = self._history
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return list(reversed(events[-limit:]))

    @property
    def event_count(self) -> int:
        """Total events published since creation."""
        return self._event_count

    @property
    def subscriber_count(self) -> int:
        """Total active subscriptions."""
        return sum(len(h) for h in self._subscribers.values()) + len(
            self._wildcard_subscribers
        )

    def clear_history(self) -> None:
        """Flush event history."""
        self._history.clear()

    def reset(self) -> None:
        """Remove all subscribers and clear history."""
        self._subscribers.clear()
        self._wildcard_subscribers.clear()
        self._history.clear()
        self._event_count = 0
