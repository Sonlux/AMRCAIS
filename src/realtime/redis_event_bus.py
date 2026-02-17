"""
Redis-backed event bus for AMRCAIS (Phase 4.1 — Production).

Extends the in-process EventBus with Redis Pub/Sub for horizontal
scalability.  Falls back to the in-process bus when Redis is
unavailable (dev / test mode).

Architecture:
    ┌────────────────┐       ┌──────────────┐       ┌────────────────┐
    │  Publisher A   │──────→│   Redis      │──────→│  Subscriber A  │
    └────────────────┘       │   Channel    │       └────────────────┘
    ┌────────────────┐       │  "amrcais:   │       ┌────────────────┐
    │  Publisher B   │──────→│   events"    │──────→│  Subscriber B  │
    └────────────────┘       └──────────────┘       └────────────────┘

The RedisEventBus wraps the base EventBus and adds cross-process
event propagation via Redis Pub/Sub.  Handlers are still invoked
in-process, but events published on one node are visible to all.

Configuration via environment variables:
    REDIS_URL           — Redis connection string (default: redis://localhost:6379/0)
    EVENT_BUS_BACKEND   — "redis" | "memory" (default: memory)

Classes:
    RedisEventBus: Redis-backed replacement for EventBus.

Example:
    >>> bus = RedisEventBus.create()
    >>> bus.subscribe(EventType.REGIME_CHANGE, handle_regime)
    >>> bus.publish(Event(EventType.REGIME_CHANGE, {"regime": 2}))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from src.realtime.event_bus import Event, EventBus, EventType, Handler

logger = logging.getLogger(__name__)

# Redis channel name
_CHANNEL = "amrcais:events"


class RedisEventBus(EventBus):
    """EventBus backed by Redis Pub/Sub.

    Publishes events to both local subscribers (in-process) and a Redis
    channel, allowing multiple process instances to share events.

    When Redis is unavailable, behaviour degrades gracefully to the
    in-process EventBus — callers need not change anything.

    Args:
        redis_url: Redis connection string.
        channel: Redis Pub/Sub channel name.
        max_history: In-process event history size.

    Example:
        >>> bus = RedisEventBus("redis://localhost:6379/0")
        >>> bus.start_listener()
        >>> bus.publish(Event(EventType.REGIME_CHANGE, {"regime": 2}))
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        channel: str = _CHANNEL,
        max_history: int = 1000,
    ) -> None:
        super().__init__(max_history=max_history)
        self._redis_url = redis_url
        self._channel = channel
        self._redis = None
        self._pubsub = None
        self._listener_thread: Optional[threading.Thread] = None
        self._is_listening = False
        self._connect()

    # ── Connection ────────────────────────────────────────────

    def _connect(self) -> None:
        """Establish Redis connection (non-fatal on failure)."""
        try:
            import redis
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                retry_on_timeout=True,
            )
            self._redis.ping()
            logger.info(f"Redis EventBus connected to {self._redis_url}")
        except ImportError:
            logger.warning(
                "redis package not installed — using in-process bus. "
                "Install with: pip install redis"
            )
            self._redis = None
        except Exception as exc:
            logger.warning(
                f"Redis connection failed ({exc}); falling back to in-process bus"
            )
            self._redis = None

    @property
    def is_redis_connected(self) -> bool:
        """Whether Redis is available."""
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False

    # ── Publishing ────────────────────────────────────────────

    def publish(self, event: Event) -> int:
        """Publish event locally and to Redis.

        Args:
            event: Event to publish.

        Returns:
            Number of local handlers that processed the event.
        """
        # Always dispatch to local handlers first
        handlers_called = super().publish(event)

        # Also broadcast via Redis if available
        if self._redis is not None:
            try:
                payload = json.dumps(event.to_dict())
                self._redis.publish(self._channel, payload)
            except Exception as exc:
                logger.warning(f"Redis publish failed: {exc}")

        return handlers_called

    async def publish_async(self, event: Event) -> int:
        """Async publish — local + Redis."""
        handlers_called = await super().publish_async(event)

        if self._redis is not None:
            try:
                payload = json.dumps(event.to_dict())
                # Run blocking Redis publish in executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, self._redis.publish, self._channel, payload
                )
            except Exception as exc:
                logger.warning(f"Redis async publish failed: {exc}")

        return handlers_called

    # ── Redis Listener ────────────────────────────────────────

    def start_listener(self) -> None:
        """Start background thread that listens for Redis Pub/Sub messages.

        Messages from other processes are dispatched to local handlers.
        """
        if self._redis is None or self._is_listening:
            return

        self._pubsub = self._redis.pubsub()
        self._pubsub.subscribe(self._channel)

        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="RedisEventBusListener",
        )
        self._is_listening = True
        self._listener_thread.start()
        logger.info("Redis EventBus listener started")

    def stop_listener(self) -> None:
        """Stop the Redis listener thread."""
        self._is_listening = False
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception:
                pass
        logger.info("Redis EventBus listener stopped")

    def _listen_loop(self) -> None:
        """Background thread: consume Redis messages and dispatch."""
        while self._is_listening:
            try:
                message = self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message and message["type"] == "message":
                    self._handle_redis_message(message["data"])
            except Exception as exc:
                if self._is_listening:
                    logger.warning(f"Redis listener error: {exc}")
                    # Brief back-off before retry
                    import time
                    time.sleep(1.0)

    def _handle_redis_message(self, data: str) -> None:
        """Deserialize a Redis message into an Event and dispatch locally.

        Args:
            data: JSON-serialized event dict.
        """
        try:
            payload = json.loads(data)
            event_type = EventType(payload["event_type"])
            event = Event(
                event_type=event_type,
                data=payload.get("data", {}),
                timestamp=datetime.fromisoformat(payload["timestamp"]),
                event_id=payload.get("event_id", ""),
                source=payload.get("source", "redis"),
            )
            # Dispatch to local handlers only (don't re-publish to Redis)
            super().publish(event)
        except Exception as exc:
            logger.debug(f"Ignoring malformed Redis event: {exc}")

    # ── Factory ───────────────────────────────────────────────

    @staticmethod
    def create(
        redis_url: Optional[str] = None,
        max_history: int = 1000,
    ) -> EventBus:
        """Factory: create the correct EventBus based on configuration.

        Reads EVENT_BUS_BACKEND from environment:
            - "redis" → RedisEventBus with auto-listener
            - "memory" (default) → plain EventBus

        Args:
            redis_url: Override Redis URL (or read from REDIS_URL env var).
            max_history: Event history limit.

        Returns:
            EventBus instance (Redis-backed or in-process).
        """
        backend = os.getenv("EVENT_BUS_BACKEND", "memory").lower()

        if backend == "redis":
            url = redis_url or os.getenv(
                "REDIS_URL", "redis://localhost:6379/0"
            )
            bus = RedisEventBus(
                redis_url=url,
                max_history=max_history,
            )
            if bus.is_redis_connected:
                bus.start_listener()
                return bus
            else:
                logger.warning(
                    "Redis requested but unavailable; falling back to memory"
                )
                return EventBus(max_history=max_history)

        return EventBus(max_history=max_history)

    # ── Cleanup ───────────────────────────────────────────────

    def reset(self) -> None:
        """Stop listener and reset all state."""
        self.stop_listener()
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        super().reset()
