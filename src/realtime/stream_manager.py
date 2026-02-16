"""
Server-Sent Events (SSE) stream manager for AMRCAIS (Phase 4.1).

Bridges the EventBus to HTTP clients via SSE.  Each connected client
receives a filtered, real-time stream of regime updates, alerts, and
analysis results.

Architecture:
    EventBus  →  StreamManager  →  asyncio.Queue per client  →  SSE response

Features:
    - Per-client asyncio queues with back-pressure (max 100 pending)
    - Configurable event-type filtering per client
    - Automatic client cleanup on disconnect
    - Connection metrics (connected, total served, events broadcast)
    - Heartbeat keep-alive every 30 s

Classes:
    ClientConnection: Tracks one SSE client.
    StreamManager: Manages all SSE client connections.

Example:
    >>> manager = StreamManager(event_bus)
    >>> manager.start()
    >>> async for chunk in manager.subscribe(client_id="dash-1"):
    ...     yield chunk        # send as SSE in FastAPI StreamingResponse
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


@dataclass
class ClientConnection:
    """Represents a connected SSE client.

    Args:
        client_id: Unique connection identifier.
        queue: asyncio.Queue feeding the SSE response.
        event_filter: Set of EventTypes to forward (None = all).
        connected_at: Epoch time of connection.
    """

    client_id: str
    queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=100)
    )
    event_filter: Optional[Set[EventType]] = None
    connected_at: float = field(default_factory=time.time)
    events_sent: int = 0


class StreamManager:
    """Manages SSE client connections and broadcasts EventBus events.

    The manager subscribes to *all* EventBus events and fans them out
    to connected clients, respecting per-client event filters.

    Args:
        event_bus: EventBus to subscribe to.
        heartbeat_interval: Seconds between keep-alive pings.

    Example:
        >>> manager = StreamManager(bus)
        >>> manager.start()
        >>> # In a FastAPI route:
        >>> async def sse_endpoint():
        ...     return StreamingResponse(
        ...         manager.subscribe("client-1"),
        ...         media_type="text/event-stream",
        ...     )
    """

    def __init__(
        self,
        event_bus: EventBus,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._bus = event_bus
        self._heartbeat_interval = heartbeat_interval
        self._clients: Dict[str, ClientConnection] = {}
        self._is_running = False
        self._total_events_broadcast = 0
        self._total_connections = 0

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to EventBus and begin broadcasting."""
        if self._is_running:
            return
        self._bus.subscribe_all(self._on_event)
        self._is_running = True
        logger.info("Stream manager started")

    def stop(self) -> None:
        """Unsubscribe and disconnect all clients."""
        self._bus.unsubscribe_all(self._on_event)
        # Signal all client queues to stop
        for client in self._clients.values():
            try:
                client.queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._clients.clear()
        self._is_running = False
        logger.info("Stream manager stopped")

    @property
    def is_running(self) -> bool:
        """Whether the manager is actively broadcasting."""
        return self._is_running

    @property
    def connected_clients(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    # ── Event handler (receives all bus events) ───────────────

    def _on_event(self, event: Event) -> None:
        """Forward event to all matching client queues."""
        data = self._format_sse(event)
        disconnected: List[str] = []

        for client_id, client in self._clients.items():
            # Apply per-client filter
            if (
                client.event_filter is not None
                and event.event_type not in client.event_filter
            ):
                continue
            try:
                client.queue.put_nowait(data)
                client.events_sent += 1
                self._total_events_broadcast += 1
            except asyncio.QueueFull:
                logger.warning(
                    f"Client {client_id} queue full — dropping event"
                )
            except Exception:
                disconnected.append(client_id)

        for cid in disconnected:
            self._remove_client(cid)

    @staticmethod
    def _format_sse(event: Event) -> str:
        """Format an Event as an SSE message string."""
        payload = json.dumps(event.to_dict(), default=str)
        return (
            f"event: {event.event_type.value}\n"
            f"id: {event.event_id}\n"
            f"data: {payload}\n\n"
        )

    # ── Client subscription ───────────────────────────────────

    async def subscribe(
        self,
        client_id: Optional[str] = None,
        event_filter: Optional[Set[EventType]] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted strings for a connected client.

        This is an async generator intended to be used as the body
        of a ``StreamingResponse``.

        Args:
            client_id: Unique client ID (auto-generated if None).
            event_filter: Set of EventTypes to forward (None = all).

        Yields:
            SSE-formatted strings (event + data + newlines).
        """
        if client_id is None:
            client_id = uuid.uuid4().hex[:12]

        client = ClientConnection(
            client_id=client_id,
            event_filter=event_filter,
        )
        self._clients[client_id] = client
        self._total_connections += 1
        logger.info(
            f"SSE client connected: {client_id} "
            f"(filter={[e.value for e in event_filter] if event_filter else 'all'})"
        )

        # Initial connection event
        yield (
            f"event: connected\n"
            f"data: {{\"client_id\": \"{client_id}\"}}\n\n"
        )

        try:
            while self._is_running:
                try:
                    msg = await asyncio.wait_for(
                        client.queue.get(),
                        timeout=self._heartbeat_interval,
                    )
                    if msg is None:
                        # Shutdown signal
                        break
                    yield msg
                except asyncio.TimeoutError:
                    # Send heartbeat keep-alive
                    yield ": heartbeat\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            self._remove_client(client_id)
            logger.info(f"SSE client disconnected: {client_id}")

    def _remove_client(self, client_id: str) -> None:
        """Remove a client connection."""
        self._clients.pop(client_id, None)

    def disconnect_client(self, client_id: str) -> bool:
        """Explicitly disconnect a client.

        Args:
            client_id: Client to disconnect.

        Returns:
            True if the client was found and disconnected.
        """
        client = self._clients.get(client_id)
        if client is None:
            return False
        try:
            client.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        self._remove_client(client_id)
        return True

    # ── Broadcast (manual) ────────────────────────────────────

    def broadcast(self, event: Event) -> int:
        """Manually broadcast an event to all clients.

        This bypasses the EventBus; useful for custom messages.

        Args:
            event: Event to broadcast.

        Returns:
            Number of clients that received the event.
        """
        self._on_event(event)
        return len(self._clients)

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return stream manager status."""
        return {
            "is_running": self._is_running,
            "connected_clients": len(self._clients),
            "total_connections": self._total_connections,
            "total_events_broadcast": self._total_events_broadcast,
            "clients": [
                {
                    "client_id": c.client_id,
                    "connected_at": c.connected_at,
                    "events_sent": c.events_sent,
                    "filter": (
                        [e.value for e in c.event_filter]
                        if c.event_filter
                        else None
                    ),
                }
                for c in self._clients.values()
            ],
        }
