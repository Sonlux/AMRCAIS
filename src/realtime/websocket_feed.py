"""
WebSocket data feed for AMRCAIS (Phase 4.1 — Live Data).

Connects to Polygon.io or Alpaca WebSocket APIs to receive real-time
market data and publish updates to the EventBus.

Supported providers:
    - Polygon.io  (POLYGON_API_KEY)
    - Alpaca      (ALPACA_API_KEY + ALPACA_SECRET_KEY)
    - Simulated   (for development / testing)

Configuration via environment variables:
    WS_PROVIDER         — "polygon" | "alpaca" | "simulated" (default: simulated)
    POLYGON_API_KEY     — Polygon.io API key
    ALPACA_API_KEY      — Alpaca API key
    ALPACA_SECRET_KEY   — Alpaca secret key
    WS_SYMBOLS          — Comma-separated symbols (default: SPY,TLT,GLD,VIX)

Architecture:
    ┌──────────────┐       ┌──────────────┐       ┌────────────┐
    │ Polygon.io / │  WS   │  WebSocket   │ Event │  EventBus  │
    │ Alpaca API   │──────→│  Feed        │──────→│  → Sched   │
    └──────────────┘       │  (reconnect) │       │  → Alert   │
                           └──────────────┘       │  → Stream  │
                                                  └────────────┘

Classes:
    WebSocketFeed: Manages WebSocket connections and event publishing.

Example:
    >>> feed = WebSocketFeed(event_bus)
    >>> await feed.start()
    >>> # Data flows automatically to EventBus → subscribers
    >>> await feed.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class WSProvider(str, Enum):
    """Supported WebSocket data providers."""
    POLYGON = "polygon"
    ALPACA = "alpaca"
    SIMULATED = "simulated"


@dataclass
class QuoteUpdate:
    """Normalized market data quote.

    Attributes:
        symbol: Ticker symbol.
        price: Last trade price.
        volume: Trade volume.
        bid: Current bid price.
        ask: Current ask price.
        timestamp: Quote timestamp (UTC).
        source: Data provider name.
    """
    symbol: str
    price: float
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": round(self.price, 4),
            "volume": self.volume,
            "bid": round(self.bid, 4),
            "ask": round(self.ask, 4),
            "timestamp": self.timestamp,
            "source": self.source,
        }


class WebSocketFeed:
    """Manages WebSocket connections to market data providers.

    Handles connection, reconnection with exponential backoff,
    message parsing, normalization, and event bus publishing.

    Args:
        event_bus: EventBus to publish DATA_UPDATED events to.
        provider: Data provider ("polygon", "alpaca", "simulated").
        symbols: List of symbols to subscribe to.
        reconnect_max_delay: Maximum reconnection delay (seconds).

    Example:
        >>> feed = WebSocketFeed(bus, provider="polygon", symbols=["SPY", "TLT"])
        >>> await feed.start()
    """

    # Default assets tracked by AMRCAIS
    DEFAULT_SYMBOLS = ["SPY", "TLT", "GLD", "VIX"]

    # Provider WebSocket URLs
    WS_URLS = {
        WSProvider.POLYGON: "wss://socket.polygon.io/stocks",
        WSProvider.ALPACA: "wss://stream.data.alpaca.markets/v2/iex",
        WSProvider.SIMULATED: None,  # No real connection
    }

    def __init__(
        self,
        event_bus: EventBus,
        provider: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        reconnect_max_delay: float = 60.0,
    ) -> None:
        self._bus = event_bus
        self._provider = WSProvider(
            provider or os.getenv("WS_PROVIDER", "simulated")
        )
        self._symbols = symbols or os.getenv(
            "WS_SYMBOLS", ",".join(self.DEFAULT_SYMBOLS)
        ).split(",")
        self._reconnect_max_delay = reconnect_max_delay

        # State
        self._ws = None
        self._is_running = False
        self._reconnect_attempts = 0
        self._last_quote: Dict[str, QuoteUpdate] = {}
        self._message_count = 0
        self._connected_since: Optional[str] = None

        # Callbacks for custom processing
        self._quote_callbacks: List[Callable[[QuoteUpdate], None]] = []

        # API keys
        self._polygon_key = os.getenv("POLYGON_API_KEY", "")
        self._alpaca_key = os.getenv("ALPACA_API_KEY", "")
        self._alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket feed.

        Connects to the provider and begins receiving data.
        Runs indefinitely until stop() is called.
        """
        self._is_running = True
        logger.info(
            f"Starting WebSocket feed: provider={self._provider.value}, "
            f"symbols={self._symbols}"
        )

        while self._is_running:
            try:
                if self._provider == WSProvider.SIMULATED:
                    await self._run_simulated()
                elif self._provider == WSProvider.POLYGON:
                    await self._run_polygon()
                elif self._provider == WSProvider.ALPACA:
                    await self._run_alpaca()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if not self._is_running:
                    break
                delay = self._reconnect_delay()
                logger.warning(
                    f"WebSocket disconnected ({exc}); "
                    f"reconnecting in {delay:.1f}s "
                    f"(attempt {self._reconnect_attempts})"
                )
                await asyncio.sleep(delay)

    async def stop(self) -> None:
        """Stop the WebSocket feed gracefully."""
        self._is_running = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected_since = None
        logger.info("WebSocket feed stopped")

    def add_quote_callback(self, callback: Callable[[QuoteUpdate], None]) -> None:
        """Register a callback for every incoming quote.

        Args:
            callback: Function(QuoteUpdate) → None.
        """
        self._quote_callbacks.append(callback)

    # ── Provider Implementations ──────────────────────────────

    async def _run_polygon(self) -> None:
        """Connect to Polygon.io WebSocket and stream data."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError(
                "websockets package required. Install with: pip install websockets"
            )

        if not self._polygon_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        url = self.WS_URLS[WSProvider.POLYGON]
        async with websockets.connect(url) as ws:
            self._ws = ws
            self._connected_since = datetime.now(timezone.utc).isoformat()
            self._reconnect_attempts = 0

            # Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self._polygon_key}))
            auth_response = await ws.recv()
            logger.info(f"Polygon auth response: {auth_response}")

            # Subscribe to trades
            subscribe_msg = {
                "action": "subscribe",
                "params": ",".join(f"T.{s}" for s in self._symbols),
            }
            await ws.send(json.dumps(subscribe_msg))

            # Process messages
            async for message in ws:
                if not self._is_running:
                    break
                self._process_polygon_message(message)

    async def _run_alpaca(self) -> None:
        """Connect to Alpaca WebSocket and stream data."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError(
                "websockets package required. Install with: pip install websockets"
            )

        if not self._alpaca_key or not self._alpaca_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required"
            )

        url = self.WS_URLS[WSProvider.ALPACA]
        async with websockets.connect(url) as ws:
            self._ws = ws
            self._connected_since = datetime.now(timezone.utc).isoformat()
            self._reconnect_attempts = 0

            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self._alpaca_key,
                "secret": self._alpaca_secret,
            }
            await ws.send(json.dumps(auth_msg))
            auth_response = await ws.recv()
            logger.info(f"Alpaca auth response: {auth_response}")

            # Subscribe to trades
            sub_msg = {
                "action": "subscribe",
                "trades": self._symbols,
            }
            await ws.send(json.dumps(sub_msg))

            # Process messages
            async for message in ws:
                if not self._is_running:
                    break
                self._process_alpaca_message(message)

    async def _run_simulated(self) -> None:
        """Generate simulated price updates for development.

        Produces synthetic quotes every 5 seconds with realistic
        random walk dynamics.
        """
        self._connected_since = datetime.now(timezone.utc).isoformat()
        self._reconnect_attempts = 0

        # Base prices for simulation
        base_prices = {
            "SPY": 520.0,
            "TLT": 95.0,
            "GLD": 195.0,
            "VIX": 18.0,
            "IEF": 98.0,
            "DXY": 104.0,
        }

        # Initialize with base prices
        prices = {s: base_prices.get(s, 100.0) for s in self._symbols}

        logger.info("Simulated WebSocket feed active (5s interval)")

        while self._is_running:
            for symbol in self._symbols:
                # Random walk with mean reversion
                base = base_prices.get(symbol, 100.0)
                current = prices[symbol]
                drift = 0.0001 * (base - current)  # mean reversion
                shock = random.gauss(0, 0.002) * current  # random shock
                new_price = max(0.01, current + drift + shock)
                prices[symbol] = new_price

                quote = QuoteUpdate(
                    symbol=symbol,
                    price=new_price,
                    volume=random.randint(100, 10000),
                    bid=new_price * 0.9999,
                    ask=new_price * 1.0001,
                    source="simulated",
                )
                self._dispatch_quote(quote)

            await asyncio.sleep(5.0)

    # ── Message Processing ────────────────────────────────────

    def _process_polygon_message(self, raw: str) -> None:
        """Parse and dispatch a Polygon.io WebSocket message.

        Args:
            raw: Raw JSON message string.
        """
        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                ev_type = msg.get("ev")
                if ev_type == "T":  # Trade
                    quote = QuoteUpdate(
                        symbol=msg.get("sym", ""),
                        price=float(msg.get("p", 0)),
                        volume=float(msg.get("s", 0)),
                        timestamp=datetime.fromtimestamp(
                            msg.get("t", 0) / 1e9, tz=timezone.utc
                        ).isoformat(),
                        source="polygon",
                    )
                    self._dispatch_quote(quote)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug(f"Polygon message parse error: {exc}")

    def _process_alpaca_message(self, raw: str) -> None:
        """Parse and dispatch an Alpaca WebSocket message.

        Args:
            raw: Raw JSON message string.
        """
        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                msg_type = msg.get("T")
                if msg_type == "t":  # Trade
                    quote = QuoteUpdate(
                        symbol=msg.get("S", ""),
                        price=float(msg.get("p", 0)),
                        volume=float(msg.get("s", 0)),
                        timestamp=msg.get("t", datetime.now(timezone.utc).isoformat()),
                        source="alpaca",
                    )
                    self._dispatch_quote(quote)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug(f"Alpaca message parse error: {exc}")

    def _dispatch_quote(self, quote: QuoteUpdate) -> None:
        """Dispatch a normalized quote to EventBus and callbacks.

        Args:
            quote: Normalized quote update.
        """
        self._message_count += 1
        self._last_quote[quote.symbol] = quote

        # Publish to EventBus
        event = Event(
            event_type=EventType.DATA_UPDATED,
            data=quote.to_dict(),
            source=f"ws_{self._provider.value}",
        )
        self._bus.publish(event)

        # Fire custom callbacks
        for cb in self._quote_callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.debug(f"Quote callback error: {exc}")

    # ── Reconnection ──────────────────────────────────────────

    def _reconnect_delay(self) -> float:
        """Compute exponential backoff delay with jitter.

        Returns:
            Delay in seconds.
        """
        self._reconnect_attempts += 1
        base = min(2 ** self._reconnect_attempts, self._reconnect_max_delay)
        jitter = random.uniform(0, base * 0.3)
        return base + jitter

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get feed status and statistics.

        Returns:
            Dict with connection state, message counts, and last quotes.
        """
        return {
            "provider": self._provider.value,
            "is_running": self._is_running,
            "connected_since": self._connected_since,
            "symbols": self._symbols,
            "message_count": self._message_count,
            "reconnect_attempts": self._reconnect_attempts,
            "last_quotes": {
                sym: q.to_dict() for sym, q in self._last_quote.items()
            },
        }
