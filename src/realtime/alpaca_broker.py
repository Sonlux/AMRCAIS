"""
Alpaca Paper Trading Integration for AMRCAIS (Phase 4.3).

Connects the PaperTradingEngine to the Alpaca Markets paper-trading
API for real order execution in a simulated environment.

Configuration via environment variables:
    ALPACA_API_KEY      — Alpaca API key (paper)
    ALPACA_SECRET_KEY   — Alpaca secret key (paper)
    ALPACA_BASE_URL     — API base URL (default: https://paper-api.alpaca.markets)

Features:
    - Submit market / limit orders via Alpaca paper API
    - Query account balances, positions, and order history
    - Translate AMRCAIS portfolio allocations → Alpaca orders
    - Event-driven: publishes trade events to EventBus

Classes:
    AlpacaPaperBroker: Connects to Alpaca paper trading API.

Example:
    >>> broker = AlpacaPaperBroker()
    >>> broker.submit_order("SPY", qty=10, side="buy")
    >>> positions = broker.get_positions()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class AlpacaPaperBroker:
    """Alpaca Markets paper trading broker integration.

    Provides a clean interface for submitting orders, querying
    positions, and managing the paper trading portfolio through
    Alpaca's API.

    Args:
        api_key: Alpaca API key (or ALPACA_API_KEY env var).
        secret_key: Alpaca secret key (or ALPACA_SECRET_KEY env var).
        base_url: Alpaca API base URL.
        event_bus: Optional EventBus for trade event publishing.

    Example:
        >>> broker = AlpacaPaperBroker()
        >>> broker.submit_order("SPY", qty=10, side="buy")
        >>> print(broker.get_account()["equity"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._base_url = base_url or os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )
        self._bus = event_bus
        self._api = None
        self._connected = False

        self._connect()

    def _connect(self) -> None:
        """Establish connection to Alpaca API."""
        if not self._api_key or not self._secret_key:
            logger.warning(
                "Alpaca credentials not configured — broker in dry-run mode. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars."
            )
            return

        try:
            from alpaca.trading.client import TradingClient
            self._api = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=True,
            )
            # Verify connection
            account = self._api.get_account()
            self._connected = True
            logger.info(
                f"Alpaca paper broker connected. "
                f"Equity: ${float(account.equity):,.2f}, "
                f"Cash: ${float(account.cash):,.2f}"
            )
        except ImportError:
            logger.warning(
                "alpaca-py package not installed — broker in dry-run mode. "
                "Install with: pip install alpaca-py"
            )
        except Exception as exc:
            logger.error(f"Alpaca connection failed: {exc}")

    @property
    def is_connected(self) -> bool:
        """Whether the broker is connected to Alpaca."""
        return self._connected

    # ── Account ───────────────────────────────────────────────

    def get_account(self) -> Dict[str, Any]:
        """Get Alpaca account details.

        Returns:
            Dict with equity, cash, buying_power, etc.
        """
        if not self._connected:
            return self._dry_run_account()

        try:
            account = self._api.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "status": account.status,
                "pattern_day_trader": account.pattern_day_trader,
                "currency": account.currency,
            }
        except Exception as exc:
            logger.error(f"Failed to get account: {exc}")
            return {"error": str(exc)}

    # ── Orders ────────────────────────────────────────────────

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Submit an order to Alpaca paper trading.

        Args:
            symbol: Ticker symbol (e.g. "SPY").
            qty: Number of shares (supports fractional).
            side: "buy" or "sell".
            order_type: "market" or "limit".
            time_in_force: "day", "gtc", "ioc", "fok".
            limit_price: Required for limit orders.

        Returns:
            Dict with order details or error.
        """
        if not self._connected:
            return self._dry_run_order(symbol, qty, side, order_type)

        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
            )
            from alpaca.trading.enums import (
                OrderSide,
                TimeInForce,
            )

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }.get(time_in_force, TimeInForce.DAY)

            if order_type == "limit" and limit_price is not None:
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                )
            else:
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                )

            order = self._api.submit_order(request)

            result = {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "qty": float(order.qty) if order.qty else qty,
                "side": order.side.value,
                "type": order.type.value,
                "status": order.status.value,
                "submitted_at": str(order.submitted_at),
            }

            logger.info(
                f"Order submitted: {side.upper()} {qty} {symbol} "
                f"({order_type}) → {result['status']}"
            )

            # Publish to EventBus
            if self._bus:
                self._bus.publish(Event(
                    event_type=EventType.DATA_UPDATED,
                    data={"trade": result, "action": "order_submitted"},
                    source="alpaca_broker",
                ))

            return result

        except Exception as exc:
            logger.error(f"Order submission failed: {exc}")
            return {"error": str(exc), "symbol": symbol}

    def get_orders(
        self,
        status: str = "all",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get order history.

        Args:
            status: "all", "open", "closed".
            limit: Maximum orders to return.

        Returns:
            List of order dicts.
        """
        if not self._connected:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            status_map = {
                "all": QueryOrderStatus.ALL,
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
            }

            request = GetOrdersRequest(
                status=status_map.get(status, QueryOrderStatus.ALL),
                limit=limit,
            )
            orders = self._api.get_orders(request)

            return [
                {
                    "order_id": str(o.id),
                    "symbol": o.symbol,
                    "qty": float(o.qty) if o.qty else 0,
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "side": o.side.value,
                    "type": o.type.value,
                    "status": o.status.value,
                    "submitted_at": str(o.submitted_at),
                    "filled_at": str(o.filled_at) if o.filled_at else None,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                }
                for o in orders
            ]
        except Exception as exc:
            logger.error(f"Failed to get orders: {exc}")
            return []

    # ── Positions ─────────────────────────────────────────────

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current portfolio positions.

        Returns:
            List of position dicts.
        """
        if not self._connected:
            return []

        try:
            positions = self._api.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "current_price": float(p.current_price),
                    "side": p.side,
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error(f"Failed to get positions: {exc}")
            return []

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """Close an entire position for a symbol.

        Args:
            symbol: Ticker symbol to close.

        Returns:
            Order dict from the closing order.
        """
        if not self._connected:
            return {"error": "broker not connected", "symbol": symbol}

        try:
            order = self._api.close_position(symbol)
            result = {
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": order.side.value,
                "status": order.status.value,
            }
            logger.info(f"Closed position: {symbol}")
            return result
        except Exception as exc:
            logger.error(f"Failed to close position {symbol}: {exc}")
            return {"error": str(exc), "symbol": symbol}

    def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions.

        Returns:
            List of closing order dicts.
        """
        if not self._connected:
            return []

        try:
            results = self._api.close_all_positions(cancel_orders=True)
            logger.info(f"Closed all positions: {len(results)} orders")
            return [{"symbol": str(r)} for r in results]
        except Exception as exc:
            logger.error(f"Failed to close all positions: {exc}")
            return []

    # ── Portfolio Allocation ──────────────────────────────────

    def execute_allocation(
        self,
        target_weights: Dict[str, float],
        total_equity: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a portfolio allocation by converting target weights to orders.

        This is the key integration point: AMRCAIS PortfolioOptimizer
        outputs target weights → this method converts them to orders.

        Args:
            target_weights: Dict of symbol → target weight (0-1).
            total_equity: Portfolio value to allocate (default: account equity).

        Returns:
            List of order result dicts.
        """
        account = self.get_account()
        equity = total_equity or account.get("equity", 0)

        if equity <= 0:
            logger.warning("Cannot execute allocation: zero equity")
            return []

        # Get current positions
        current_positions = {
            p["symbol"]: p for p in self.get_positions()
        }

        orders = []

        for symbol, target_weight in target_weights.items():
            target_value = equity * target_weight
            current_value = current_positions.get(symbol, {}).get("market_value", 0)
            current_price = current_positions.get(symbol, {}).get("current_price")

            if current_price is None or current_price <= 0:
                logger.debug(f"Skipping {symbol}: no price data")
                continue

            diff_value = target_value - current_value
            diff_shares = abs(diff_value) / current_price

            # Skip tiny trades (< $100 or < 0.5 share)
            if abs(diff_value) < 100 or diff_shares < 0.5:
                continue

            side = "buy" if diff_value > 0 else "sell"
            qty = round(diff_shares, 2)

            result = self.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market",
            )
            orders.append(result)

        logger.info(
            f"Executed allocation: {len(orders)} orders for "
            f"{len(target_weights)} symbols"
        )
        return orders

    # ── Dry-Run Fallbacks ─────────────────────────────────────

    def _dry_run_account(self) -> Dict[str, Any]:
        """Simulated account for when Alpaca is not connected."""
        return {
            "equity": 100_000.0,
            "cash": 100_000.0,
            "buying_power": 200_000.0,
            "portfolio_value": 100_000.0,
            "status": "DRY_RUN",
            "note": "Alpaca not connected — simulated account",
        }

    def _dry_run_order(
        self, symbol: str, qty: float, side: str, order_type: str
    ) -> Dict[str, Any]:
        """Simulated order for when Alpaca is not connected."""
        import uuid
        result = {
            "order_id": uuid.uuid4().hex[:12],
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "status": "DRY_RUN",
            "submitted_at": datetime.now().isoformat(),
            "note": "Alpaca not connected — dry-run order",
        }
        logger.info(f"DRY-RUN order: {side.upper()} {qty} {symbol}")
        return result

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get broker connection status.

        Returns:
            Dict with connection state and account summary.
        """
        return {
            "connected": self._connected,
            "base_url": self._base_url,
            "account": self.get_account() if self._connected else self._dry_run_account(),
        }
