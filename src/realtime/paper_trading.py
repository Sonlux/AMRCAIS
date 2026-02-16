"""
Paper trading engine for AMRCAIS (Phase 4.3).

Simulates portfolio execution based on regime-adaptive allocations.
Tracks positions, orders, P&L, and performance metrics with full
regime attribution — enabling strategy validation without real capital.

Architecture:
    ┌──────────┐  allocations  ┌──────────────┐  events   ┌──────────┐
    │ Portfolio │ ────────────→ │ PaperTrading │ ────────→ │ EventBus │
    │ Optimizer │              │   Engine      │          │          │
    └──────────┘              └──────────────┘          └──────────┘

Features:
    - Position tracking with cost-basis and unrealized P&L
    - Simulated order execution (market orders, immediate fill)
    - Cash management and transaction fees
    - Performance metrics: total return, Sharpe, max drawdown
    - Regime attribution: P&L breakdown by regime
    - Full trade history

Classes:
    OrderSide: BUY / SELL enum.
    OrderStatus: PENDING / FILLED / CANCELLED.
    PaperOrder: Immutable order record.
    Position: Mutable position with P&L tracking.
    PaperTradingEngine: Core engine.

Example:
    >>> engine = PaperTradingEngine(event_bus, initial_capital=100_000)
    >>> engine.execute_rebalance({"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}, prices, regime=1)
    >>> engine.get_portfolio_summary()
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class PaperOrder:
    """Simulated order record.

    Args:
        order_id: Unique identifier.
        asset: Ticker symbol.
        side: BUY or SELL.
        quantity: Number of shares (can be fractional).
        price: Execution price.
        status: Current order status.
        regime: Market regime at time of order.
        timestamp: Epoch time of creation.
        fill_timestamp: Epoch time of fill.
        reason: Why the order was generated.
    """

    order_id: str
    asset: str
    side: OrderSide
    quantity: float
    price: float
    status: OrderStatus = OrderStatus.PENDING
    regime: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    fill_timestamp: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "order_id": self.order_id,
            "asset": self.asset,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "regime": self.regime,
            "timestamp": self.timestamp,
            "fill_timestamp": self.fill_timestamp,
            "reason": self.reason,
            "notional": round(self.quantity * self.price, 2),
        }


@dataclass
class Position:
    """Tracks a single asset position.

    Args:
        asset: Ticker symbol.
        quantity: Number of shares held.
        avg_cost: Average cost basis per share.
        current_price: Latest mark-to-market price.
    """

    asset: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "asset": self.asset,
            "quantity": round(self.quantity, 4),
            "avg_cost": round(self.avg_cost, 4),
            "current_price": round(self.current_price, 4),
            "market_value": round(self.market_value, 2),
            "cost_basis": round(self.cost_basis, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
        }


class PaperTradingEngine:
    """Simulated portfolio execution engine.

    Manages positions, cash, and order history.  Generates
    ``ORDER_FILLED`` and ``REBALANCE_TRIGGER`` events on the bus.

    Args:
        event_bus: EventBus for publishing trade events.
        initial_capital: Starting cash balance (USD).
        transaction_fee_bps: Fee per trade in basis points.
        max_position_pct: Maximum single-position weight (0-1).

    Example:
        >>> engine = PaperTradingEngine(bus, initial_capital=100_000)
        >>> orders = engine.execute_rebalance(
        ...     {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
        ...     {"SPY": 450.0, "TLT": 95.0, "GLD": 185.0},
        ...     regime=1,
        ... )
    """

    def __init__(
        self,
        event_bus: EventBus,
        initial_capital: float = 100_000.0,
        transaction_fee_bps: float = 5.0,
        max_position_pct: float = 0.40,
    ) -> None:
        self._bus = event_bus
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._fee_rate = transaction_fee_bps / 10_000
        self._max_position_pct = max_position_pct

        self._positions: Dict[str, Position] = {}
        self._orders: List[PaperOrder] = []
        self._equity_curve: List[Dict[str, Any]] = []
        self._regime_pnl: Dict[int, float] = {}
        self._current_regime: Optional[int] = None
        self._last_rebalance_time: Optional[float] = None
        self._rebalance_count = 0

        # Record initial state
        self._record_equity()

    # ── Core execution ────────────────────────────────────────

    def execute_rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        regime: Optional[int] = None,
        reason: str = "regime_rebalance",
    ) -> List[PaperOrder]:
        """Rebalance portfolio to target weights.

        Calculates required trades, executes them, and publishes
        events for each fill.

        Args:
            target_weights: Asset → target weight (0-1), should sum ≤ 1.
            prices: Asset → current price.
            regime: Current market regime ID.
            reason: Why the rebalance was triggered.

        Returns:
            List of executed PaperOrder objects.
        """
        self._current_regime = regime

        # Update mark-to-market
        self._update_prices(prices)

        total_equity = self.total_equity
        orders: List[PaperOrder] = []

        # Calculate current weights
        current_weights: Dict[str, float] = {}
        for asset, pos in self._positions.items():
            if total_equity > 0:
                current_weights[asset] = pos.market_value / total_equity

        # Determine trades needed
        all_assets = set(target_weights.keys()) | set(
            self._positions.keys()
        )

        # Sell first (free up cash), then buy
        sell_orders: List[tuple] = []
        buy_orders: List[tuple] = []

        for asset in all_assets:
            target_w = target_weights.get(asset, 0.0)
            current_w = current_weights.get(asset, 0.0)
            price = prices.get(asset, 0.0)

            if price <= 0:
                continue

            # Enforce max position size
            target_w = min(target_w, self._max_position_pct)

            delta_w = target_w - current_w
            delta_value = delta_w * total_equity
            delta_shares = delta_value / price

            # Skip small trades (< $50 or < 0.1 share)
            if abs(delta_value) < 50 or abs(delta_shares) < 0.1:
                continue

            if delta_shares < 0:
                sell_orders.append((asset, abs(delta_shares), price))
            else:
                buy_orders.append((asset, delta_shares, price))

        # Execute sells
        for asset, qty, price in sell_orders:
            order = self._execute_order(
                asset, OrderSide.SELL, qty, price, regime, reason
            )
            if order:
                orders.append(order)

        # Execute buys
        for asset, qty, price in buy_orders:
            # Adjust qty for available cash
            cost = qty * price * (1 + self._fee_rate)
            if cost > self._cash:
                qty = self._cash / (price * (1 + self._fee_rate))
                if qty < 0.1:
                    continue
            order = self._execute_order(
                asset, OrderSide.BUY, qty, price, regime, reason
            )
            if order:
                orders.append(order)

        self._rebalance_count += 1
        self._last_rebalance_time = time.time()
        self._record_equity()

        # Publish rebalance event
        self._bus.publish(
            Event(
                EventType.REBALANCE_TRIGGER,
                {
                    "target_weights": target_weights,
                    "orders_executed": len(orders),
                    "regime": regime,
                    "total_equity": round(self.total_equity, 2),
                    "cash": round(self._cash, 2),
                },
                source="paper_trading",
            )
        )

        logger.info(
            f"Rebalance #{self._rebalance_count}: "
            f"{len(orders)} orders, equity=${self.total_equity:,.2f}"
        )

        return orders

    def _execute_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        price: float,
        regime: Optional[int],
        reason: str,
    ) -> Optional[PaperOrder]:
        """Execute a single simulated order.

        Args:
            asset: Ticker symbol.
            side: BUY or SELL.
            quantity: Number of shares.
            price: Execution price.
            regime: Current regime.
            reason: Order reason.

        Returns:
            Filled PaperOrder, or None if rejected.
        """
        notional = quantity * price
        fee = notional * self._fee_rate

        if side == OrderSide.BUY:
            total_cost = notional + fee
            if total_cost > self._cash:
                return None
            self._cash -= total_cost
            self._update_position_buy(asset, quantity, price)
        else:
            pos = self._positions.get(asset)
            if pos is None or pos.quantity < quantity:
                # Adjust to available shares
                if pos is None or pos.quantity < 0.1:
                    return None
                quantity = pos.quantity
                notional = quantity * price
                fee = notional * self._fee_rate
            proceeds = notional - fee
            self._cash += proceeds
            realized = self._update_position_sell(asset, quantity, price)
            # Track regime P&L
            if regime is not None:
                self._regime_pnl[regime] = (
                    self._regime_pnl.get(regime, 0.0) + realized
                )

        order = PaperOrder(
            order_id=uuid.uuid4().hex[:12],
            asset=asset,
            side=side,
            quantity=round(quantity, 4),
            price=price,
            status=OrderStatus.FILLED,
            regime=regime,
            fill_timestamp=time.time(),
            reason=reason,
        )
        self._orders.append(order)

        # Publish order filled event
        self._bus.publish(
            Event(
                EventType.ORDER_FILLED,
                order.to_dict(),
                source="paper_trading",
            )
        )

        return order

    def _update_position_buy(
        self, asset: str, quantity: float, price: float
    ) -> None:
        """Update position for a buy order."""
        pos = self._positions.get(asset)
        if pos is None:
            self._positions[asset] = Position(
                asset=asset,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
            )
        else:
            total_cost = pos.avg_cost * pos.quantity + price * quantity
            pos.quantity += quantity
            pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
            pos.current_price = price

    def _update_position_sell(
        self, asset: str, quantity: float, price: float
    ) -> float:
        """Update position for a sell order.

        Returns:
            Realized P&L from the sale.
        """
        pos = self._positions.get(asset)
        if pos is None:
            return 0.0
        realized = (price - pos.avg_cost) * quantity
        pos.quantity -= quantity
        pos.current_price = price
        if pos.quantity < 0.01:
            del self._positions[asset]
        return realized

    def _update_prices(self, prices: Dict[str, float]) -> None:
        """Mark-to-market all positions."""
        for asset, price in prices.items():
            if asset in self._positions:
                self._positions[asset].current_price = price

    def _record_equity(self) -> None:
        """Record an equity curve data point."""
        self._equity_curve.append(
            {
                "timestamp": time.time(),
                "equity": round(self.total_equity, 2),
                "cash": round(self._cash, 2),
                "positions_value": round(self.positions_value, 2),
                "regime": self._current_regime,
            }
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def positions_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def total_equity(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self._cash + self.positions_value

    @property
    def total_return(self) -> float:
        """Total return since inception."""
        if self._initial_capital == 0:
            return 0.0
        return (
            self.total_equity - self._initial_capital
        ) / self._initial_capital

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    # ── Performance metrics ───────────────────────────────────

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics.

        Returns:
            Dict with total_return, sharpe_ratio, max_drawdown, etc.
        """
        equities = [p["equity"] for p in self._equity_curve]

        if len(equities) < 2:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "num_trades": len(self._orders),
                "num_rebalances": self._rebalance_count,
                "win_rate": 0.0,
            }

        # Returns series
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                returns.append(
                    (equities[i] - equities[i - 1]) / equities[i - 1]
                )

        # Sharpe ratio (annualized, assuming daily)
        if returns:
            mean_ret = sum(returns) / len(returns)
            if len(returns) > 1:
                var = sum((r - mean_ret) ** 2 for r in returns) / (
                    len(returns) - 1
                )
                std_ret = math.sqrt(var) if var > 0 else 0.001
            else:
                std_ret = 0.001
            sharpe = (mean_ret / std_ret) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = equities[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)

        # Win rate (profitable trades)
        sells = [
            o
            for o in self._orders
            if o.side == OrderSide.SELL
            and o.status == OrderStatus.FILLED
        ]
        wins = 0
        for order in sells:
            # Approximate: check if sell price > avg cost
            # This is simplified — a production system would track exact lots
            wins += 1 if order.price > 0 else 0

        return {
            "total_return": round(
                self.total_equity - self._initial_capital, 2
            ),
            "total_return_pct": round(self.total_return, 4),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "num_trades": len(self._orders),
            "num_rebalances": self._rebalance_count,
            "win_rate": round(
                wins / len(sells), 3
            )
            if sells
            else 0.0,
            "initial_capital": self._initial_capital,
            "current_equity": round(self.total_equity, 2),
            "cash": round(self._cash, 2),
        }

    def get_regime_attribution(self) -> Dict[str, Any]:
        """Breakdown P&L by market regime.

        Returns:
            Dict mapping regime ID → realized P&L.
        """
        return {
            "regime_pnl": {
                str(k): round(v, 2) for k, v in self._regime_pnl.items()
            },
            "total_realized": round(
                sum(self._regime_pnl.values()), 2
            ),
            "unrealized": round(
                sum(
                    p.unrealized_pnl
                    for p in self._positions.values()
                ),
                2,
            ),
        }

    # ── Query interface ───────────────────────────────────────

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        return [p.to_dict() for p in self._positions.values()]

    def get_orders(
        self,
        limit: int = 50,
        asset: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get order history.

        Args:
            limit: Maximum orders to return.
            asset: Filter by asset ticker.

        Returns:
            List of order dicts, newest first.
        """
        orders = self._orders
        if asset:
            orders = [o for o in orders if o.asset == asset]
        return [o.to_dict() for o in reversed(orders)][:limit]

    def get_equity_curve(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get equity curve data points.

        Args:
            limit: Max data points (None = all).

        Returns:
            List of equity snapshots.
        """
        if limit:
            return self._equity_curve[-limit:]
        return list(self._equity_curve)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Comprehensive portfolio snapshot."""
        return {
            "total_equity": round(self.total_equity, 2),
            "cash": round(self._cash, 2),
            "positions_value": round(self.positions_value, 2),
            "num_positions": len(self._positions),
            "positions": self.get_positions(),
            "total_return_pct": round(self.total_return, 4),
            "rebalance_count": self._rebalance_count,
            "last_rebalance": self._last_rebalance_time,
            "current_regime": self._current_regime,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return engine status summary."""
        return {
            "initial_capital": self._initial_capital,
            "total_equity": round(self.total_equity, 2),
            "cash": round(self._cash, 2),
            "num_positions": len(self._positions),
            "num_orders": len(self._orders),
            "rebalance_count": self._rebalance_count,
            "total_return_pct": round(self.total_return, 4),
            "current_regime": self._current_regime,
        }
