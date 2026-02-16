"""
Phase 4: Real-Time + Execution API routes.

Endpoints:
    GET  /api/phase4/status            — Phase 4 system status overview
    GET  /api/phase4/events            — Event bus history
    GET  /api/phase4/alerts            — Query alerts
    POST /api/phase4/alerts/acknowledge— Acknowledge an alert
    GET  /api/phase4/alerts/config     — Get alert configuration
    POST /api/phase4/alerts/config     — Update alert configuration
    GET  /api/phase4/stream            — SSE event stream
    GET  /api/phase4/stream/status     — Stream manager status
    GET  /api/phase4/portfolio         — Paper trading portfolio summary
    GET  /api/phase4/portfolio/metrics — Performance metrics
    GET  /api/phase4/portfolio/equity  — Equity curve
    GET  /api/phase4/portfolio/attribution — Regime P&L attribution
    GET  /api/phase4/trades            — Order history
    POST /api/phase4/rebalance         — Trigger portfolio rebalance
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.dependencies import get_system
from api.schemas import (
    AlertConfigItem,
    AlertConfigRequest,
    AlertConfigResponse,
    AlertResponse,
    AlertsResponse,
    EquityCurvePoint,
    EquityCurveResponse,
    EventResponse,
    EventsResponse,
    PaperOrderResponse,
    PerformanceMetricsResponse,
    Phase4StatusResponse,
    PortfolioSummaryResponse,
    PositionResponse,
    RebalanceRequest,
    RebalanceResponse,
    RegimeAttributionResponse,
    StreamStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────


def _get_event_bus():
    """Get the event bus from the system."""
    system = get_system()
    bus = getattr(system, "event_bus", None)
    if bus is None:
        raise HTTPException(
            status_code=503,
            detail="Event bus not available",
        )
    return bus


def _get_alert_engine():
    """Get the alert engine from the system."""
    system = get_system()
    engine = getattr(system, "alert_engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Alert engine not available",
        )
    return engine


def _get_stream_manager():
    """Get the stream manager from the system."""
    system = get_system()
    manager = getattr(system, "stream_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail="Stream manager not available",
        )
    return manager


def _get_paper_trading():
    """Get the paper trading engine from the system."""
    system = get_system()
    engine = getattr(system, "paper_trading", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Paper trading engine not available",
        )
    return engine


# ── System Status ─────────────────────────────────────────────────


@router.get("/status", response_model=Phase4StatusResponse)
async def phase4_status():
    """Phase 4 real-time system status overview.

    Returns:
        Combined status of event bus, scheduler, alert engine,
        stream manager, and paper trading engine.
    """
    system = get_system()

    result: Dict[str, Any] = {
        "event_bus": {},
        "scheduler": {},
        "alert_engine": {},
        "stream_manager": {},
        "paper_trading": {},
    }

    bus = getattr(system, "event_bus", None)
    if bus:
        result["event_bus"] = {
            "event_count": bus.event_count,
            "subscriber_count": bus.subscriber_count,
        }

    scheduler = getattr(system, "scheduler", None)
    if scheduler:
        result["scheduler"] = scheduler.get_status()

    alert_engine = getattr(system, "alert_engine", None)
    if alert_engine:
        result["alert_engine"] = alert_engine.get_status()

    stream_manager = getattr(system, "stream_manager", None)
    if stream_manager:
        result["stream_manager"] = stream_manager.get_status()

    paper_trading = getattr(system, "paper_trading", None)
    if paper_trading:
        result["paper_trading"] = paper_trading.get_status()

    return Phase4StatusResponse(**result)


# ── Event Bus ─────────────────────────────────────────────────────


@router.get("/events", response_model=EventsResponse)
async def get_events(
    event_type: Optional[str] = Query(
        None, description="Filter by event type"
    ),
    limit: int = Query(50, ge=1, le=500),
):
    """Query event bus history.

    Args:
        event_type: Optional event type filter.
        limit: Maximum events to return.

    Returns:
        List of events, newest first.
    """
    bus = _get_event_bus()

    from src.realtime.event_bus import EventType

    et = None
    if event_type:
        try:
            et = EventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown event type: {event_type}",
            )

    history = bus.get_history(event_type=et, limit=limit)
    events = [
        EventResponse(
            event_type=e.event_type.value,
            data=e.data,
            timestamp=e.timestamp.isoformat() if hasattr(e.timestamp, 'isoformat') else str(e.timestamp),
            event_id=e.event_id,
            source=e.source,
        )
        for e in history
    ]

    return EventsResponse(events=events, total=len(events))


# ── Alerts ────────────────────────────────────────────────────────


@router.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    alert_type: Optional[str] = Query(
        None, description="Filter by alert type"
    ),
    severity: Optional[str] = Query(
        None, description="Filter by severity"
    ),
    limit: int = Query(50, ge=1, le=500),
    unacknowledged_only: bool = Query(False),
):
    """Query alerts with optional filters.

    Args:
        alert_type: Alert category filter.
        severity: Severity level filter.
        limit: Maximum alerts to return.
        unacknowledged_only: Only return unacknowledged alerts.

    Returns:
        Filtered list of alerts.
    """
    engine = _get_alert_engine()

    from src.realtime.alert_engine import AlertSeverity, AlertType

    at = None
    if alert_type:
        try:
            at = AlertType(alert_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown alert type: {alert_type}",
            )

    sev = None
    if severity:
        try:
            sev = AlertSeverity(severity)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown severity: {severity}",
            )

    alerts_data = engine.get_alerts(
        alert_type=at,
        severity=sev,
        limit=limit,
        unacknowledged_only=unacknowledged_only,
    )

    status = engine.get_status()

    return AlertsResponse(
        alerts=[AlertResponse(**a) for a in alerts_data],
        total=status["total_alerts"],
        unacknowledged=status["unacknowledged"],
    )


@router.post("/alerts/acknowledge")
async def acknowledge_alert(alert_id: str = Query(...)):
    """Acknowledge an alert.

    Args:
        alert_id: ID of the alert to acknowledge.

    Returns:
        Success status.
    """
    engine = _get_alert_engine()
    success = engine.acknowledge_alert(alert_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alert not found: {alert_id}",
        )

    return {"status": "acknowledged", "alert_id": alert_id}


@router.get("/alerts/config", response_model=AlertConfigResponse)
async def get_alert_config():
    """Get current alert configuration for all alert types."""
    engine = _get_alert_engine()
    raw = engine.get_config()
    configs = {
        k: AlertConfigItem(**v) for k, v in raw.items()
    }
    return AlertConfigResponse(configs=configs)


@router.post("/alerts/config")
async def update_alert_config(request: AlertConfigRequest):
    """Update alert configuration for a specific type.

    Args:
        request: Alert type and new config values.

    Returns:
        Updated configuration for the alert type.
    """
    engine = _get_alert_engine()

    from src.realtime.alert_engine import AlertConfig, AlertType

    try:
        at = AlertType(request.alert_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown alert type: {request.alert_type}",
        )

    # Get current config and merge
    current = engine.get_config(at)
    current_vals = current.get(at.value, {})
    new_config = AlertConfig(
        enabled=(
            request.enabled
            if request.enabled is not None
            else current_vals.get("enabled", True)
        ),
        cooldown_seconds=(
            request.cooldown_seconds
            if request.cooldown_seconds is not None
            else current_vals.get("cooldown_seconds", 300)
        ),
        threshold=(
            request.threshold
            if request.threshold is not None
            else current_vals.get("threshold", 0.0)
        ),
    )

    engine.update_config(at, new_config)

    return {
        "status": "updated",
        "alert_type": at.value,
        "config": {
            "enabled": new_config.enabled,
            "cooldown_seconds": new_config.cooldown_seconds,
            "threshold": new_config.threshold,
        },
    }


# ── SSE Stream ────────────────────────────────────────────────────


@router.get("/stream")
async def sse_stream(
    client_id: Optional[str] = Query(None),
    filter: Optional[str] = Query(
        None,
        description="Comma-separated event types to subscribe to",
    ),
):
    """Server-Sent Events stream for real-time updates.

    Connect to receive live regime changes, alerts, and analysis results.

    Args:
        client_id: Optional client identifier.
        filter: Comma-separated list of event types to receive.

    Returns:
        SSE event stream.
    """
    manager = _get_stream_manager()

    from src.realtime.event_bus import EventType

    event_filter = None
    if filter:
        event_filter = set()
        for name in filter.split(","):
            try:
                event_filter.add(EventType(name.strip()))
            except ValueError:
                pass

    return StreamingResponse(
        manager.subscribe(
            client_id=client_id,
            event_filter=event_filter,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/stream/status", response_model=StreamStatusResponse)
async def stream_status():
    """Get SSE stream manager status."""
    manager = _get_stream_manager()
    status = manager.get_status()
    return StreamStatusResponse(**status)


# ── Paper Trading: Portfolio ──────────────────────────────────────


@router.get("/portfolio", response_model=PortfolioSummaryResponse)
async def get_portfolio():
    """Get current paper trading portfolio snapshot.

    Returns:
        Portfolio summary including positions, cash, and returns.
    """
    engine = _get_paper_trading()
    summary = engine.get_portfolio_summary()

    positions = [PositionResponse(**p) for p in summary["positions"]]

    return PortfolioSummaryResponse(
        total_equity=summary["total_equity"],
        cash=summary["cash"],
        positions_value=summary["positions_value"],
        num_positions=summary["num_positions"],
        positions=positions,
        total_return_pct=summary["total_return_pct"],
        rebalance_count=summary["rebalance_count"],
        last_rebalance=summary.get("last_rebalance"),
        current_regime=summary.get("current_regime"),
    )


@router.get("/portfolio/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """Get paper trading performance metrics.

    Returns:
        Total return, Sharpe ratio, max drawdown, win rate, etc.
    """
    engine = _get_paper_trading()
    metrics = engine.get_performance_metrics()
    return PerformanceMetricsResponse(**metrics)


@router.get("/portfolio/equity", response_model=EquityCurveResponse)
async def get_equity_curve(
    limit: Optional[int] = Query(None, ge=1, le=10000),
):
    """Get paper trading equity curve.

    Args:
        limit: Maximum data points to return.

    Returns:
        Time series of portfolio equity values.
    """
    engine = _get_paper_trading()
    curve = engine.get_equity_curve(limit=limit)
    points = [EquityCurvePoint(**p) for p in curve]
    return EquityCurveResponse(
        curve=points, total_points=len(points)
    )


@router.get(
    "/portfolio/attribution", response_model=RegimeAttributionResponse
)
async def get_regime_attribution():
    """Get P&L breakdown by market regime.

    Returns:
        Realized and unrealized P&L attributed to each regime.
    """
    engine = _get_paper_trading()
    attribution = engine.get_regime_attribution()
    return RegimeAttributionResponse(**attribution)


# ── Paper Trading: Orders ─────────────────────────────────────────


@router.get("/trades")
async def get_trades(
    limit: int = Query(50, ge=1, le=500),
    asset: Optional[str] = Query(None),
):
    """Get paper trade order history.

    Args:
        limit: Maximum orders to return.
        asset: Filter by asset ticker.

    Returns:
        List of orders, newest first.
    """
    engine = _get_paper_trading()
    orders = engine.get_orders(limit=limit, asset=asset)
    return {
        "orders": [PaperOrderResponse(**o) for o in orders],
        "total": len(orders),
    }


@router.post("/rebalance", response_model=RebalanceResponse)
async def trigger_rebalance(request: RebalanceRequest):
    """Trigger a paper portfolio rebalance.

    Executes simulated trades to reach target allocation.

    Args:
        request: Target weights, prices, and optional regime.

    Returns:
        List of executed orders and updated portfolio state.
    """
    engine = _get_paper_trading()

    # Validate weights sum
    total_weight = sum(request.target_weights.values())
    if total_weight > 1.05:
        raise HTTPException(
            status_code=400,
            detail=f"Target weights sum to {total_weight:.2f}, must be ≤ 1.0",
        )

    orders = engine.execute_rebalance(
        target_weights=request.target_weights,
        prices=request.prices,
        regime=request.regime,
        reason=request.reason,
    )

    return RebalanceResponse(
        orders=[PaperOrderResponse(**o.to_dict()) for o in orders],
        total_equity=engine.total_equity,
        cash=engine.cash,
        num_orders=len(orders),
    )
