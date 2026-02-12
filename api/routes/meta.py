"""
Meta-learning API routes.

Endpoints:
    GET /api/meta/performance         — Classifier accuracy metrics
    GET /api/meta/weights             — Current ensemble weights
    GET /api/meta/weights/history     — Weight evolution over time
    GET /api/meta/recalibrations      — Recalibration event log
    GET /api/meta/health              — System health summary
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.dependencies import get_system
from api.schemas import (
    HealthResponse,
    PerformanceResponse,
    RecalibrationEvent,
    RecalibrationResponse,
    WeightHistoryPoint,
    WeightHistoryResponse,
    WeightsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance():
    """Return performance metrics for the last 30 days."""
    system = get_system()

    if system.meta_learner is None:
        raise HTTPException(status_code=503, detail="Meta-learner not initialized")

    try:
        metrics = system.meta_learner.get_performance_metrics(lookback_days=30)
    except Exception as exc:
        logger.error(f"Failed to get performance metrics: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Convert regime distribution keys to strings for JSON
    dist = {str(k): v for k, v in metrics.regime_distribution.items()}

    return PerformanceResponse(
        stability_score=metrics.regime_stability_score,
        stability_rating=metrics.stability_rating,
        transition_count=metrics.transition_count,
        avg_disagreement=metrics.avg_disagreement,
        high_disagreement_days=metrics.high_disagreement_days,
        total_classifications=metrics.total_classifications,
        regime_distribution=dist,
    )


@router.get("/weights", response_model=WeightsResponse)
async def get_weights():
    """Return current ensemble classifier weights."""
    system = get_system()

    if system.meta_learner:
        weights = system.meta_learner.get_adaptive_weights()
        is_adaptive = system.meta_learner.enable_adaptive_weights
    elif system.ensemble:
        weights = dict(system.ensemble.weights)
        is_adaptive = False
    else:
        weights = {"hmm": 0.30, "ml": 0.25, "correlation": 0.25, "volatility": 0.20}
        is_adaptive = False

    return WeightsResponse(weights=weights, is_adaptive=is_adaptive)


@router.get("/weights/history", response_model=WeightHistoryResponse)
async def get_weight_history():
    """Return ensemble weight evolution over time.

    Note: Full weight history persistence is not yet implemented.
    Currently returns a single snapshot of current weights.
    """
    system = get_system()

    current_weights = {}
    if system.meta_learner:
        current_weights = system.meta_learner.get_adaptive_weights()
    elif system.ensemble:
        current_weights = dict(system.ensemble.weights)

    history = []
    if current_weights:
        history.append(
            WeightHistoryPoint(
                date=datetime.now().strftime("%Y-%m-%d"),
                weights=current_weights,
            )
        )

    return WeightHistoryResponse(history=history)


@router.get("/recalibrations", response_model=RecalibrationResponse)
async def get_recalibrations():
    """Return log of recalibration events."""
    system = get_system()

    events: list[RecalibrationEvent] = []
    total = 0
    last_recal = None

    if system.meta_learner:
        total = system.meta_learner._recalibration_count
        if system.meta_learner._last_recalibration:
            last_recal = system.meta_learner._last_recalibration.isoformat()

        # Check current recalibration status for informational purposes
        try:
            decision = system.meta_learner.check_recalibration_needed()
            if decision.should_recalibrate:
                events.append(
                    RecalibrationEvent(
                        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                        trigger_reason=", ".join(str(r) for r in decision.reasons),
                        severity=decision.severity,
                        urgency=decision.urgency_level,
                        recommendations=decision.recommendations,
                    )
                )
        except Exception as exc:
            logger.warning(f"Recalibration check failed: {exc}")

    return RecalibrationResponse(
        events=events,
        total_recalibrations=total,
        last_recalibration=last_recal,
    )


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Return comprehensive system health summary."""
    system = get_system()

    if system.meta_learner is None:
        return HealthResponse(
            system_status="degraded",
            needs_recalibration=False,
            urgency="none",
        )

    try:
        report = system.meta_learner.get_system_health_report()
    except Exception as exc:
        logger.error(f"Health report failed: {exc}")
        return HealthResponse(
            system_status="error",
            needs_recalibration=False,
            urgency="none",
            reasons=[str(exc)],
        )

    recal = report.get("recalibration", {})

    return HealthResponse(
        system_status=report.get("system_status", "unknown"),
        needs_recalibration=recal.get("needed", False),
        urgency=recal.get("urgency", "none"),
        severity=recal.get("severity", 0.0),
        reasons=recal.get("reasons", []),
        performance_30d=report.get("performance_30d", {}),
        performance_7d=report.get("performance_7d", {}),
        alerts=report.get("alerts", {}),
        adaptive_weights=report.get("adaptive_weights"),
    )
