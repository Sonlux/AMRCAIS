"""
Regime detection API routes.

Endpoints:
    GET /api/regime/current       — Current regime classification
    GET /api/regime/history       — Regime history with date range
    GET /api/regime/classifiers   — Individual classifier votes
    GET /api/regime/transitions   — Regime transition matrix
    GET /api/regime/disagreement  — Disagreement index time series
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_system, get_cached_analysis, set_cached_analysis
from api.schemas import (
    ClassifierVoteEntry,
    ClassifierVotesResponse,
    DisagreementPoint,
    DisagreementResponse,
    RegimeHistoryPoint,
    RegimeHistoryResponse,
    RegimeResponse,
    TransitionMatrixResponse,
)
from src.regime_detection.base import REGIME_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_analysis(system) -> dict:
    """Run or retrieve cached analysis."""
    cached = get_cached_analysis()
    if cached is not None:
        return cached

    try:
        result = system.analyze()
        set_cached_analysis(result)
        return result
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/current", response_model=RegimeResponse)
async def get_current_regime():
    """Return the current regime classification with classifier details."""
    system = get_system()
    analysis = _run_analysis(system)
    regime = analysis.get("regime", {})

    # Build classifier votes as string→int for JSON compatibility
    individual = regime.get("individual_predictions", {})
    votes = {str(k): int(v) for k, v in individual.items()}

    # Probabilities — normalise keys to strings
    probs_raw = regime.get("probabilities", {})
    probs = {str(k): float(v) for k, v in probs_raw.items()}

    return RegimeResponse(
        regime=regime.get("id", 1),
        regime_name=regime.get("name", REGIME_NAMES.get(1, "Unknown")),
        confidence=regime.get("confidence", 0.0),
        disagreement=regime.get("disagreement", 0.0),
        classifier_votes=votes,
        probabilities=probs,
        transition_warning=regime.get("transition_warning", False),
        timestamp=datetime.now(),
    )


@router.get("/history", response_model=RegimeHistoryResponse)
async def get_regime_history(
    start: Optional[str] = Query(
        None, description="Start date YYYY-MM-DD (default: 90 days ago)"
    ),
    end: Optional[str] = Query(
        None, description="End date YYYY-MM-DD (default: today)"
    ),
):
    """Return regime classification history within a date range.

    If the ensemble has a prediction history in memory, that is used.
    Otherwise a synthetic history is derived by running predict_sequence
    over the stored market data.
    """
    system = get_system()

    # Determine date bounds
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    start_dt = (
        datetime.strptime(start, "%Y-%m-%d")
        if start
        else end_dt - timedelta(days=90)
    )

    history: list[RegimeHistoryPoint] = []

    # Try ensemble prediction history first (in-memory)
    if system.ensemble and hasattr(system.ensemble, "_prediction_history"):
        for result in system.ensemble._prediction_history:
            ts = result.timestamp
            if ts and start_dt <= ts <= end_dt:
                history.append(
                    RegimeHistoryPoint(
                        date=ts.strftime("%Y-%m-%d"),
                        regime=result.regime,
                        regime_name=result.regime_name,
                        confidence=result.confidence,
                        disagreement=getattr(result, "disagreement", 0.0),
                    )
                )

    # If no in-memory history, attempt predict_sequence over market data
    if not history and system.ensemble and hasattr(system, "market_data"):
        data = system.market_data
        if data is not None and len(data) > 60:
            try:
                results = system.ensemble.predict_sequence(data, window=60, step=5)
                idx = data.index
                for i, res in enumerate(results):
                    point_idx = min(60 + i * 5, len(idx) - 1)
                    dt = idx[point_idx]
                    dt_obj = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
                    if start_dt <= dt_obj <= end_dt:
                        history.append(
                            RegimeHistoryPoint(
                                date=dt_obj.strftime("%Y-%m-%d"),
                                regime=res.regime,
                                regime_name=res.regime_name,
                                confidence=res.confidence,
                                disagreement=getattr(res, "disagreement", 0.0),
                            )
                        )
            except Exception as exc:
                logger.warning(f"predict_sequence failed: {exc}")

    return RegimeHistoryResponse(
        history=history,
        total_points=len(history),
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )


@router.get("/classifiers", response_model=ClassifierVotesResponse)
async def get_classifier_votes():
    """Return individual classifier votes and weights for the latest prediction."""
    system = get_system()
    analysis = _run_analysis(system)
    regime = analysis.get("regime", {})

    votes_raw = regime.get("individual_predictions", {})
    weights = dict(system.ensemble.weights) if system.ensemble else {}

    entries = []
    for clf_name, vote in votes_raw.items():
        entries.append(
            ClassifierVoteEntry(
                classifier=clf_name,
                regime=int(vote),
                confidence=regime.get("confidence", 0.0),
                weight=weights.get(clf_name, 0.0),
            )
        )

    return ClassifierVotesResponse(
        votes=entries,
        ensemble_regime=regime.get("id", 1),
        ensemble_confidence=regime.get("confidence", 0.0),
        weights=weights,
    )


@router.get("/transitions", response_model=TransitionMatrixResponse)
async def get_transition_matrix():
    """Return 4×4 regime transition count matrix.

    Rows = 'from' regime (0-indexed → regime 1-4),
    Cols = 'to' regime.
    """
    system = get_system()

    # Build from prediction history
    matrix = [[0] * 4 for _ in range(4)]
    total = 0

    if system.ensemble and hasattr(system.ensemble, "_prediction_history"):
        hist = system.ensemble._prediction_history
        for i in range(1, len(hist)):
            from_r = hist[i - 1].regime
            to_r = hist[i].regime
            if 1 <= from_r <= 4 and 1 <= to_r <= 4:
                matrix[from_r - 1][to_r - 1] += 1
                if from_r != to_r:
                    total += 1

    return TransitionMatrixResponse(
        matrix=matrix,
        regime_names=[REGIME_NAMES[r] for r in range(1, 5)],
        total_transitions=total,
    )


@router.get("/disagreement", response_model=DisagreementResponse)
async def get_disagreement_series(
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
):
    """Return disagreement index time series."""
    system = get_system()
    threshold = 0.6

    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    start_dt = (
        datetime.strptime(start, "%Y-%m-%d")
        if start
        else end_dt - timedelta(days=90)
    )

    points: list[DisagreementPoint] = []

    if system.ensemble and hasattr(system.ensemble, "_disagreement_history"):
        for ts, val in system.ensemble._disagreement_history:
            if start_dt <= ts <= end_dt:
                points.append(
                    DisagreementPoint(
                        date=ts.strftime("%Y-%m-%d"),
                        disagreement=round(val, 4),
                        threshold_exceeded=val > threshold,
                    )
                )

    avg_d = sum(p.disagreement for p in points) / len(points) if points else 0.0
    max_d = max((p.disagreement for p in points), default=0.0)

    return DisagreementResponse(
        series=points,
        avg_disagreement=round(avg_d, 4),
        max_disagreement=round(max_d, 4),
        threshold=threshold,
    )
