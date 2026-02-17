"""
Phase 5: Network Effects + Moat API routes.

Endpoints:
    GET  /api/phase5/status              — Phase 5 system status overview

    # Knowledge Base (5.1)
    GET  /api/phase5/knowledge/summary   — Knowledge base summary
    GET  /api/phase5/transitions         — Transition history
    POST /api/phase5/transitions         — Record a transition
    POST /api/phase5/transitions/search  — Pattern similarity search
    GET  /api/phase5/anomalies           — Anomaly catalog
    POST /api/phase5/anomalies           — Record an anomaly
    GET  /api/phase5/anomalies/stats     — Anomaly statistics
    POST /api/phase5/macro-impact        — Record macro impact
    GET  /api/phase5/macro-impact/stats  — Macro impact statistics

    # Alternative Data (5.3)
    GET  /api/phase5/alt-data/status     — Alt data integrator status
    GET  /api/phase5/alt-data/signals    — All signals for regime
    POST /api/phase5/alt-data/values     — Set signal values
    GET  /api/phase5/alt-data/vote       — Regime vote from alt data
    GET  /api/phase5/alt-data/types      — List signal types

    # Research Publisher (5.4)
    GET  /api/phase5/research/summary    — Publisher summary
    GET  /api/phase5/research/reports    — List reports
    POST /api/phase5/research/case-study — Generate case study
    POST /api/phase5/research/backtest   — Generate backtest report
    POST /api/phase5/research/factors    — Generate factor analysis

    # User Manager (5.2)
    GET  /api/phase5/users/summary       — User manager summary
    GET  /api/phase5/users               — List users
    POST /api/phase5/users               — Create user
    GET  /api/phase5/users/{user_id}     — Get user
    PUT  /api/phase5/users/{user_id}     — Update user
    DELETE /api/phase5/users/{user_id}   — Delete user
    GET  /api/phase5/annotations         — List all annotations
    POST /api/phase5/annotations         — Create annotation
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_system
from api.schemas import (
    AltDataSetRequest,
    AltDataSignalResponse,
    AltDataStatusResponse,
    AltDataVoteResponse,
    AnomaliesResponse,
    AnomalyRecordRequest,
    AnomalyRecordResponse,
    AnomalyStatsResponse,
    AnnotationCreateRequest,
    AnnotationResponse,
    AnnotationsResponse,
    BacktestReportRequest,
    CaseStudyRequest,
    FactorAnalysisRequest,
    KnowledgeSummaryResponse,
    MacroImpactRequest,
    MacroImpactStatsResponse,
    PatternMatchResponse,
    PatternSearchRequest,
    PatternSearchResponse,
    Phase5StatusResponse,
    ReportListResponse,
    ResearchPublisherSummaryResponse,
    ResearchReportResponse,
    TransitionRecordRequest,
    TransitionRecordResponse,
    TransitionsResponse,
    UserCreateRequest,
    UserListResponse,
    UserManagerSummaryResponse,
    UserResponse,
    UserUpdateRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────


def _get_knowledge_base():
    """Get the knowledge base from the system."""
    system = get_system()
    kb = getattr(system, "knowledge_base", None)
    if kb is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not available",
        )
    return kb


def _get_alt_data():
    """Get the alt data integrator from the system."""
    system = get_system()
    adi = getattr(system, "alt_data_integrator", None)
    if adi is None:
        raise HTTPException(
            status_code=503,
            detail="Alt data integrator not available",
        )
    return adi


def _get_research_publisher():
    """Get the research publisher from the system."""
    system = get_system()
    pub = getattr(system, "research_publisher", None)
    if pub is None:
        raise HTTPException(
            status_code=503,
            detail="Research publisher not available",
        )
    return pub


def _get_user_manager():
    """Get the user manager from the system."""
    system = get_system()
    mgr = getattr(system, "user_manager", None)
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="User manager not available",
        )
    return mgr


# ── Phase 5 Status ────────────────────────────────────────────────


@router.get("/status", response_model=Phase5StatusResponse)
async def phase5_status():
    """Phase 5 system status overview."""
    system = get_system()
    result: Dict[str, Any] = {}

    kb = getattr(system, "knowledge_base", None)
    result["knowledge_base"] = kb.get_summary() if kb else {"status": "not_available"}

    adi = getattr(system, "alt_data_integrator", None)
    result["alt_data"] = adi.get_status() if adi else {"status": "not_available"}

    pub = getattr(system, "research_publisher", None)
    result["research_publisher"] = pub.get_summary() if pub else {"status": "not_available"}

    mgr = getattr(system, "user_manager", None)
    result["user_manager"] = mgr.get_summary() if mgr else {"status": "not_available"}

    return Phase5StatusResponse(**result)


# ── Knowledge Base (5.1) ─────────────────────────────────────────


@router.get("/knowledge/summary", response_model=KnowledgeSummaryResponse)
async def knowledge_summary():
    """Get knowledge base summary statistics."""
    kb = _get_knowledge_base()
    return KnowledgeSummaryResponse(**kb.get_summary())


@router.get("/transitions", response_model=TransitionsResponse)
async def get_transitions(
    from_regime: Optional[int] = Query(None, ge=1, le=4),
    to_regime: Optional[int] = Query(None, ge=1, le=4),
    limit: int = Query(50, ge=1, le=200),
):
    """Query regime transition history."""
    kb = _get_knowledge_base()
    records = kb.get_transitions(
        from_regime=from_regime, to_regime=to_regime, limit=limit
    )
    return TransitionsResponse(
        transitions=[
            TransitionRecordResponse(**r.to_dict()) for r in records
        ],
        total=len(records),
    )


@router.post("/transitions", response_model=TransitionRecordResponse)
async def record_transition(request: TransitionRecordRequest):
    """Record a regime transition event."""
    kb = _get_knowledge_base()
    record = kb.record_transition(
        from_regime=request.from_regime,
        to_regime=request.to_regime,
        confidence=request.confidence,
        disagreement=request.disagreement,
        detection_latency_days=request.detection_latency_days,
        leading_indicators=request.leading_indicators,
        classifier_accuracy=request.classifier_accuracy,
        notes=request.notes,
    )
    return TransitionRecordResponse(**record.to_dict())


@router.post("/transitions/search", response_model=PatternSearchResponse)
async def search_transitions(request: PatternSearchRequest):
    """Find historical transitions similar to current conditions."""
    kb = _get_knowledge_base()
    matches = kb.find_similar_transitions(
        current_indicators=request.current_indicators,
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    return PatternSearchResponse(
        matches=[
            PatternMatchResponse(
                transition=TransitionRecordResponse(**m.record.to_dict()),
                similarity=m.similarity,
                days_ago=m.days_ago,
                outcome_summary=m.outcome_summary,
            )
            for m in matches
        ],
        query_indicators=request.current_indicators,
    )


@router.get("/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(
    anomaly_type: Optional[str] = Query(None),
    asset_pair: Optional[str] = Query(None),
    regime: Optional[int] = Query(None, ge=1, le=4),
    limit: int = Query(50, ge=1, le=200),
):
    """Query anomaly catalog."""
    kb = _get_knowledge_base()
    records = kb.get_anomalies(
        anomaly_type=anomaly_type,
        asset_pair=asset_pair,
        regime=regime,
        limit=limit,
    )
    return AnomaliesResponse(
        anomalies=[AnomalyRecordResponse(**r.to_dict()) for r in records],
        total=len(records),
    )


@router.post("/anomalies", response_model=AnomalyRecordResponse)
async def record_anomaly(request: AnomalyRecordRequest):
    """Record a cross-asset anomaly."""
    kb = _get_knowledge_base()
    record = kb.record_anomaly(
        anomaly_type=request.anomaly_type,
        asset_pair=request.asset_pair,
        regime=request.regime,
        z_score=request.z_score,
        expected_value=request.expected_value,
        actual_value=request.actual_value,
    )
    return AnomalyRecordResponse(**record.to_dict())


@router.get("/anomalies/stats", response_model=AnomalyStatsResponse)
async def anomaly_stats(
    anomaly_type: Optional[str] = Query(None),
):
    """Get aggregate anomaly statistics."""
    kb = _get_knowledge_base()
    stats = kb.get_anomaly_stats(anomaly_type=anomaly_type)
    return AnomalyStatsResponse(**stats)


@router.post("/macro-impact")
async def record_macro_impact(request: MacroImpactRequest):
    """Record a macro surprise impact observation."""
    kb = _get_knowledge_base()
    kb.record_macro_impact(
        indicator=request.indicator,
        regime=request.regime,
        impact_pct=request.impact_pct,
    )
    return {"status": "recorded", "indicator": request.indicator}


@router.get("/macro-impact/stats", response_model=MacroImpactStatsResponse)
async def macro_impact_stats(
    indicator: Optional[str] = Query(None),
):
    """Get macro impact aggregate statistics."""
    kb = _get_knowledge_base()
    stats = kb.get_macro_impact_stats(indicator=indicator)
    return MacroImpactStatsResponse(stats=stats)


# ── Alternative Data (5.3) ───────────────────────────────────────


@router.get("/alt-data/status", response_model=AltDataStatusResponse)
async def alt_data_status():
    """Get alternative data integrator status."""
    adi = _get_alt_data()
    return AltDataStatusResponse(**adi.get_status())


@router.get("/alt-data/types")
async def alt_data_types():
    """List all supported alternative data signal types."""
    adi = _get_alt_data()
    return {"signal_types": adi.get_signal_types()}


@router.post("/alt-data/values")
async def set_alt_data_values(request: AltDataSetRequest):
    """Set current alternative data signal values."""
    adi = _get_alt_data()
    adi.set_all_values(request.values)
    return {
        "status": "updated",
        "signals_set": len(request.values),
        "signal_names": list(request.values.keys()),
    }


@router.get("/alt-data/signals")
async def get_alt_data_signals(
    regime: int = Query(..., ge=1, le=4),
):
    """Get all available signals interpreted for a regime."""
    adi = _get_alt_data()
    signals = adi.get_all_signals(regime=regime)
    return {
        "signals": [s.to_dict() for s in signals],
        "regime": regime,
        "count": len(signals),
    }


@router.get("/alt-data/vote", response_model=AltDataVoteResponse)
async def alt_data_vote(
    regime: int = Query(..., ge=1, le=4),
):
    """Get a weighted regime vote from alternative data signals."""
    adi = _get_alt_data()
    vote = adi.get_regime_vote(regime=regime)
    return AltDataVoteResponse(**vote)


# ── Research Publisher (5.4) ──────────────────────────────────────


@router.get("/research/summary", response_model=ResearchPublisherSummaryResponse)
async def research_summary():
    """Get research publisher status summary."""
    pub = _get_research_publisher()
    return ResearchPublisherSummaryResponse(**pub.get_summary())


@router.get("/research/reports", response_model=ReportListResponse)
async def list_reports(
    report_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
):
    """List generated research reports."""
    pub = _get_research_publisher()
    reports = pub.get_reports(report_type=report_type, limit=limit)
    return ReportListResponse(
        reports=[ResearchReportResponse(**r.to_dict()) for r in reports],
        total=len(reports),
    )


@router.post("/research/case-study", response_model=ResearchReportResponse)
async def generate_case_study(request: CaseStudyRequest):
    """Generate a regime transition case study."""
    pub = _get_research_publisher()
    report = pub.generate_transition_case_study(
        from_regime=request.from_regime,
        to_regime=request.to_regime,
        limit=request.limit,
    )
    return ResearchReportResponse(**report.to_dict())


@router.post("/research/backtest", response_model=ResearchReportResponse)
async def generate_backtest_report(request: BacktestReportRequest):
    """Generate a backtest results report."""
    pub = _get_research_publisher()
    report = pub.generate_backtest_report(
        backtest_results=request.backtest_results,
    )
    return ResearchReportResponse(**report.to_dict())


@router.post("/research/factors", response_model=ResearchReportResponse)
async def generate_factor_analysis(request: FactorAnalysisRequest):
    """Generate a factor exposure analysis."""
    pub = _get_research_publisher()
    report = pub.generate_factor_analysis(
        factor_data=request.factor_data,
    )
    return ResearchReportResponse(**report.to_dict())


# ── User Manager (5.2) ───────────────────────────────────────────


@router.get("/users/summary", response_model=UserManagerSummaryResponse)
async def user_manager_summary():
    """Get user manager status summary."""
    mgr = _get_user_manager()
    return UserManagerSummaryResponse(**mgr.get_summary())


@router.get("/users", response_model=UserListResponse)
async def list_users(
    role: Optional[str] = Query(None),
    active_only: bool = Query(True),
):
    """List all users, optionally filtered."""
    mgr = _get_user_manager()

    from src.knowledge.user_manager import UserRole

    role_filter = None
    if role:
        try:
            role_filter = UserRole(role)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {role}. Valid: {[r.value for r in UserRole]}",
            )

    users = mgr.list_users(role=role_filter, active_only=active_only)
    return UserListResponse(
        users=[UserResponse(**u.to_dict()) for u in users],
        total=len(users),
    )


@router.post("/users", response_model=UserResponse)
async def create_user(request: UserCreateRequest):
    """Create a new user account.

    Returns the user (API key is included only in the response body
    as ``api_key`` — store it securely, it cannot be retrieved later).
    """
    mgr = _get_user_manager()

    from src.knowledge.user_manager import UserRole

    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {request.role}. Valid: {[r.value for r in UserRole]}",
        )

    try:
        user, api_key = mgr.create_user(
            name=request.name,
            email=request.email,
            role=role,
            preferences=request.preferences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    resp = user.to_dict()
    resp["api_key"] = api_key  # Only returned at creation time
    return UserResponse(**user.to_dict())


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by ID."""
    mgr = _get_user_manager()
    user = mgr.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user.to_dict())


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, request: UserUpdateRequest):
    """Update a user's details."""
    mgr = _get_user_manager()

    from src.knowledge.user_manager import UserRole

    role = None
    if request.role:
        try:
            role = UserRole(request.role)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {request.role}",
            )

    user = mgr.update_user(
        user_id,
        name=request.name,
        role=role,
        is_active=request.is_active,
        preferences=request.preferences,
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(**user.to_dict())


@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete a user account."""
    mgr = _get_user_manager()
    if not mgr.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "user_id": user_id}


@router.get("/annotations", response_model=AnnotationsResponse)
async def list_annotations(
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List shared annotations."""
    mgr = _get_user_manager()
    notes = mgr.get_annotations(user_id=user_id, limit=limit)
    return AnnotationsResponse(
        annotations=[AnnotationResponse(**n) for n in notes],
        total=len(notes),
    )


@router.post("/annotations", response_model=AnnotationResponse)
async def create_annotation(request: AnnotationCreateRequest):
    """Create a shared annotation."""
    mgr = _get_user_manager()
    note = mgr.add_annotation(
        user_id=request.user_id,
        content=request.content,
        context=request.context,
    )
    if not note:
        raise HTTPException(
            status_code=404,
            detail=f"User not found: {request.user_id}",
        )
    return AnnotationResponse(**note)
