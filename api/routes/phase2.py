"""
Phase 2: Intelligence Expansion API routes.

Endpoints:
    GET /api/phase2/transition-forecast   — Forward regime transition probabilities
    GET /api/phase2/multi-timeframe       — Daily/weekly/monthly regime stack
    GET /api/phase2/contagion/analyze     — Full contagion network analysis
    GET /api/phase2/contagion/spillover   — Spillover decomposition only
    GET /api/phase2/surprise-decay/index  — Cumulative surprise index
    GET /api/phase2/surprise-decay/curves — Decay curve projections
    POST /api/phase2/surprise-decay/add   — Register a new macro surprise
    GET /api/phase2/narrative             — Daily narrative briefing
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_system, get_cached_analysis, set_cached_analysis
from api.schemas import (
    ContagionAnalysisResponse,
    DecayCurvePoint,
    DecayCurveResponse,
    DecaySurpriseResponse,
    GrangerLinkResponse,
    ModuleSignalResponse,
    MultiTimeframeResponse,
    NarrativeResponse,
    NetworkGraphResponse,
    SpilloverResponse,
    SurpriseIndexResponse,
    TimeframeRegimeResponse,
    TransitionForecastResponse,
)
from src.regime_detection.base import REGIME_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Helpers ──────────────────────────────────────────────────────


def _run_analysis(system) -> dict:
    """Run or retrieve cached full analysis."""
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


# ─── Regime Transition Forecast ───────────────────────────────────


@router.get("/transition-forecast", response_model=TransitionForecastResponse)
async def get_transition_forecast(
    horizon: int = Query(30, ge=1, le=90, description="Forecast horizon in days"),
):
    """Return forward-looking regime transition probabilities.

    Blends HMM-derived transition dynamics with leading indicator
    signals (disagreement trend, VIX term structure, credit spreads,
    equity-bond correlation, yield curve butterfly).
    """
    system = get_system()

    if not hasattr(system, "transition_model") or system.transition_model is None:
        raise HTTPException(
            status_code=503,
            detail="Transition model not initialized — insufficient data",
        )

    analysis = _run_analysis(system)
    regime_info = analysis.get("regime", {})
    current_regime = regime_info.get("id", 1)

    # Check if forecast is already in the cached analysis
    forecast_raw = analysis.get("transition_forecast")
    if forecast_raw and isinstance(forecast_raw, dict):
        return TransitionForecastResponse(
            current_regime=current_regime,
            horizon_days=forecast_raw.get("horizon_days", horizon),
            hmm_probs=_int_keys_to_str(forecast_raw.get("hmm_probs", {})),
            indicator_probs=_int_keys_to_str(forecast_raw.get("indicator_probs", {})),
            blended_probs=_int_keys_to_str(forecast_raw.get("blended_probs", {})),
            leading_indicators=forecast_raw.get("leading_indicators", {}),
            transition_risk=forecast_raw.get("transition_risk", 0.0),
            most_likely_next=forecast_raw.get("most_likely_next", current_regime),
            most_likely_next_name=REGIME_NAMES.get(
                forecast_raw.get("most_likely_next", current_regime), ""
            ),
            confidence=forecast_raw.get("confidence", 0.0),
        )

    # Run forecast directly
    try:
        data = system.data_pipeline.get_latest_data()
        if data is None or data.empty:
            raise HTTPException(status_code=503, detail="No market data available")

        forecast = system.transition_model.predict(
            current_regime=current_regime,
            market_data=data,
            horizon=horizon,
        )

        return TransitionForecastResponse(
            current_regime=current_regime,
            horizon_days=horizon,
            hmm_probs=_int_keys_to_str(forecast.hmm_probs),
            indicator_probs=_int_keys_to_str(forecast.indicator_probs),
            blended_probs=_int_keys_to_str(forecast.blended_probs),
            leading_indicators=forecast.leading_indicators,
            transition_risk=forecast.transition_risk,
            most_likely_next=forecast.most_likely_next,
            most_likely_next_name=REGIME_NAMES.get(forecast.most_likely_next, ""),
            confidence=forecast.confidence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Transition forecast failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _int_keys_to_str(d: dict) -> Dict[str, float]:
    """Convert integer dict keys to strings for JSON serialization."""
    return {str(k): float(v) for k, v in d.items()}


# ─── Multi-Timeframe Regime Detection ─────────────────────────────


@router.get("/multi-timeframe", response_model=MultiTimeframeResponse)
async def get_multi_timeframe():
    """Return regime classification across daily, weekly, and monthly timeframes.

    Detects conflicts between timeframes and provides a trade signal
    interpretation (e.g. 'buy the dip' when daily=crisis but weekly+monthly=bullish).
    """
    system = get_system()
    analysis = _run_analysis(system)

    # Try cached result first
    mt_raw = analysis.get("multi_timeframe")
    if mt_raw and isinstance(mt_raw, dict):
        return _build_multi_timeframe_response(mt_raw)

    # Run multi-timeframe detection directly
    if not hasattr(system, "multi_timeframe") or system.multi_timeframe is None:
        raise HTTPException(
            status_code=503,
            detail="Multi-timeframe detector not initialized",
        )

    try:
        data = system.data_pipeline.get_latest_data()
        if data is None or data.empty:
            raise HTTPException(status_code=503, detail="No market data available")

        result = system.multi_timeframe.predict(data)
        return _build_multi_timeframe_response(result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Multi-timeframe prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _build_multi_timeframe_response(raw: dict) -> MultiTimeframeResponse:
    """Convert raw multi-timeframe dict to response model."""
    timeframes_raw = raw.get("timeframes", {})

    def _tf(name: str) -> TimeframeRegimeResponse:
        tf = timeframes_raw.get(name, {})
        return TimeframeRegimeResponse(
            timeframe=name,
            regime=tf.get("regime", 1),
            regime_name=tf.get("regime_name", REGIME_NAMES.get(tf.get("regime", 1), "")),
            confidence=tf.get("confidence", 0.0),
            disagreement=tf.get("disagreement", 0.0),
            transition_warning=tf.get("transition_warning", False),
            duration=tf.get("duration", 0),
        )

    return MultiTimeframeResponse(
        daily=_tf("daily"),
        weekly=_tf("weekly"),
        monthly=_tf("monthly"),
        conflict_detected=raw.get("conflict_detected", False),
        highest_conviction=raw.get("highest_conviction", "daily"),
        trade_signal=raw.get("trade_signal", ""),
        agreement_score=raw.get("agreement_score", 1.0),
    )


# ─── Contagion Network ───────────────────────────────────────────


@router.get("/contagion/analyze", response_model=ContagionAnalysisResponse)
async def get_contagion_analysis():
    """Run full contagion network analysis: Granger causality + spillover + graph.

    Returns directional causality links, Diebold-Yilmaz spillover
    decomposition, a network graph for visualization, and contagion
    flags indicating systemic risk.
    """
    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get("contagion", {})

    if "error" in raw:
        raise HTTPException(status_code=500, detail=raw["error"])

    if not raw:
        raise HTTPException(
            status_code=503,
            detail="Contagion module not initialized or no data",
        )

    # Signal
    signal_data = raw.get("signal", {})
    signal = _parse_signal("contagion", signal_data)

    # Granger links
    details = raw.get("details", {})
    granger_raw = details.get("granger_network", details.get("granger_results", []))
    granger_links = [
        GrangerLinkResponse(
            cause=g.get("cause", ""),
            effect=g.get("effect", ""),
            f_stat=g.get("f_stat", 0.0),
            p_value=g.get("p_value", 1.0),
            lag=g.get("lag", 1),
            significant=g.get("significant", False),
        )
        for g in granger_raw
    ]

    # Spillover
    spillover_raw = details.get("spillover", {})
    spillover = SpilloverResponse(
        total_spillover_index=spillover_raw.get("total_spillover_index", 0.0),
        directional_to=_int_keys_to_str(spillover_raw.get("directional_to", {})),
        directional_from=_int_keys_to_str(spillover_raw.get("directional_from", {})),
        net_spillover=_int_keys_to_str(spillover_raw.get("net_spillover", {})),
        pairwise=spillover_raw.get("pairwise", []),
        assets=spillover_raw.get("assets", []),
    )

    # Network graph
    network_raw = details.get("network_graph", details.get("network", {}))
    network = NetworkGraphResponse(
        nodes=network_raw.get("nodes", []),
        edges=network_raw.get("edges", []),
        total_nodes=len(network_raw.get("nodes", [])),
        total_edges=len(network_raw.get("edges", [])),
    )

    # Contagion flags
    contagion_flags = details.get("contagion_flags", {})

    return ContagionAnalysisResponse(
        signal=signal,
        granger_network=granger_links,
        spillover=spillover,
        network_graph=network,
        contagion_flags=contagion_flags,
        n_significant_links=sum(1 for g in granger_links if g.significant),
        network_density=_compute_density(len(granger_links), len(spillover.assets)),
    )


@router.get("/contagion/spillover", response_model=SpilloverResponse)
async def get_spillover_only():
    """Return Diebold-Yilmaz spillover decomposition only (lighter endpoint)."""
    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get("contagion", {})

    details = raw.get("details", {})
    spillover_raw = details.get("spillover", {})

    return SpilloverResponse(
        total_spillover_index=spillover_raw.get("total_spillover_index", 0.0),
        directional_to=_int_keys_to_str(spillover_raw.get("directional_to", {})),
        directional_from=_int_keys_to_str(spillover_raw.get("directional_from", {})),
        net_spillover=_int_keys_to_str(spillover_raw.get("net_spillover", {})),
        pairwise=spillover_raw.get("pairwise", []),
        assets=spillover_raw.get("assets", []),
    )


def _compute_density(n_edges: int, n_nodes: int) -> float:
    """D = edges / (nodes * (nodes - 1)) for a directed graph."""
    if n_nodes < 2:
        return 0.0
    return round(n_edges / (n_nodes * (n_nodes - 1)), 4)


# ─── Surprise Decay ──────────────────────────────────────────────


@router.get("/surprise-decay/index", response_model=SurpriseIndexResponse)
async def get_surprise_index():
    """Return the cumulative macro surprise index with active component breakdown."""
    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get("surprise_decay", {})

    if not raw:
        raise HTTPException(
            status_code=503,
            detail="Surprise decay module not initialized",
        )

    details = raw.get("details", {})

    # cumulative_index may be a nested dict (from real module) or a flat
    # structure where index/direction/components are at the details level.
    cumulative = details.get("cumulative_index", {})
    if isinstance(cumulative, dict):
        # Real module output: cumulative_index is a dict with index, direction, etc.
        return SurpriseIndexResponse(
            index=cumulative.get("index", 0.0),
            direction=cumulative.get("direction", "neutral"),
            components=cumulative.get("components", {}),
            active_surprises=cumulative.get("active_surprises", 0),
            total_historical=cumulative.get("total_historical", 0),
        )
    else:
        # Flat structure (backward compat / mock data)
        return SurpriseIndexResponse(
            index=float(cumulative) if cumulative else 0.0,
            direction=details.get("direction", "neutral"),
            components=details.get("components", {}),
            active_surprises=details.get("active_surprises", 0),
            total_historical=details.get("total_historical", 0),
        )


@router.get("/surprise-decay/curves", response_model=DecayCurveResponse)
async def get_decay_curves(
    days_forward: int = Query(30, ge=1, le=90, description="Days to project forward"),
):
    """Return forward decay curves for all active macro surprises.

    Each curve shows the projected impact over the next N days,
    useful for visualizing when surprises will become stale.
    """
    system = get_system()

    module = system.modules.get("surprise_decay")
    if module is None:
        raise HTTPException(
            status_code=503,
            detail="Surprise decay module not initialized",
        )

    try:
        curves_raw = module.get_decay_curves(forward_days=days_forward)
        curves: Dict[str, List[DecayCurvePoint]] = {}
        for indicator, points in curves_raw.items():
            curves[indicator] = [
                DecayCurvePoint(
                    day=p["day"],
                    impact=round(p["impact"], 6),
                    is_stale=p.get("is_stale", False),
                )
                for p in points
            ]
        return DecayCurveResponse(curves=curves)
    except Exception as exc:
        logger.error(f"Decay curve computation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/surprise-decay/active", response_model=List[DecaySurpriseResponse])
async def get_active_surprises():
    """Return all currently active (non-stale) macro surprises."""
    system = get_system()

    module = system.modules.get("surprise_decay")
    if module is None:
        raise HTTPException(
            status_code=503,
            detail="Surprise decay module not initialized",
        )

    now = datetime.now()
    active = []
    for s in module.active_surprises:
        active.append(
            DecaySurpriseResponse(
                indicator=s.indicator,
                surprise=s.surprise,
                release_date=s.release_date.strftime("%Y-%m-%d"),
                half_life_days=s.half_life_days,
                initial_weight=s.initial_weight,
                regime_at_release=s.regime_at_release,
            )
        )
    return active


# ─── Narrative ────────────────────────────────────────────────────


@router.get("/narrative", response_model=NarrativeResponse)
async def get_narrative():
    """Generate a daily natural-language market briefing.

    Produces a multi-section report based on current regime,
    module signals, transition risks, and positioning implications.
    """
    system = get_system()
    analysis = _run_analysis(system)

    # Check cached narrative
    narrative_raw = analysis.get("narrative")
    if narrative_raw and isinstance(narrative_raw, dict):
        return NarrativeResponse(
            headline=narrative_raw.get("headline", ""),
            regime_section=narrative_raw.get("regime_section", ""),
            signal_section=narrative_raw.get("signal_section", ""),
            risk_section=narrative_raw.get("risk_section", ""),
            positioning_section=narrative_raw.get("positioning_section", ""),
            full_text=narrative_raw.get("full_text", ""),
            data_sources=narrative_raw.get("data_sources", {}),
        )

    # Generate fresh narrative
    if not hasattr(system, "narrative_generator") or system.narrative_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Narrative generator not initialized",
        )

    try:
        briefing = system.narrative_generator.generate(analysis)
        return NarrativeResponse(
            headline=briefing.headline,
            regime_section=briefing.regime_section,
            signal_section=briefing.signal_section,
            risk_section=briefing.risk_section,
            positioning_section=briefing.positioning_section,
            full_text=briefing.full_text,
            data_sources=briefing.data_sources,
        )
    except Exception as exc:
        logger.error(f"Narrative generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Helpers ──────────────────────────────────────────────────────


def _parse_signal(name: str, signal_data: Any) -> ModuleSignalResponse:
    """Extract a ModuleSignalResponse from raw signal data."""
    if isinstance(signal_data, dict):
        return ModuleSignalResponse(
            module=name,
            signal=signal_data.get("signal", "neutral"),
            strength=float(signal_data.get("strength", 0.0)),
            confidence=float(signal_data.get("confidence", 0.5)),
            explanation=signal_data.get("explanation", ""),
            regime_context=signal_data.get("regime_context", ""),
        )
    sig_str = str(signal_data) if signal_data else "neutral"
    return ModuleSignalResponse(
        module=name,
        signal=sig_str if sig_str in {"bullish", "bearish", "neutral", "cautious"} else "neutral",
        strength=0.0,
        confidence=0.5,
        explanation="",
        regime_context="",
    )
