"""
Analytical module API routes.

Endpoints:
    GET /api/modules/summary            — All 5 module signals
    GET /api/modules/{name}/analyze     — Full analysis for one module
    GET /api/modules/{name}/history     — Signal history for a module
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Path

from api.dependencies import get_system, get_cached_analysis, set_cached_analysis
from api.schemas import (
    ModuleAnalysisResponse,
    ModuleSignalResponse,
    ModuleSummaryResponse,
    SignalHistoryPoint,
    SignalHistoryResponse,
)
from src.regime_detection.base import REGIME_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_MODULES = {"macro", "yield_curve", "options", "factors", "correlations"}


def _parse_module_signal(name: str, raw: dict) -> ModuleSignalResponse:
    """Extract a ModuleSignalResponse from the raw analysis dict for a module."""
    signal_data = raw.get("signal", {})

    # signal_data may be a stringified dict, a ModuleSignal.to_dict(), or a plain string
    if isinstance(signal_data, dict):
        return ModuleSignalResponse(
            module=name,
            signal=signal_data.get("signal", "neutral"),
            strength=float(signal_data.get("strength", 0.0)),
            confidence=float(signal_data.get("confidence", 0.5)),
            explanation=signal_data.get("explanation", ""),
            regime_context=signal_data.get("regime_context", ""),
        )
    # Plain string fallback (e.g. "bullish")
    sig_str = str(signal_data) if signal_data else "neutral"
    return ModuleSignalResponse(
        module=name,
        signal=sig_str if sig_str in {"bullish", "bearish", "neutral", "cautious"} else "neutral",
        strength=0.0,
        confidence=0.5,
        explanation="",
        regime_context="",
    )


def _run_analysis(system) -> dict:
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


@router.get("/summary", response_model=ModuleSummaryResponse)
async def get_module_summary():
    """Return current signal from every analytical module."""
    system = get_system()
    analysis = _run_analysis(system)

    regime_info = analysis.get("regime", {})
    modules_raw = analysis.get("modules", {})

    signals = []
    for name in VALID_MODULES:
        raw = modules_raw.get(name, {})
        if "error" in raw:
            signals.append(
                ModuleSignalResponse(
                    module=name,
                    signal="neutral",
                    strength=0.0,
                    explanation=f"Module error: {raw['error']}",
                )
            )
        else:
            signals.append(_parse_module_signal(name, raw))

    return ModuleSummaryResponse(
        signals=signals,
        current_regime=regime_info.get("id", 1),
        regime_name=regime_info.get("name", REGIME_NAMES.get(1, "")),
        timestamp=datetime.now(),
    )


@router.get("/{name}/analyze", response_model=ModuleAnalysisResponse)
async def get_module_analysis(
    name: str = Path(..., description="Module name"),
):
    """Run full analysis for a single module and return results."""
    if name not in VALID_MODULES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{name}'. Valid: {sorted(VALID_MODULES)}",
        )

    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get(name, {})

    if "error" in raw:
        raise HTTPException(status_code=500, detail=raw["error"])

    signal = _parse_module_signal(name, raw)

    # Raw metrics = everything except the signal key
    details: Dict[str, Any] = raw.get("details", {})
    # Flatten non-serializable items
    safe_details: Dict[str, Any] = {}
    for k, v in details.items():
        try:
            # Quick JSON serialization check
            import json
            json.dumps(v)
            safe_details[k] = v
        except (TypeError, ValueError):
            safe_details[k] = str(v)

    # Regime parameters
    module_instance = system.modules.get(name)
    regime_params: Dict[str, Any] = {}
    if module_instance:
        try:
            regime_params = module_instance.get_regime_parameters(
                system._current_regime or 1
            )
        except Exception:
            pass

    return ModuleAnalysisResponse(
        module=name,
        signal=signal,
        raw_metrics=safe_details,
        regime_parameters=regime_params,
    )


@router.get("/{name}/history", response_model=SignalHistoryResponse)
async def get_module_signal_history(
    name: str = Path(..., description="Module name"),
):
    """Return signal history for a module.

    Note: Full per-module signal history requires persistent storage that
    is not yet implemented.  For now this returns an empty list — the
    frontend should show a placeholder.
    """
    if name not in VALID_MODULES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{name}'. Valid: {sorted(VALID_MODULES)}",
        )

    # TODO: Persist module signals per analysis run so we can return history.
    return SignalHistoryResponse(module=name, history=[])
