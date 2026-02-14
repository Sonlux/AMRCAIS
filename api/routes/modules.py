"""
Analytical module API routes.

Endpoints:
    GET /api/modules/summary            — All 5 module signals
    GET /api/modules/{name}/analyze     — Full analysis for one module
    GET /api/modules/{name}/history     — Signal history for a module
    GET /api/modules/yield_curve/curve  — Yield curve snapshot (tenors + yields)
    GET /api/modules/options/surface    — Implied volatility surface grid
"""

import logging
import math
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
    VolSurfaceDataResponse,
    YieldCurveDataResponse,
)
from src.regime_detection.base import REGIME_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_MODULES = {"macro", "yield_curve", "options", "factors", "correlations", "contagion", "surprise_decay"}


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

    Retrieves persisted module signals from the database. Each analysis
    run persists signals, so history grows over time.
    """
    if name not in VALID_MODULES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown module '{name}'. Valid: {sorted(VALID_MODULES)}",
        )

    system = get_system()

    history: list[SignalHistoryPoint] = []

    # Load from database if pipeline and storage are available
    if system.pipeline and hasattr(system.pipeline, "storage"):
        try:
            df = system.pipeline.storage.load_module_signals(
                module_name=name,
                limit=500,
            )
            if not df.empty:
                for idx, row in df.iterrows():
                    # Convert index (datetime) to ISO string
                    date_str = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                    
                    # Safely extract regime_id (may be None or NaN)
                    raw_regime = row.get("Regime_ID")
                    regime_id = None
                    if raw_regime is not None:
                        try:
                            import math
                            if not math.isnan(float(raw_regime)):
                                regime_id = int(raw_regime)
                        except (ValueError, TypeError):
                            pass
                    
                    history.append(
                        SignalHistoryPoint(
                            date=date_str,
                            signal=row.get("Signal", "neutral"),
                            strength=float(row.get("Strength", 0.0)),
                            confidence=float(row.get("Confidence", 0.5)),
                            regime=regime_id,
                        )
                    )
        except Exception as exc:
            logger.warning(f"Failed to load signal history for {name}: {exc}")

    return SignalHistoryResponse(module=name, history=history)


# ─── Yield Curve Surface ─────────────────────────────────────────


@router.get("/yield_curve/curve", response_model=YieldCurveDataResponse)
async def get_yield_curve_data():
    """Return yield curve snapshot with observed yields + interpolated points.

    Uses the YieldCurveAnalyzer to extract tenor-yield pairs from the
    latest analysis and interpolate a smooth curve via cubic splines.
    """
    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get("yield_curve", {})

    regime_info = analysis.get("regime", {})
    regime_id = regime_info.get("id", 1)

    # The YieldCurveAnalyzer stores yields in raw details
    details = raw.get("details", {})
    yields_dict = details.get("yields", {})

    # If no yields from details, try raw_metrics top level
    if not yields_dict:
        yields_dict = raw.get("yields", {})

    if not yields_dict:
        # Fallback: build from the module instance directly
        module_inst = system.modules.get("yield_curve")
        if module_inst and module_inst.curve_history:
            latest = module_inst.curve_history[-1]
            yields_dict = latest.yields

    # Convert tenor strings -> years and sort
    tenor_to_years = {
        "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
        "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
    }

    tenors: list[float] = []
    yield_vals: list[float] = []

    for tenor_str in sorted(yields_dict.keys(), key=lambda t: tenor_to_years.get(t, 99)):
        if tenor_str in tenor_to_years:
            tenors.append(tenor_to_years[tenor_str])
            yield_vals.append(float(yields_dict[tenor_str]))

    if len(tenors) < 2:
        raise HTTPException(
            status_code=503,
            detail="Insufficient yield data — need at least 2 tenor points",
        )

    # Optional: interpolate via module's cubic spline method
    module_inst = system.modules.get("yield_curve")
    if module_inst and len(yields_dict) >= 3:
        try:
            interp_targets = [
                0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30,
            ]
            interp = module_inst.interpolate_curve(yields_dict, interp_targets)
            if interp:
                tenors = sorted(interp.keys())
                yield_vals = [interp[t] for t in tenors]
        except Exception as exc:
            logger.warning(f"Yield curve interpolation failed: {exc}")

    # Shape info from analysis
    curve_shape = raw.get("curve_shape", details.get("curve_shape", "normal"))
    slope_2_10 = raw.get("slope_2_10", details.get("slope_2_10"))
    slope_3m_10 = raw.get("slope_3m_10", details.get("slope_3m_10"))
    curvature = raw.get("curvature", details.get("curvature"))

    return YieldCurveDataResponse(
        tenors=tenors,
        yields=yield_vals,
        curve_shape=str(curve_shape),
        slope_2_10=float(slope_2_10) if slope_2_10 is not None else None,
        slope_3m_10=float(slope_3m_10) if slope_3m_10 is not None else None,
        curvature=float(curvature) if curvature is not None else None,
        regime=regime_id,
        regime_name=REGIME_NAMES.get(regime_id, ""),
    )


# ─── Volatility Surface ──────────────────────────────────────────

# Moneyness grid: 80% → 120% in 5% steps
_MONEYNESS_GRID = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]

# Expiry grid (trading days)
_EXPIRY_GRID = [7, 14, 30, 60, 90, 120, 180, 270, 365]


def _build_parametric_surface(
    atm_vol: float,
    regime: int,
) -> list[list[float]]:
    """Build a synthetic IV surface from ATM vol using a SABR-like smile model.

    The smile is parameterised by regime:
      - Risk-Off: steep put skew, backwardation
      - Risk-On: mild smile, contango
      - Stagflation: elevated baseline, moderate skew
      - Disinflationary Boom: flat smile, low vol
    """
    # Regime-dependent skew and term-structure parameters
    params = {
        1: {"skew": 0.08, "convexity": 0.04, "ts_slope": 0.02},   # Risk-On
        2: {"skew": 0.22, "convexity": 0.10, "ts_slope": -0.04},  # Risk-Off
        3: {"skew": 0.14, "convexity": 0.06, "ts_slope": 0.01},   # Stagflation
        4: {"skew": 0.05, "convexity": 0.03, "ts_slope": 0.03},   # Disinfl. Boom
    }
    p = params.get(regime, params[1])

    grid: list[list[float]] = []
    for dte in _EXPIRY_GRID:
        sqrt_t = math.sqrt(dte / 365.0)
        # Term structure adjustment (contango/backwardation)
        ts_adj = p["ts_slope"] * (sqrt_t - math.sqrt(30 / 365.0))
        row: list[float] = []
        for m in _MONEYNESS_GRID:
            log_m = math.log(m)  # log-moneyness
            # Smile: skew * log_m + convexity * log_m^2
            smile = -p["skew"] * log_m + p["convexity"] * log_m ** 2
            iv = max(1.0, atm_vol * (1 + smile + ts_adj))
            row.append(round(iv, 2))
        grid.append(row)
    return grid


@router.get("/options/surface", response_model=VolSurfaceDataResponse)
async def get_vol_surface():
    """Return a volatility surface grid for 3D rendering.

    Uses VIX as the ATM vol proxy and builds a parametric SABR-like
    surface whose skew and term-structure characteristics adapt to the
    current regime.
    """
    system = get_system()
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    raw = modules_raw.get("options", {})

    regime_info = analysis.get("regime", {})
    regime_id = regime_info.get("id", 1)

    # ATM vol = VIX
    details = raw.get("details", {})
    atm_vol = raw.get("vix_level") or details.get("vix_level")

    if atm_vol is None:
        # Try extracting VIX from the data pipeline directly
        try:
            data = system.data_pipeline.get_latest_data()
            if data is not None:
                for col in ("VIX", "VIXCLS"):
                    if col in data.columns:
                        series = data[col].dropna()
                        if len(series):
                            atm_vol = float(series.iloc[-1])
                            break
        except Exception:
            pass

    if atm_vol is None:
        atm_vol = 20.0  # sensible default

    iv_grid = _build_parametric_surface(atm_vol, regime_id)

    return VolSurfaceDataResponse(
        moneyness=_MONEYNESS_GRID,
        expiry_days=_EXPIRY_GRID,
        iv_grid=iv_grid,
        atm_vol=round(atm_vol, 2),
        regime=regime_id,
        regime_name=REGIME_NAMES.get(regime_id, ""),
    )
