"""
Phase 3: Prediction Engine API routes.

Endpoints:
    GET  /api/phase3/return-forecast          — Regime-conditional return forecasts
    GET  /api/phase3/return-forecast/{asset}   — Single-asset return forecast
    GET  /api/phase3/regime-coefficients/{asset}— Fitted regime coefficients
    GET  /api/phase3/tail-risk                 — Regime-conditional VaR / CVaR
    GET  /api/phase3/portfolio-optimize        — Regime-aware portfolio optimisation
    GET  /api/phase3/alpha-signals             — Anomaly-based alpha signals
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_system, get_cached_analysis, set_cached_analysis
from api.schemas import (
    AlphaSignalResponse,
    AlphaSignalsResponse,
    HedgeRecommendationResponse,
    PortfolioOptimizationResponse,
    RegimeAllocationResponse,
    RegimeCoefficientsResponse,
    ReturnForecastResponse,
    ReturnForecastsResponse,
    ScenarioVaRResponse,
    TailRiskResponse,
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


def _get_current_regime(system) -> int:
    """Resolve current regime from cached analysis or system state."""
    analysis = _run_analysis(system)
    return analysis.get("regime", {}).get("id", 1)


def _parse_transition_probs(
    raw: Optional[str], current_regime: int
) -> Dict[int, float]:
    """Parse transition probs from query string or use defaults."""
    if raw:
        try:
            import json
            parsed = json.loads(raw)
            return {int(k): float(v) for k, v in parsed.items()}
        except Exception:
            pass
    # Default: 70% stay, 10% each other
    probs: Dict[int, float] = {}
    for r in [1, 2, 3, 4]:
        probs[r] = 0.70 if r == current_regime else 0.10
    return probs


# ─── Return Forecast Endpoints ───────────────────────────────────


@router.get("/return-forecast", response_model=ReturnForecastsResponse)
async def get_return_forecasts():
    """Return regime-conditional return forecasts for all assets.

    Uses the prediction engine's Hamilton-style regime-switching
    regression models.  Each asset has separate alpha and factor
    loadings per regime.
    """
    system = get_system()
    regime = _get_current_regime(system)

    if not hasattr(system, "return_forecaster") or system.return_forecaster is None:
        raise HTTPException(
            status_code=503,
            detail="Return forecaster not initialized",
        )

    forecaster = system.return_forecaster
    if not forecaster.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Return forecaster not fitted — insufficient data",
        )

    try:
        all_forecasts = forecaster.predict_all(current_regime=regime)
        items = [
            ReturnForecastResponse(
                asset=fc.asset,
                regime=fc.regime,
                expected_return=fc.expected_return,
                volatility=fc.volatility,
                r_squared_regime=fc.r_squared_regime,
                r_squared_static=fc.r_squared_static,
                r_squared_improvement=fc.r_squared_improvement,
                kelly_fraction=fc.kelly_fraction,
                factor_contributions=fc.factor_contributions,
                confidence=fc.confidence,
            )
            for fc in all_forecasts.values()
        ]
        return ReturnForecastsResponse(
            current_regime=regime,
            regime_name=REGIME_NAMES.get(regime, ""),
            forecasts=items,
        )
    except Exception as exc:
        logger.error(f"Return forecast failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/return-forecast/{asset}",
    response_model=ReturnForecastResponse,
)
async def get_asset_return_forecast(asset: str):
    """Return forecast for a single asset."""
    system = get_system()
    regime = _get_current_regime(system)

    if not hasattr(system, "return_forecaster") or system.return_forecaster is None:
        raise HTTPException(status_code=503, detail="Return forecaster not initialized")

    forecaster = system.return_forecaster
    if not forecaster.is_fitted:
        raise HTTPException(status_code=503, detail="Return forecaster not fitted")

    try:
        fc = forecaster.predict(asset=asset.upper(), current_regime=regime)
        return ReturnForecastResponse(
            asset=fc.asset,
            regime=fc.regime,
            expected_return=fc.expected_return,
            volatility=fc.volatility,
            r_squared_regime=fc.r_squared_regime,
            r_squared_static=fc.r_squared_static,
            r_squared_improvement=fc.r_squared_improvement,
            kelly_fraction=fc.kelly_fraction,
            factor_contributions=fc.factor_contributions,
            confidence=fc.confidence,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No model for asset '{asset}'")
    except Exception as exc:
        logger.error(f"Forecast for {asset} failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/regime-coefficients/{asset}",
    response_model=RegimeCoefficientsResponse,
)
async def get_regime_coefficients(asset: str):
    """Return fitted regression coefficients for all regimes."""
    system = get_system()

    if not hasattr(system, "return_forecaster") or system.return_forecaster is None:
        raise HTTPException(status_code=503, detail="Return forecaster not initialized")

    coeffs = system.return_forecaster.get_regime_coefficients(asset.upper())
    if not coeffs:
        raise HTTPException(
            status_code=404, detail=f"No coefficients for asset '{asset}'"
        )

    return RegimeCoefficientsResponse(
        asset=asset.upper(),
        regimes={str(k): v for k, v in coeffs.items()},
    )


# ─── Tail Risk Endpoints ─────────────────────────────────────────


@router.get("/tail-risk", response_model=TailRiskResponse)
async def get_tail_risk(
    transition_probs: Optional[str] = Query(
        None,
        description='JSON dict of regime→probability, e.g. {"1":0.65,"2":0.20,"3":0.10,"4":0.05}',
    ),
    portfolio: Optional[str] = Query(
        None,
        description='JSON dict of asset→weight, e.g. {"SPX":0.6,"TLT":0.3,"GLD":0.1}',
    ),
):
    """Compute regime-conditional VaR / CVaR with scenario attribution.

    Shows VaR broken down by regime transition scenario, identifies
    the primary tail risk driver, and provides hedge recommendations.
    """
    system = get_system()
    regime = _get_current_regime(system)

    if not hasattr(system, "tail_risk") or system.tail_risk is None:
        raise HTTPException(
            status_code=503, detail="Tail risk analyzer not initialized"
        )

    analyzer = system.tail_risk
    if not analyzer.is_fitted:
        raise HTTPException(
            status_code=503, detail="Tail risk analyzer not fitted"
        )

    # Parse portfolio weights
    import json

    if portfolio:
        try:
            weights = {k: float(v) for k, v in json.loads(portfolio).items()}
        except Exception:
            raise HTTPException(
                status_code=400, detail="Invalid portfolio JSON"
            )
    else:
        weights = {"SPX": 0.60, "TLT": 0.25, "GLD": 0.15}

    probs = _parse_transition_probs(transition_probs, regime)

    try:
        result = analyzer.analyze(
            portfolio_weights=weights,
            current_regime=regime,
            transition_probs=probs,
        )
        return TailRiskResponse(
            current_regime=result.current_regime,
            weighted_var=result.weighted_var,
            weighted_cvar=result.weighted_cvar,
            scenarios=[
                ScenarioVaRResponse(
                    from_regime=s.from_regime,
                    to_regime=s.to_regime,
                    to_regime_name=s.to_regime_name,
                    probability=s.probability,
                    var_99=s.var_99,
                    cvar_99=s.cvar_99,
                    contribution=s.contribution,
                    risk_drivers=s.risk_drivers,
                )
                for s in result.scenarios
            ],
            worst_scenario=result.worst_scenario,
            worst_scenario_var=result.worst_scenario_var,
            tail_risk_driver=result.tail_risk_driver,
            hedge_recommendations=[
                HedgeRecommendationResponse(
                    scenario=h.scenario,
                    instrument=h.instrument,
                    rationale=h.rationale,
                    urgency=h.urgency,
                )
                for h in result.hedge_recommendations
            ],
            portfolio_weights=result.portfolio_weights,
        )
    except Exception as exc:
        logger.error(f"Tail risk analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Portfolio Optimization Endpoints ─────────────────────────────


@router.get("/portfolio-optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    transition_probs: Optional[str] = Query(
        None,
        description='JSON dict of regime→probability',
    ),
    rebalance_threshold: float = Query(
        0.40, ge=0.1, le=1.0,
        description="Transition probability threshold for rebalance trigger",
    ),
):
    """Compute regime-aware optimal portfolio allocation.

    Solves mean-variance for each regime scenario, then blends
    allocations by transition probabilities.  Includes transaction
    cost estimates and rebalance triggers.
    """
    system = get_system()
    regime = _get_current_regime(system)

    if not hasattr(system, "portfolio_optimizer") or system.portfolio_optimizer is None:
        raise HTTPException(
            status_code=503, detail="Portfolio optimizer not initialized"
        )

    optimizer = system.portfolio_optimizer
    if not optimizer.is_fitted:
        raise HTTPException(
            status_code=503, detail="Portfolio optimizer not fitted"
        )

    probs = _parse_transition_probs(transition_probs, regime)

    try:
        result = optimizer.optimize(
            current_regime=regime,
            transition_probs=probs,
            rebalance_threshold=rebalance_threshold,
        )
        return PortfolioOptimizationResponse(
            current_regime=result.current_regime,
            blended_weights=result.blended_weights,
            regime_allocations=[
                RegimeAllocationResponse(
                    regime=a.regime,
                    regime_name=a.regime_name,
                    weights=a.weights,
                    expected_return=a.expected_return,
                    expected_volatility=a.expected_volatility,
                    sharpe_ratio=a.sharpe_ratio,
                )
                for a in result.regime_allocations
            ],
            rebalance_trigger=result.rebalance_trigger,
            rebalance_reason=result.rebalance_reason,
            transaction_cost_estimate=result.transaction_cost_estimate,
            expected_return=result.expected_return,
            expected_volatility=result.expected_volatility,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown_constraint=result.max_drawdown_constraint,
        )
    except Exception as exc:
        logger.error(f"Portfolio optimization failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Alpha Signal Endpoints ──────────────────────────────────────


@router.get("/alpha-signals", response_model=AlphaSignalsResponse)
async def get_alpha_signals():
    """Generate anomaly-based alpha signals for the current regime.

    Converts active correlation anomalies into regime-conditional
    tradeable signals with backtested win rates and holding periods.
    """
    system = get_system()
    regime = _get_current_regime(system)

    if not hasattr(system, "alpha_signals") or system.alpha_signals is None:
        raise HTTPException(
            status_code=503, detail="Alpha signal generator not initialized"
        )

    gen = system.alpha_signals
    if not gen.is_fitted:
        raise HTTPException(
            status_code=503, detail="Alpha signal generator not fitted"
        )

    # Collect active anomalies from correlation module
    analysis = _run_analysis(system)
    modules_raw = analysis.get("modules", {})
    corr_raw = modules_raw.get("correlations", {})
    details = corr_raw.get("details", {})

    # Build anomaly dict from correlation details
    active_anomalies: Dict[str, float] = {}
    anomalies = details.get("anomalies", details.get("z_scores", {}))
    if isinstance(anomalies, dict):
        for pair, value in anomalies.items():
            if isinstance(value, (int, float)) and abs(value) > 0.1:
                active_anomalies[pair] = float(value)

    # Add VIX spike if options module shows elevated vol
    options_raw = modules_raw.get("options", {})
    opt_details = options_raw.get("details", {})
    vix = opt_details.get("vix", opt_details.get("vix_level", 0))
    if isinstance(vix, (int, float)) and vix > 25:
        active_anomalies["VIX_spike"] = min(1.0, (vix - 20) / 20)

    if not active_anomalies:
        # Provide baseline response even with no anomalies
        return AlphaSignalsResponse(
            signals=[],
            composite_score=0.0,
            top_signal=None,
            regime=regime,
            n_active_anomalies=0,
            regime_context=f"No actionable anomalies in {REGIME_NAMES.get(regime, 'current')} regime.",
        )

    try:
        result = gen.generate(
            current_regime=regime,
            active_anomalies=active_anomalies,
        )
        return AlphaSignalsResponse(
            signals=[
                AlphaSignalResponse(
                    anomaly_type=s.anomaly_type,
                    direction=s.direction,
                    rationale=s.rationale,
                    strength=s.strength,
                    confidence=s.confidence,
                    holding_period_days=s.holding_period_days,
                    historical_win_rate=s.historical_win_rate,
                    regime=s.regime,
                )
                for s in result.signals
            ],
            composite_score=result.composite_score,
            top_signal=result.top_signal,
            regime=result.regime,
            n_active_anomalies=result.n_active_anomalies,
            regime_context=result.regime_context,
        )
    except Exception as exc:
        logger.error(f"Alpha signal generation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
