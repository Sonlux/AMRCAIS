"""
Tail Risk Attribution (Phase 3.2).

Regime-conditional VaR / CVaR with scenario-based attribution.
Shows not just *how much* risk, but *where* the risk comes from
when regime transitions are considered.

    Standard VaR:   -2.3%
    AMRCAIS VaR:    If Risk-On persists (65%): -1.4%
                    If transition to Risk-Off: -4.8%  ← THIS is the risk
                    Transition-probability weighted:   -2.3%

Key features:
    - Per-regime covariance matrices
    - Conditional VaR / CVaR per transition scenario
    - Attribution: which scenario drives tail risk
    - Hedging recommendations (instrument-specific)

Classes:
    TailRiskAnalyzer: Main tail risk engine

Example:
    >>> analyzer = TailRiskAnalyzer()
    >>> analyzer.fit(market_data, regime_series)
    >>> result = analyzer.analyze(
    ...     portfolio_weights={"SPX": 0.6, "TLT": 0.3, "GLD": 0.1},
    ...     current_regime=1,
    ...     transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05},
    ... )
    >>> print(result.weighted_var)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────

REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}

# Default hedge recommendations per regime transition
_HEDGE_TEMPLATES: Dict[Tuple[int, int], Dict[str, str]] = {
    (1, 2): {
        "instrument": "TLT puts / VIX call spreads",
        "rationale": "Risk-Off transition: buy diversification insurance",
    },
    (1, 3): {
        "instrument": "GLD calls / TIPS",
        "rationale": "Stagflation: real assets hedge inflation + growth drag",
    },
    (2, 1): {
        "instrument": "Reduce hedges gradually",
        "rationale": "Recovery forming — let risk-on exposures run",
    },
    (3, 4): {
        "instrument": "Duration longs (TLT calls)",
        "rationale": "Disinflation: bonds rally as inflation fades",
    },
}


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class ScenarioVaR:
    """VaR / CVaR for a single regime-transition scenario.

    Attributes:
        from_regime: Current regime.
        to_regime: Target regime.
        to_regime_name: Human-readable name.
        probability: Transition probability.
        var_99: 1-day 99 % VaR (negative number = loss).
        cvar_99: Conditional VaR (Expected Shortfall).
        contribution: Probability-weighted VaR contribution.
        risk_drivers: Per-asset contribution to the scenario VaR.
    """

    from_regime: int
    to_regime: int
    to_regime_name: str
    probability: float
    var_99: float
    cvar_99: float
    contribution: float
    risk_drivers: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_regime": self.from_regime,
            "to_regime": self.to_regime,
            "to_regime_name": self.to_regime_name,
            "probability": round(self.probability, 4),
            "var_99": round(self.var_99, 6),
            "cvar_99": round(self.cvar_99, 6),
            "contribution": round(self.contribution, 6),
            "risk_drivers": {k: round(v, 6) for k, v in self.risk_drivers.items()},
        }


@dataclass
class HedgeRecommendation:
    """Single hedging recommendation.

    Attributes:
        scenario: Target regime transition scenario.
        instrument: Suggested hedge instrument.
        rationale: Reason for the recommendation.
        urgency: low / medium / high.
    """

    scenario: str
    instrument: str
    rationale: str
    urgency: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "instrument": self.instrument,
            "rationale": self.rationale,
            "urgency": self.urgency,
        }


@dataclass
class TailRiskResult:
    """Complete tail risk analysis output.

    Attributes:
        current_regime: Active regime.
        weighted_var: Transition-probability-weighted VaR.
        weighted_cvar: Transition-probability-weighted CVaR.
        scenarios: Per-scenario VaR breakdown.
        worst_scenario: Name of the highest-loss scenario.
        worst_scenario_var: VaR of the worst scenario.
        tail_risk_driver: Primary asset driving tail risk.
        hedge_recommendations: Instrument-specific hedges.
        portfolio_weights: Weights used for computation.
    """

    current_regime: int
    weighted_var: float
    weighted_cvar: float
    scenarios: List[ScenarioVaR]
    worst_scenario: str
    worst_scenario_var: float
    tail_risk_driver: str
    hedge_recommendations: List[HedgeRecommendation]
    portfolio_weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime,
            "weighted_var": round(self.weighted_var, 6),
            "weighted_cvar": round(self.weighted_cvar, 6),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "worst_scenario": self.worst_scenario,
            "worst_scenario_var": round(self.worst_scenario_var, 6),
            "tail_risk_driver": self.tail_risk_driver,
            "hedge_recommendations": [h.to_dict() for h in self.hedge_recommendations],
            "portfolio_weights": {
                k: round(v, 4) for k, v in self.portfolio_weights.items()
            },
        }


# ─── Tail Risk Analyzer ──────────────────────────────────────────


class TailRiskAnalyzer:
    """Regime-conditional tail risk attribution engine.

    Computes separate covariance matrices for each regime, derives
    scenario-conditional VaR / CVaR, and generates hedge recommendations.

    Args:
        confidence_level: Percentile for VaR (default 0.99 → 99 %).
        min_obs_per_regime: Minimum observations for covariance estimation.

    Example:
        >>> tra = TailRiskAnalyzer()
        >>> tra.fit(prices, regime_series)
        >>> result = tra.analyze({"SPX": 0.6, "TLT": 0.3, "GLD": 0.1}, 1,
        ...                      {1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05})
        >>> print(result.weighted_var)
    """

    def __init__(
        self,
        confidence_level: float = 0.99,
        min_obs_per_regime: int = 30,
    ) -> None:
        self.confidence_level = confidence_level
        self.min_obs_per_regime = min_obs_per_regime

        # {regime: covariance DataFrame}
        self._cov_matrices: Dict[int, pd.DataFrame] = {}
        # {regime: mean return Series}
        self._mean_returns: Dict[int, pd.Series] = {}
        # {regime: return DataFrame for historical simulation}
        self._regime_returns: Dict[int, pd.DataFrame] = {}
        self._assets: List[str] = []
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── Fitting ───────────────────────────────────────────────────

    def fit(
        self,
        market_data: pd.DataFrame,
        regime_series: pd.Series,
        assets: Optional[List[str]] = None,
    ) -> None:
        """Estimate per-regime covariance matrices and mean returns.

        Args:
            market_data: DataFrame with asset return columns.
            regime_series: Regime labels aligned to market_data.
            assets: Asset tickers (auto-detected if None).

        Raises:
            ValueError: If inputs are empty.
        """
        if market_data.empty or regime_series.empty:
            raise ValueError("market_data and regime_series must be non-empty")

        # Auto-detect return columns
        if assets is None:
            assets = [
                c.replace("_returns", "")
                for c in market_data.columns
                if c.endswith("_returns")
            ]
        if not assets:
            assets = list(
                market_data.select_dtypes(include=[np.number]).columns[:6]
            )
        self._assets = assets

        ret_cols = [
            f"{a}_returns" if f"{a}_returns" in market_data.columns else a
            for a in assets
        ]
        existing = [c for c in ret_cols if c in market_data.columns]
        if not existing:
            raise ValueError("No return columns found in market_data")

        common_idx = market_data.index.intersection(regime_series.index)
        returns = market_data.loc[common_idx, existing].dropna()
        regimes = regime_series.loc[returns.index]

        for regime_id in sorted(regimes.unique()):
            mask = regimes == regime_id
            r = returns[mask]
            if len(r) < self.min_obs_per_regime:
                logger.info(
                    f"Regime {regime_id}: only {len(r)} obs, "
                    f"using full-sample fallback for cov"
                )
                r = returns  # fallback

            self._cov_matrices[regime_id] = r.cov()
            self._mean_returns[regime_id] = r.mean()
            self._regime_returns[regime_id] = r

        self._is_fitted = True
        logger.info(
            f"TailRiskAnalyzer fitted: {len(self._cov_matrices)} regime covariance matrices, "
            f"{len(existing)} assets"
        )

    # ── Analysis ──────────────────────────────────────────────────

    def analyze(
        self,
        portfolio_weights: Dict[str, float],
        current_regime: int,
        transition_probs: Optional[Dict[int, float]] = None,
    ) -> TailRiskResult:
        """Run full tail-risk attribution.

        Args:
            portfolio_weights: Asset → weight mapping.
            current_regime: Current regime (1-4).
            transition_probs: Regime → probability (sums to ~1).
                Defaults to 70 % stay + uniform for others.

        Returns:
            TailRiskResult with scenario VaR, CVaR, drivers, hedges.
        """
        if not self._is_fitted:
            raise RuntimeError("TailRiskAnalyzer not fitted. Call fit() first.")

        # Default transition probs
        if transition_probs is None:
            transition_probs = self._default_transition_probs(current_regime)

        # Normalise weights to assets we have data for
        w = self._align_weights(portfolio_weights)

        scenarios: List[ScenarioVaR] = []
        for regime_id, prob in transition_probs.items():
            if prob <= 0:
                continue
            sv = self._compute_scenario_var(w, current_regime, regime_id, prob)
            scenarios.append(sv)

        # Weighted VaR / CVaR
        weighted_var = sum(s.contribution for s in scenarios)
        weighted_cvar = sum(
            s.probability * s.cvar_99 for s in scenarios
        )

        # Worst scenario
        worst = min(scenarios, key=lambda s: s.var_99) if scenarios else None
        worst_name = worst.to_regime_name if worst else "N/A"
        worst_var = worst.var_99 if worst else 0.0

        # Primary tail risk driver
        driver = self._identify_tail_risk_driver(scenarios)

        # Hedge recommendations
        hedges = self._generate_hedges(current_regime, scenarios)

        return TailRiskResult(
            current_regime=current_regime,
            weighted_var=weighted_var,
            weighted_cvar=weighted_cvar,
            scenarios=scenarios,
            worst_scenario=worst_name,
            worst_scenario_var=worst_var,
            tail_risk_driver=driver,
            hedge_recommendations=hedges,
            portfolio_weights=portfolio_weights,
        )

    def get_regime_covariance(self, regime: int) -> Optional[pd.DataFrame]:
        """Return the covariance matrix for a regime (or None)."""
        return self._cov_matrices.get(regime)

    # ── Private helpers ───────────────────────────────────────────

    def _align_weights(self, weights: Dict[str, float]) -> np.ndarray:
        """Align portfolio weights to fitted asset order."""
        w = np.zeros(len(self._assets))
        for i, asset in enumerate(self._assets):
            w[i] = weights.get(asset, 0.0)

        total = np.sum(np.abs(w))
        if total > 0:
            w = w / total
        else:
            w = np.ones(len(self._assets)) / len(self._assets)
        return w

    def _compute_scenario_var(
        self,
        w: np.ndarray,
        from_regime: int,
        to_regime: int,
        probability: float,
    ) -> ScenarioVaR:
        """Compute VaR/CVaR for a single transition scenario."""
        cov = self._get_cov_matrix(to_regime)
        mu = self._get_mean_returns(to_regime)

        # Parametric approach: portfolio return ~ N(w'mu, w'Σw)
        port_mu = float(w @ mu)
        port_var_param = float(w @ cov @ w)
        port_std = np.sqrt(port_var_param) if port_var_param > 0 else 1e-8

        # z-score for confidence level
        from scipy.stats import norm

        z = norm.ppf(1 - self.confidence_level)
        var_99 = port_mu + z * port_std  # negative number

        # CVaR (expected shortfall) under normality
        cvar_99 = port_mu - port_std * norm.pdf(z) / (1 - self.confidence_level)

        # Historical simulation fallback if we have enough data
        regime_rets = self._regime_returns.get(to_regime)
        if regime_rets is not None and len(regime_rets) >= 50:
            port_rets = (regime_rets.values @ w[:regime_rets.shape[1]])
            hist_var = float(np.percentile(port_rets, (1 - self.confidence_level) * 100))
            # Blend parametric and historical
            var_99 = 0.5 * var_99 + 0.5 * hist_var
            tail_rets = port_rets[port_rets <= hist_var]
            if len(tail_rets) > 0:
                cvar_99 = 0.5 * cvar_99 + 0.5 * float(np.mean(tail_rets))

        # Risk drivers: marginal contribution of each asset
        risk_drivers: Dict[str, float] = {}
        for i, asset in enumerate(self._assets):
            if port_std > 0:
                marginal = float((cov @ w)[i] * w[i] / port_var_param)
            else:
                marginal = 0.0
            risk_drivers[asset] = marginal

        return ScenarioVaR(
            from_regime=from_regime,
            to_regime=to_regime,
            to_regime_name=REGIME_NAMES.get(to_regime, f"Regime {to_regime}"),
            probability=probability,
            var_99=var_99,
            cvar_99=cvar_99,
            contribution=probability * var_99,
            risk_drivers=risk_drivers,
        )

    def _get_cov_matrix(self, regime: int) -> np.ndarray:
        """Return covariance matrix for regime, with fallback."""
        if regime in self._cov_matrices:
            return self._cov_matrices[regime].values
        # Fallback: average of available
        if self._cov_matrices:
            mats = list(self._cov_matrices.values())
            return np.mean([m.values for m in mats], axis=0)
        n = len(self._assets)
        return np.eye(n) * 0.0004  # ~2% daily vol

    def _get_mean_returns(self, regime: int) -> np.ndarray:
        """Return mean return vector for regime, with fallback."""
        if regime in self._mean_returns:
            return self._mean_returns[regime].values
        if self._mean_returns:
            vecs = list(self._mean_returns.values())
            return np.mean([v.values for v in vecs], axis=0)
        return np.zeros(len(self._assets))

    def _default_transition_probs(
        self, current_regime: int
    ) -> Dict[int, float]:
        """Default transition probabilities: 70 % stay, 10 % each other."""
        probs: Dict[int, float] = {}
        for r in [1, 2, 3, 4]:
            probs[r] = 0.70 if r == current_regime else 0.10
        return probs

    def _identify_tail_risk_driver(self, scenarios: List[ScenarioVaR]) -> str:
        """Identify the asset contributing most to worst-case VaR."""
        if not scenarios:
            return "N/A"
        worst = min(scenarios, key=lambda s: s.var_99)
        if not worst.risk_drivers:
            return "N/A"
        return max(worst.risk_drivers, key=lambda k: abs(worst.risk_drivers[k]))

    def _generate_hedges(
        self, current_regime: int, scenarios: List[ScenarioVaR]
    ) -> List[HedgeRecommendation]:
        """Generate instrument-specific hedge recommendations."""
        hedges: List[HedgeRecommendation] = []

        # Sort scenarios by (negative) contribution — worst first
        high_risk = [
            s for s in sorted(scenarios, key=lambda s: s.contribution)
            if s.to_regime != current_regime and s.probability > 0.05
        ]

        for sc in high_risk[:3]:  # top 3 risk scenarios
            key = (current_regime, sc.to_regime)
            tmpl = _HEDGE_TEMPLATES.get(key)
            if tmpl:
                urgency = (
                    "high" if sc.probability > 0.30
                    else "medium" if sc.probability > 0.15
                    else "low"
                )
                hedges.append(
                    HedgeRecommendation(
                        scenario=f"Transition to {sc.to_regime_name}",
                        instrument=tmpl["instrument"],
                        rationale=tmpl["rationale"],
                        urgency=urgency,
                    )
                )
            else:
                hedges.append(
                    HedgeRecommendation(
                        scenario=f"Transition to {sc.to_regime_name}",
                        instrument="Reduce exposure / increase cash",
                        rationale="Generic risk reduction for unmodelled transition",
                        urgency="low",
                    )
                )

        return hedges
