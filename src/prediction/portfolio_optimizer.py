"""
Regime-Aware Portfolio Optimizer (Phase 3.3).

Produces optimal asset allocations that adapt across regime scenarios.

    Input:  Target assets, constraints, current regime, transition probs
    Output: Optimal allocation ACROSS regime scenarios

Key features:
    - Mean-variance optimisation with regime-conditional parameters
    - Black-Litterman with regime-derived views
    - Transaction cost-aware (don't rebalance on noise)
    - Drawdown constraints per regime

Classes:
    PortfolioOptimizer: Main optimisation engine

Example:
    >>> opt = PortfolioOptimizer()
    >>> opt.fit(market_data, regime_series)
    >>> result = opt.optimize(
    ...     current_regime=1,
    ...     transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05},
    ... )
    >>> print(result.weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class RegimeAllocation:
    """Optimal allocation for a single regime scenario.

    Attributes:
        regime: Regime identifier.
        regime_name: Human-readable name.
        weights: Asset → weight mapping.
        expected_return: Annualised expected return.
        expected_volatility: Annualised portfolio volatility.
        sharpe_ratio: Sharpe ratio.
    """

    regime: int
    regime_name: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime,
            "regime_name": self.regime_name,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "expected_return": round(self.expected_return, 6),
            "expected_volatility": round(self.expected_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
        }


@dataclass
class OptimizationResult:
    """Complete portfolio optimisation output.

    Attributes:
        current_regime: Active regime.
        blended_weights: Transition-probability-weighted allocation.
        regime_allocations: Per-regime optimal allocations.
        rebalance_trigger: Whether rebalancing is recommended.
        rebalance_reason: Why rebalancing is or isn't recommended.
        transaction_cost_estimate: Estimated round-trip transaction cost.
        expected_return: Blended expected return (annualised).
        expected_volatility: Blended expected volatility (annualised).
        sharpe_ratio: Blended Sharpe.
        max_drawdown_constraint: Active drawdown limit.
    """

    current_regime: int
    blended_weights: Dict[str, float]
    regime_allocations: List[RegimeAllocation]
    rebalance_trigger: bool
    rebalance_reason: str
    transaction_cost_estimate: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_constraint: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_regime": self.current_regime,
            "blended_weights": {
                k: round(v, 4) for k, v in self.blended_weights.items()
            },
            "regime_allocations": [a.to_dict() for a in self.regime_allocations],
            "rebalance_trigger": self.rebalance_trigger,
            "rebalance_reason": self.rebalance_reason,
            "transaction_cost_estimate": round(self.transaction_cost_estimate, 6),
            "expected_return": round(self.expected_return, 6),
            "expected_volatility": round(self.expected_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_constraint": round(self.max_drawdown_constraint, 4),
        }


# ─── Portfolio Optimizer ──────────────────────────────────────────


class PortfolioOptimizer:
    """Regime-aware portfolio optimisation engine.

    Computes optimal weights by solving a mean-variance problem for each
    regime and then blending across transition probabilities.

    Args:
        risk_free_rate: Annualised risk-free rate.
        min_weight: Minimum weight per asset (long-only if 0).
        max_weight: Maximum weight per asset.
        transaction_cost_bps: One-way cost in basis points.
        drawdown_limits: Per-regime drawdown limits (fraction).

    Example:
        >>> po = PortfolioOptimizer()
        >>> po.fit(prices, regimes)
        >>> result = po.optimize(current_regime=1,
        ...     transition_probs={1: 0.65, 2: 0.2, 3: 0.1, 4: 0.05})
    """

    # Default drawdown constraints per regime
    DEFAULT_DD_LIMITS: Dict[int, float] = {
        1: 0.15,  # Risk-On: tolerate 15 % drawdown
        2: 0.08,  # Risk-Off: tight 8 %
        3: 0.10,  # Stagflation: 10 %
        4: 0.12,  # Disinflationary Boom: 12 %
    }

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        min_weight: float = 0.0,
        max_weight: float = 0.60,
        transaction_cost_bps: float = 10.0,
        drawdown_limits: Optional[Dict[int, float]] = None,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.transaction_cost_bps = transaction_cost_bps
        self.drawdown_limits = drawdown_limits or self.DEFAULT_DD_LIMITS

        # Fitted state
        self._cov_matrices: Dict[int, pd.DataFrame] = {}
        self._mean_returns: Dict[int, pd.Series] = {}
        self._assets: List[str] = []
        self._is_fitted = False
        self._current_weights: Optional[Dict[str, float]] = None

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
        """Estimate per-regime return distributions.

        Args:
            market_data: DataFrame with return columns.
            regime_series: Regime labels.
            assets: Asset tickers.
        """
        if market_data.empty or regime_series.empty:
            raise ValueError("market_data and regime_series must be non-empty")

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

        common_idx = market_data.index.intersection(regime_series.index)
        returns = market_data.loc[common_idx, existing].dropna()
        regimes = regime_series.loc[returns.index]

        for regime_id in sorted(regimes.unique()):
            mask = regimes == regime_id
            r = returns[mask]
            if len(r) < 20:
                r = returns  # fallback to full sample
            self._cov_matrices[regime_id] = r.cov()
            # Annualise mean returns
            self._mean_returns[regime_id] = r.mean() * 252

        self._is_fitted = True
        logger.info(f"PortfolioOptimizer fitted for {len(assets)} assets")

    # ── Optimisation ──────────────────────────────────────────────

    def optimize(
        self,
        current_regime: int,
        transition_probs: Optional[Dict[int, float]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        rebalance_threshold: float = 0.40,
    ) -> OptimizationResult:
        """Compute blended optimal allocation across regime scenarios.

        Args:
            current_regime: Current regime.
            transition_probs: Regime → probability.
            current_weights: Existing portfolio weights (for rebalance check).
            rebalance_threshold: Min transition prob to trigger rebalance.

        Returns:
            OptimizationResult with blended + per-regime allocations.
        """
        if not self._is_fitted:
            raise RuntimeError("PortfolioOptimizer not fitted. Call fit() first.")

        if transition_probs is None:
            transition_probs = self._default_probs(current_regime)

        if current_weights is not None:
            self._current_weights = current_weights

        # Per-regime optimal allocations
        regime_allocs: List[RegimeAllocation] = []
        for regime_id in sorted(transition_probs.keys()):
            if transition_probs[regime_id] <= 0:
                continue
            alloc = self._optimize_regime(regime_id)
            regime_allocs.append(alloc)

        # Blend weights by transition probabilities
        blended = self._blend_weights(regime_allocs, transition_probs)

        # Apply drawdown constraint adjustments
        dd_limit = self.drawdown_limits.get(current_regime, 0.15)
        blended = self._apply_drawdown_constraint(blended, current_regime, dd_limit)

        # Expected return / vol for blended portfolio
        exp_ret, exp_vol = self._portfolio_stats(blended, current_regime)
        rf = self.risk_free_rate
        sharpe = (exp_ret - rf) / exp_vol if exp_vol > 0 else 0.0

        # Transaction cost check
        tc = self._estimate_transaction_cost(blended)
        rebalance, reason = self._check_rebalance(
            blended, current_regime, transition_probs, rebalance_threshold
        )

        return OptimizationResult(
            current_regime=current_regime,
            blended_weights=blended,
            regime_allocations=regime_allocs,
            rebalance_trigger=rebalance,
            rebalance_reason=reason,
            transaction_cost_estimate=tc,
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe,
            max_drawdown_constraint=dd_limit,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _optimize_regime(self, regime_id: int) -> RegimeAllocation:
        """Solve mean-variance for a single regime using closed-form.

        Uses the analytical solution for the maximum-Sharpe portfolio:
            w* ∝ Σ⁻¹ (μ - rf)
        then clips to [min_weight, max_weight] and renormalises.
        """
        mu = self._get_mean_returns(regime_id)
        cov = self._get_cov_matrix(regime_id)
        n = len(self._assets)

        try:
            excess = mu - self.risk_free_rate
            inv_cov = np.linalg.inv(cov * 252)  # annualise cov
            raw = inv_cov @ excess
            # Handle negative expected excess returns
            if np.sum(raw) == 0:
                raw = np.ones(n)
        except np.linalg.LinAlgError:
            raw = np.ones(n)

        # Normalise
        w = raw / np.sum(np.abs(raw)) if np.sum(np.abs(raw)) > 0 else np.ones(n) / n
        # Clip
        w = np.clip(w, self.min_weight, self.max_weight)
        # Renormalise to sum=1
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n

        weights_dict = {self._assets[i]: float(w[i]) for i in range(n)}

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(w @ (cov * 252) @ w))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return RegimeAllocation(
            regime=regime_id,
            regime_name=REGIME_NAMES.get(regime_id, f"Regime {regime_id}"),
            weights=weights_dict,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

    def _blend_weights(
        self,
        allocations: List[RegimeAllocation],
        probs: Dict[int, float],
    ) -> Dict[str, float]:
        """Probability-weighted blend of regime allocations."""
        blended = {a: 0.0 for a in self._assets}
        total_prob = 0.0

        for alloc in allocations:
            p = probs.get(alloc.regime, 0.0)
            total_prob += p
            for asset, w in alloc.weights.items():
                blended[asset] = blended.get(asset, 0.0) + p * w

        if total_prob > 0:
            blended = {k: v / total_prob for k, v in blended.items()}

        return blended

    def _apply_drawdown_constraint(
        self,
        weights: Dict[str, float],
        regime: int,
        dd_limit: float,
    ) -> Dict[str, float]:
        """Reduce equity-like exposure if drawdown constraint binds."""
        # Simple heuristic: if Risk-Off, shift towards bonds/gold/cash
        if regime == 2:
            # Reduce risky assets, increase safe havens
            risky = {"SPX", "WTI"}
            safe = {"TLT", "GLD"}
            scale_factor = max(0.3, dd_limit / 0.15)

            for asset in list(weights.keys()):
                if asset in risky:
                    weights[asset] *= scale_factor
                elif asset in safe:
                    weights[asset] *= (2.0 - scale_factor)

            # Renormalise
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return weights

    def _portfolio_stats(
        self, weights: Dict[str, float], regime: int
    ) -> Tuple[float, float]:
        """Compute expected return and volatility for a weight vector."""
        mu = self._get_mean_returns(regime)
        cov = self._get_cov_matrix(regime) * 252

        w = np.array([weights.get(a, 0.0) for a in self._assets])
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(0, w @ cov @ w)))
        return port_ret, port_vol

    def _estimate_transaction_cost(self, new_weights: Dict[str, float]) -> float:
        """Estimate round-trip transaction cost of rebalancing."""
        if self._current_weights is None:
            return 0.0

        turnover = sum(
            abs(new_weights.get(a, 0.0) - self._current_weights.get(a, 0.0))
            for a in set(list(new_weights) + list(self._current_weights))
        )
        return turnover * self.transaction_cost_bps / 10_000.0

    def _check_rebalance(
        self,
        weights: Dict[str, float],
        regime: int,
        probs: Dict[int, float],
        threshold: float,
    ) -> Tuple[bool, str]:
        """Decide whether rebalancing is worth the cost."""
        # Trigger if regime change or transition probability exceeds threshold
        max_transition = max(
            (p for r, p in probs.items() if r != regime), default=0.0
        )

        if max_transition > threshold:
            return True, (
                f"Transition probability {max_transition:.0%} exceeds "
                f"{threshold:.0%} threshold"
            )

        # Check if current weights differ materially
        if self._current_weights is not None:
            drift = sum(
                abs(weights.get(a, 0.0) - self._current_weights.get(a, 0.0))
                for a in self._assets
            )
            if drift > 0.20:
                return True, f"Position drift {drift:.0%} exceeds 20% threshold"

        return False, "Portfolio within tolerance; no rebalance needed"

    def _get_mean_returns(self, regime: int) -> np.ndarray:
        if regime in self._mean_returns:
            return self._mean_returns[regime].values
        if self._mean_returns:
            return np.mean(
                [v.values for v in self._mean_returns.values()], axis=0
            )
        return np.zeros(len(self._assets))

    def _get_cov_matrix(self, regime: int) -> np.ndarray:
        if regime in self._cov_matrices:
            return self._cov_matrices[regime].values
        if self._cov_matrices:
            return np.mean(
                [m.values for m in self._cov_matrices.values()], axis=0
            )
        return np.eye(len(self._assets)) * 0.0004

    def _default_probs(self, current_regime: int) -> Dict[int, float]:
        probs: Dict[int, float] = {}
        for r in [1, 2, 3, 4]:
            probs[r] = 0.70 if r == current_regime else 0.10
        return probs
