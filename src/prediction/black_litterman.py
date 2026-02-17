"""
Black-Litterman Portfolio Optimizer for AMRCAIS (Phase 3.3).

Implements the Black-Litterman model with regime-derived views,
providing a Bayesian approach to combining market equilibrium with
AMRCAIS regime intelligence.

Theory:
    The Black-Litterman model starts with the market equilibrium
    (implied returns from market-cap weights) and combines them
    with "views" — beliefs about asset returns.  AMRCAIS derives
    these views from its regime classification:

    - Risk-On Growth (regime 1) → bullish equities, bearish bonds
    - Risk-Off Crisis (regime 2) → bullish bonds & gold, bearish equities
    - Stagflation (regime 3) → bullish gold & commodities, bearish both stocks & bonds
    - Disinflationary Boom (regime 4) → bullish tech & growth equities, bullish bonds

    The confidence in each view is set by the regime classification
    confidence and the transition probability weights.

Mathematics:
    Posterior returns:
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]

    Where:
        π  = implied equilibrium returns
        P  = pick matrix (which assets are in each view)
        Q  = view returns
        Ω  = view uncertainty matrix
        τ  = scalar (uncertainty about equilibrium)
        Σ  = covariance matrix

Classes:
    BlackLitterman: BL model with regime-derived views.

Example:
    >>> bl = BlackLitterman(risk_free_rate=0.05)
    >>> bl.fit(market_data, regime_series)
    >>> result = bl.optimize(regime=1, confidence=0.75,
    ...     transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.prediction.portfolio_optimizer import (
    REGIME_NAMES,
    OptimizationResult,
    PortfolioOptimizer,
    RegimeAllocation,
)

logger = logging.getLogger(__name__)


# ─── Regime View Definitions ─────────────────────────────────────

# Format: regime → list of (asset, expected_excess_return, confidence_weight)
# Returns are expressed as annualised excess returns (over risk-free)

REGIME_VIEWS: Dict[int, List[Tuple[str, float, float]]] = {
    # Risk-On Growth: strong equities, weak bonds
    1: [
        ("SPX", 0.12, 0.9),     # equities +12 %
        ("TLT", -0.02, 0.6),    # bonds underperform
        ("GLD", 0.02, 0.4),     # gold flat
        ("WTI", 0.08, 0.5),     # commodities up
        ("DXY", 0.01, 0.3),     # dollar neutral
    ],
    # Risk-Off Crisis: flight to safety
    2: [
        ("SPX", -0.15, 0.9),    # equities down sharply
        ("TLT", 0.10, 0.85),    # bonds rally
        ("GLD", 0.12, 0.8),     # gold safe haven
        ("WTI", -0.10, 0.6),    # crude declines
        ("DXY", 0.03, 0.4),     # dollar mixed
    ],
    # Stagflation: inflation + slow growth
    3: [
        ("SPX", -0.05, 0.7),    # equities weak
        ("TLT", -0.03, 0.6),    # bonds also weak (inflation)
        ("GLD", 0.15, 0.85),    # gold excels
        ("WTI", 0.10, 0.7),     # commodities inflation hedge
        ("DXY", -0.02, 0.4),    # dollar weakens
    ],
    # Disinflationary Boom: low inflation + growth
    4: [
        ("SPX", 0.10, 0.8),     # equities good
        ("TLT", 0.06, 0.7),     # bonds also good
        ("GLD", 0.00, 0.3),     # gold neutral
        ("WTI", 0.03, 0.4),     # commodities flat
        ("DXY", 0.02, 0.3),     # dollar neutral
    ],
}


@dataclass
class BLResult:
    """Black-Litterman model output.

    Attributes:
        posterior_returns: BL posterior expected returns.
        posterior_cov: BL posterior covariance matrix.
        optimal_weights: Optimal portfolio weights.
        equilibrium_returns: Market-implied equilibrium returns.
        view_returns: View returns (Q vector).
        view_confidence: Confidence in each view.
        regime: Source regime for views.
        method: "black_litterman"
    """

    posterior_returns: Dict[str, float] = field(default_factory=dict)
    posterior_cov: Optional[np.ndarray] = None
    optimal_weights: Dict[str, float] = field(default_factory=dict)
    equilibrium_returns: Dict[str, float] = field(default_factory=dict)
    view_returns: Dict[str, float] = field(default_factory=dict)
    view_confidence: Dict[str, float] = field(default_factory=dict)
    regime: int = 0
    method: str = "black_litterman"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "regime": self.regime,
            "posterior_returns": {k: round(v, 6) for k, v in self.posterior_returns.items()},
            "optimal_weights": {k: round(v, 4) for k, v in self.optimal_weights.items()},
            "equilibrium_returns": {k: round(v, 6) for k, v in self.equilibrium_returns.items()},
            "view_returns": {k: round(v, 6) for k, v in self.view_returns.items()},
            "view_confidence": {k: round(v, 4) for k, v in self.view_confidence.items()},
        }


class BlackLitterman:
    """Black-Litterman portfolio optimizer with regime-derived views.

    Combines the market equilibrium with AMRCAIS regime views to
    produce posterior expected returns, which are then optimised
    using mean-variance.

    Args:
        risk_free_rate: Annualised risk-free rate.
        tau: Scalar for uncertainty about the equilibrium (typically 0.025-0.05).
        market_cap_weights: Market-cap weights for equilibrium (default: equal).
        risk_aversion: Risk aversion parameter δ for equilibrium returns.
        min_weight: Minimum portfolio weight per asset.
        max_weight: Maximum portfolio weight per asset.

    Example:
        >>> bl = BlackLitterman()
        >>> bl.fit(prices_df, regimes_series)
        >>> result = bl.optimize(regime=2, confidence=0.8)
        >>> print(result.optimal_weights)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        tau: float = 0.05,
        market_cap_weights: Optional[Dict[str, float]] = None,
        risk_aversion: float = 2.5,
        min_weight: float = 0.0,
        max_weight: float = 0.60,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Will be populated in fit()
        self._assets: List[str] = []
        self._cov: Optional[np.ndarray] = None
        self._market_cap_weights: Optional[Dict[str, float]] = market_cap_weights
        self._equilibrium_returns: Optional[np.ndarray] = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ── Fitting ───────────────────────────────────────────────

    def fit(
        self,
        market_data: pd.DataFrame,
        regime_series: Optional[pd.Series] = None,
        assets: Optional[List[str]] = None,
    ) -> None:
        """Estimate covariance and equilibrium returns from historical data.

        Args:
            market_data: DataFrame with return columns (daily).
            regime_series: Optional regime labels (not used for fitting,
                           but stored for view generation).
            assets: Asset list (auto-detected if None).
        """
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

        # Get return columns
        ret_cols = [
            f"{a}_returns" if f"{a}_returns" in market_data.columns else a
            for a in assets
        ]
        existing = [c for c in ret_cols if c in market_data.columns]
        returns = market_data[existing].dropna()

        # Annualised covariance
        self._cov = returns.cov().values * 252

        # Market cap weights (default: equal weight)
        if self._market_cap_weights:
            w_mkt = np.array([self._market_cap_weights.get(a, 1.0 / len(assets)) for a in assets])
        else:
            w_mkt = np.ones(len(assets)) / len(assets)

        w_mkt = w_mkt / np.sum(w_mkt)  # normalise

        # Implied equilibrium excess returns: π = δΣw
        self._equilibrium_returns = self.risk_aversion * self._cov @ w_mkt

        self._is_fitted = True
        logger.info(
            f"Black-Litterman fitted for {len(assets)} assets. "
            f"Equilibrium returns: "
            + ", ".join(f"{a}={r:.4f}" for a, r in zip(assets, self._equilibrium_returns))
        )

    # ── Optimisation ──────────────────────────────────────────

    def optimize(
        self,
        regime: int,
        confidence: float = 0.7,
        transition_probs: Optional[Dict[int, float]] = None,
        custom_views: Optional[Dict[str, float]] = None,
    ) -> BLResult:
        """Compute Black-Litterman posterior and optimal weights.

        Args:
            regime: Current regime ID (determines views).
            confidence: Regime classification confidence (0-1).
            transition_probs: Regime → probability (blends views).
            custom_views: Optional custom views override (asset → return).

        Returns:
            BLResult with posterior returns and optimal weights.
        """
        if not self._is_fitted:
            raise RuntimeError("BlackLitterman not fitted. Call fit() first.")

        n = len(self._assets)
        Σ = self._cov
        π = self._equilibrium_returns

        # Build views from regime
        if custom_views:
            Q, P, Ω = self._custom_views_to_matrices(custom_views, confidence)
        elif transition_probs:
            Q, P, Ω = self._blended_views(transition_probs, confidence)
        else:
            Q, P, Ω = self._regime_views(regime, confidence)

        if Q is None or len(Q) == 0:
            # No views → return equilibrium
            eq_weights = self._max_sharpe_weights(π, Σ)
            return BLResult(
                posterior_returns={self._assets[i]: float(π[i]) for i in range(n)},
                posterior_cov=Σ,
                optimal_weights=eq_weights,
                equilibrium_returns={self._assets[i]: float(π[i]) for i in range(n)},
                view_returns={},
                view_confidence={},
                regime=regime,
            )

        # Black-Litterman posterior
        τΣ = self.tau * Σ
        τΣ_inv = np.linalg.inv(τΣ)

        # Posterior precision = (τΣ)⁻¹ + P'Ω⁻¹P
        Ω_inv = np.linalg.inv(Ω)
        posterior_precision = τΣ_inv + P.T @ Ω_inv @ P

        # Posterior mean
        posterior_mean = np.linalg.solve(
            posterior_precision,
            τΣ_inv @ π + P.T @ Ω_inv @ Q,
        )

        # Posterior covariance
        posterior_cov = np.linalg.inv(posterior_precision)

        # Optimal weights from posterior
        optimal_weights = self._max_sharpe_weights(posterior_mean, posterior_cov)

        # Build view metadata
        view_assets = []
        for row in range(P.shape[0]):
            idx = np.argmax(np.abs(P[row]))
            view_assets.append(self._assets[idx])

        return BLResult(
            posterior_returns={self._assets[i]: float(posterior_mean[i]) for i in range(n)},
            posterior_cov=posterior_cov,
            optimal_weights=optimal_weights,
            equilibrium_returns={self._assets[i]: float(π[i]) for i in range(n)},
            view_returns={view_assets[i]: float(Q[i]) for i in range(len(Q))},
            view_confidence={
                view_assets[i]: float(1.0 / Ω[i, i]) if Ω[i, i] > 0 else 0.0
                for i in range(len(Q))
            },
            regime=regime,
        )

    # ── View Construction ─────────────────────────────────────

    def _regime_views(
        self,
        regime: int,
        confidence: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert regime views to BL matrices.

        Args:
            regime: Regime ID.
            confidence: Regime classification confidence.

        Returns:
            Tuple of (Q, P, Ω) or (None, None, None) if no views.
        """
        views = REGIME_VIEWS.get(regime, [])
        if not views:
            return None, None, None

        # Filter to assets in our universe
        valid_views = []
        for asset, ret, view_conf in views:
            if asset in self._assets:
                valid_views.append((asset, ret, view_conf * confidence))

        if not valid_views:
            return None, None, None

        k = len(valid_views)  # number of views
        n = len(self._assets)  # number of assets

        Q = np.zeros(k)
        P = np.zeros((k, n))
        omega_diag = np.zeros(k)

        for i, (asset, ret, conf) in enumerate(valid_views):
            asset_idx = self._assets.index(asset)
            Q[i] = ret
            P[i, asset_idx] = 1.0
            # View uncertainty: inversely proportional to confidence
            # Higher confidence → lower uncertainty
            omega_diag[i] = self.tau * self._cov[asset_idx, asset_idx] / max(conf, 0.01)

        Ω = np.diag(omega_diag)
        return Q, P, Ω

    def _blended_views(
        self,
        transition_probs: Dict[int, float],
        confidence: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Create probability-weighted blended views from multiple regimes.

        Args:
            transition_probs: Regime → probability.
            confidence: Regime confidence (applied to all views).

        Returns:
            Tuple of (Q, P, Ω).
        """
        # Collect all unique asset-level views
        asset_views: Dict[str, Tuple[float, float]] = {}  # asset → (weighted_return, total_weight)

        for regime_id, prob in transition_probs.items():
            if prob <= 0.01:
                continue
            views = REGIME_VIEWS.get(regime_id, [])
            for asset, ret, view_conf in views:
                if asset in self._assets:
                    prev_ret, prev_weight = asset_views.get(asset, (0.0, 0.0))
                    weight = prob * view_conf * confidence
                    asset_views[asset] = (
                        prev_ret + ret * weight,
                        prev_weight + weight,
                    )

        if not asset_views:
            return None, None, None

        # Normalise
        final_views = []
        for asset, (weighted_ret, total_weight) in asset_views.items():
            if total_weight > 0:
                final_views.append((asset, weighted_ret / total_weight, total_weight))

        k = len(final_views)
        n = len(self._assets)

        Q = np.zeros(k)
        P = np.zeros((k, n))
        omega_diag = np.zeros(k)

        for i, (asset, ret, conf) in enumerate(final_views):
            asset_idx = self._assets.index(asset)
            Q[i] = ret
            P[i, asset_idx] = 1.0
            omega_diag[i] = self.tau * self._cov[asset_idx, asset_idx] / max(conf, 0.01)

        Ω = np.diag(omega_diag)
        return Q, P, Ω

    def _custom_views_to_matrices(
        self,
        views: Dict[str, float],
        confidence: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert custom views dict to BL matrices.

        Args:
            views: Asset → expected return.
            confidence: Overall confidence.

        Returns:
            Tuple of (Q, P, Ω).
        """
        valid = [(a, r) for a, r in views.items() if a in self._assets]
        if not valid:
            return None, None, None

        k = len(valid)
        n = len(self._assets)

        Q = np.zeros(k)
        P = np.zeros((k, n))
        omega_diag = np.zeros(k)

        for i, (asset, ret) in enumerate(valid):
            asset_idx = self._assets.index(asset)
            Q[i] = ret
            P[i, asset_idx] = 1.0
            omega_diag[i] = self.tau * self._cov[asset_idx, asset_idx] / max(confidence, 0.01)

        Ω = np.diag(omega_diag)
        return Q, P, Ω

    # ── Weight Calculation ────────────────────────────────────

    def _max_sharpe_weights(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> Dict[str, float]:
        """Maximum Sharpe ratio weights from expected returns and covariance.

        Args:
            mu: Expected returns vector.
            cov: Covariance matrix.

        Returns:
            Dict of asset → weight.
        """
        n = len(self._assets)
        rf = self.risk_free_rate

        try:
            excess = mu - rf
            inv_cov = np.linalg.inv(cov)
            raw = inv_cov @ excess

            if np.sum(np.abs(raw)) == 0:
                raw = np.ones(n)
        except np.linalg.LinAlgError:
            raw = np.ones(n)

        # Normalise
        w = raw / np.sum(np.abs(raw)) if np.sum(np.abs(raw)) > 0 else np.ones(n) / n

        # Clip to constraints
        w = np.clip(w, self.min_weight, self.max_weight)

        # Renormalise
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n

        return {self._assets[i]: float(w[i]) for i in range(n)}

    # ── Integration with PortfolioOptimizer ───────────────────

    def to_optimization_result(
        self,
        bl_result: BLResult,
        transition_probs: Optional[Dict[int, float]] = None,
    ) -> OptimizationResult:
        """Convert BLResult to standard OptimizationResult.

        This enables drop-in replacement of the standard mean-variance
        optimizer with Black-Litterman.

        Args:
            bl_result: Output from optimize().
            transition_probs: Regime probabilities.

        Returns:
            OptimizationResult compatible with the rest of AMRCAIS.
        """
        regime = bl_result.regime
        weights = bl_result.optimal_weights

        # Compute portfolio stats
        n = len(self._assets)
        w = np.array([weights.get(a, 0.0) for a in self._assets])
        mu = np.array([bl_result.posterior_returns.get(a, 0.0) for a in self._assets])
        cov = bl_result.posterior_cov if bl_result.posterior_cov is not None else self._cov

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(0, w @ cov @ w)))
        sharpe = (port_ret - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        # Create a single regime allocation from the BL result
        alloc = RegimeAllocation(
            regime=regime,
            regime_name=REGIME_NAMES.get(regime, f"Regime {regime}"),
            weights=weights,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

        return OptimizationResult(
            current_regime=regime,
            blended_weights=weights,
            regime_allocations=[alloc],
            rebalance_trigger=True,
            rebalance_reason="Black-Litterman signal",
            transaction_cost_estimate=0.0,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            max_drawdown_constraint=0.15,
        )
