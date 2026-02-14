"""
Cross-Asset Contagion Network for AMRCAIS.

Models cross-asset relationships as a dynamic directed graph rather than a
static correlation matrix.  Key analytical capabilities:

1. **Granger causality testing** — which assets lead which in each regime.
2. **Diebold-Yilmaz spillover index** — quantifies total, directional and
   net systemic connectedness using VAR forecast error variance decomposition.
3. **Contagion detection** — identifies when a shock in one asset propagates
   to others faster or more intensely than expected for the current regime.
4. **Regime-conditional network** — the graph structure itself changes with
   regimes; this is the novel insight.

Classes:
    ContagionNetwork: Full contagion analysis engine.
    SpilloverResult: Container for DY spillover decomposition.
    GrangerResult: Pairwise Granger causality test output.

Example:
    >>> from src.modules.contagion_network import ContagionNetwork
    >>> net = ContagionNetwork()
    >>> net.update_regime(1, 0.85)
    >>> result = net.analyze(market_data)
    >>> print(result["spillover_index"])
    >>> print(result["granger_network"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)

# Default asset universe
DEFAULT_ASSETS: List[str] = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"]


@dataclass
class GrangerResult:
    """Result of a pairwise Granger causality test.

    Attributes:
        cause: The causing asset name.
        effect: The affected asset name.
        f_statistic: F-statistic from the OLS test.
        p_value: p-value of the F test.
        lag: Number of lags used.
        significant: Whether causality is significant at alpha.
    """

    cause: str
    effect: str
    f_statistic: float
    p_value: float
    lag: int
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "f_stat": round(self.f_statistic, 4),
            "p_value": round(self.p_value, 4),
            "lag": self.lag,
            "significant": self.significant,
        }


@dataclass
class SpilloverResult:
    """Diebold-Yilmaz spillover decomposition.

    Attributes:
        total_spillover_index: System-wide connectedness (0-100%).
        directional_to: How much each asset spills TO others.
        directional_from: How much each asset receives FROM others.
        net_spillover: directional_to - directional_from per asset.
        pairwise: Full pairwise spillover matrix.
        assets: Asset names in order.
    """

    total_spillover_index: float
    directional_to: Dict[str, float]
    directional_from: Dict[str, float]
    net_spillover: Dict[str, float]
    pairwise: np.ndarray
    assets: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_spillover_index": round(self.total_spillover_index, 4),
            "directional_to": {k: round(v, 4) for k, v in self.directional_to.items()},
            "directional_from": {k: round(v, 4) for k, v in self.directional_from.items()},
            "net_spillover": {k: round(v, 4) for k, v in self.net_spillover.items()},
            "pairwise": self.pairwise.round(4).tolist(),
            "assets": self.assets,
        }


class ContagionNetwork(AnalyticalModule):
    """Cross-asset contagion network with Granger causality and spillover analysis.

    Models the dynamic web of cross-asset influence as a directed graph.
    The graph structure is regime-conditional — different regimes produce
    fundamentally different causal structures.

    Regime-adaptive behavior:
        - Risk-On Growth (1): Low connectedness, idiosyncratic moves dominate.
        - Risk-Off Crisis (2): High connectedness, contagion spikes, VIX leads.
        - Stagflation (3): Unusual patterns — gold and commodities decouple from equities.
        - Disinflationary Boom (4): Moderate connectedness, bonds lead.

    Args:
        config_path: Path to YAML configuration directory.
        max_lag: Maximum lag for Granger causality tests.
        var_lag: Lag order for VAR model in spillover analysis.
        forecast_horizon: Forecast horizon for variance decomposition.
        significance_level: Alpha for Granger causality tests.

    Example:
        >>> net = ContagionNetwork()
        >>> net.update_regime(2, 0.9)  # Risk-Off
        >>> result = net.analyze(market_data)
        >>> # Expect high spillover index in crisis
    """

    # Expected connectedness levels per regime (for anomaly detection)
    REGIME_CONNECTEDNESS_BASELINES: Dict[int, Dict[str, float]] = {
        1: {"total_spillover": 25.0, "std": 10.0},
        2: {"total_spillover": 55.0, "std": 12.0},
        3: {"total_spillover": 40.0, "std": 10.0},
        4: {"total_spillover": 30.0, "std": 10.0},
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_lag: int = 5,
        var_lag: int = 2,
        forecast_horizon: int = 10,
        significance_level: float = 0.05,
        assets: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="ContagionNetwork", config_path=config_path)
        self.max_lag = max_lag
        self.var_lag = var_lag
        self.forecast_horizon = forecast_horizon
        self.significance_level = significance_level
        self.assets = assets or list(DEFAULT_ASSETS)

        # Cached results
        self._last_granger: List[GrangerResult] = []
        self._last_spillover: Optional[SpilloverResult] = None
        self._network_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # AnalyticalModule interface
    # ------------------------------------------------------------------

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run full contagion network analysis.

        Args:
            data: DataFrame with asset price columns.

        Returns:
            Dict with signal, granger, spillover, network, and contagion info.
        """
        # Filter to available assets
        available = [a for a in self.assets if a in data.columns]
        if len(available) < 2:
            return {
                "signal": self.create_signal(
                    signal="neutral", strength=0.0,
                    explanation="Insufficient assets for contagion analysis",
                    regime_context=f"Regime {self.current_regime}",
                ),
                "error": "Need at least 2 assets",
            }

        # Compute returns
        returns_df = data[available].pct_change().dropna()
        if len(returns_df) < 30:
            return {
                "signal": self.create_signal(
                    signal="neutral", strength=0.0,
                    explanation="Insufficient data for contagion analysis",
                    regime_context=f"Regime {self.current_regime}",
                ),
                "error": "Need at least 30 return observations",
            }

        # 1. Granger causality
        granger_results = self._run_granger_tests(returns_df, available)
        self._last_granger = granger_results

        # 2. Diebold-Yilmaz spillover
        spillover = self._compute_spillover(returns_df, available)
        self._last_spillover = spillover

        # 3. Build network graph
        network = self._build_network(granger_results, spillover, available)

        # 4. Detect contagion anomalies
        contagion_flags = self._detect_contagion(spillover)

        # 5. Generate signal
        signal, strength, explanation = self._generate_signal(
            spillover, contagion_flags, granger_results
        )

        module_signal = self.create_signal(
            signal=signal,
            strength=strength,
            explanation=explanation,
            regime_context=f"Regime {self.current_regime} contagion network",
        )

        # Record history
        self._network_history.append({
            "total_spillover": spillover.total_spillover_index if spillover else 0,
            "n_significant_granger": sum(1 for g in granger_results if g.significant),
            "contagion_detected": any(contagion_flags.values()) if contagion_flags else False,
        })

        return {
            "signal": module_signal,
            "granger_network": [g.to_dict() for g in granger_results],
            "spillover": spillover.to_dict() if spillover else {},
            "network_graph": network,
            "contagion_flags": contagion_flags,
            "n_significant_links": sum(1 for g in granger_results if g.significant),
            "network_density": self._network_density(granger_results, len(available)),
        }

    def get_regime_parameters(self, regime: int) -> Dict[str, Any]:
        """Get expected network parameters for a regime."""
        baseline = self.REGIME_CONNECTEDNESS_BASELINES.get(regime, {})
        return {
            "expected_spillover": baseline.get("total_spillover", 30),
            "spillover_std": baseline.get("std", 10),
            "regime": regime,
        }

    # ------------------------------------------------------------------
    # Granger Causality
    # ------------------------------------------------------------------

    def _run_granger_tests(
        self,
        returns: pd.DataFrame,
        assets: List[str],
    ) -> List[GrangerResult]:
        """Pairwise Granger causality tests for all asset pairs.

        Uses an OLS-based F-test: does adding lagged values of asset X
        improve the prediction of asset Y beyond Y's own lags?

        Args:
            returns: Asset return DataFrame.
            assets: Asset names.

        Returns:
            List of GrangerResult for each pair.
        """
        results: List[GrangerResult] = []

        for cause in assets:
            for effect in assets:
                if cause == effect:
                    continue

                best = self._granger_test_pair(
                    returns[cause].values.copy(),
                    returns[effect].values.copy(),
                    max_lag=self.max_lag,
                )
                if best is not None:
                    results.append(GrangerResult(
                        cause=cause,
                        effect=effect,
                        f_statistic=best["f_stat"],
                        p_value=best["p_value"],
                        lag=best["lag"],
                        significant=best["p_value"] < self.significance_level,
                    ))

        return results

    def _granger_test_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """OLS-based Granger causality test for a single pair.

        Tests whether lagged x values improve prediction of y beyond
        y's own lagged values.

        Args:
            x: Potential cause series.
            y: Potential effect series.
            max_lag: Maximum lag to test.

        Returns:
            Dict with best lag's f_stat, p_value, lag, or None if test fails.
        """
        n = len(y)
        best_result: Optional[Dict[str, Any]] = None
        best_p = 1.0

        for lag in range(1, max_lag + 1):
            if n <= 2 * lag + 2:
                continue

            # Build restricted model: y_t = c + sum(b_j * y_{t-j})
            Y = y[lag:]
            X_restricted = np.column_stack(
                [np.ones(n - lag)] + [y[lag - j - 1: n - j - 1] for j in range(lag)]
            )

            # Build unrestricted model: y_t = c + sum(b_j * y_{t-j}) + sum(g_j * x_{t-j})
            X_unrestricted = np.column_stack(
                [X_restricted] + [x[lag - j - 1: n - j - 1] for j in range(lag)]
            )

            try:
                # Restricted OLS
                beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
                resid_r = Y - X_restricted @ beta_r
                ssr_r = float(np.sum(resid_r ** 2))

                # Unrestricted OLS
                beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
                resid_u = Y - X_unrestricted @ beta_u
                ssr_u = float(np.sum(resid_u ** 2))

                # F-test
                df1 = lag  # number of restrictions
                df2 = len(Y) - X_unrestricted.shape[1]
                if df2 <= 0 or ssr_u <= 0:
                    continue

                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                if f_stat < 0:
                    f_stat = 0.0

                # Approximate p-value using F-distribution survival function
                p_value = self._f_survival(f_stat, df1, df2)

                if p_value < best_p:
                    best_p = p_value
                    best_result = {
                        "f_stat": f_stat,
                        "p_value": p_value,
                        "lag": lag,
                    }
            except (np.linalg.LinAlgError, ValueError):
                continue

        return best_result

    @staticmethod
    def _f_survival(f_stat: float, df1: int, df2: int) -> float:
        """Approximate F-distribution survival function (p-value).

        Uses the regularized incomplete beta function approximation.
        Falls back to a rough approximation if scipy is not available.

        Args:
            f_stat: F statistic.
            df1: Numerator degrees of freedom.
            df2: Denominator degrees of freedom.

        Returns:
            Approximate p-value.
        """
        try:
            from scipy.stats import f as f_dist
            return float(f_dist.sf(f_stat, df1, df2))
        except ImportError:
            # Rough approximation: use chi-square approximation
            chi2 = f_stat * df1
            # Very rough: p ~ exp(-chi2 / 2) for large chi2
            return float(min(1.0, np.exp(-chi2 / 2.0)))

    # ------------------------------------------------------------------
    # Diebold-Yilmaz Spillover Index
    # ------------------------------------------------------------------

    def _compute_spillover(
        self,
        returns: pd.DataFrame,
        assets: List[str],
    ) -> Optional[SpilloverResult]:
        """Compute Diebold-Yilmaz spillover index via VAR forecast error
        variance decomposition.

        Simplified implementation: fits a VAR(p) model via OLS, then computes
        generalized forecast error variance decomposition.

        Args:
            returns: Asset return DataFrame.
            assets: Asset names.

        Returns:
            SpilloverResult or None if computation fails.
        """
        k = len(assets)
        T = len(returns)

        if T < self.var_lag + self.forecast_horizon + 10:
            logger.warning("Insufficient data for spillover computation")
            return None

        # Standardize returns
        ret_vals = returns[assets].values.copy()
        means = ret_vals.mean(axis=0)
        stds = ret_vals.std(axis=0)
        stds[stds == 0] = 1.0
        ret_std = (ret_vals - means) / stds

        try:
            # Fit VAR(p) via OLS
            coef_matrices, sigma_u = self._fit_var(ret_std, self.var_lag)

            # Compute generalized forecast error variance decomposition
            fevd = self._generalized_fevd(
                coef_matrices, sigma_u, self.forecast_horizon, k
            )

            # Normalize rows to 100%
            row_sums = fevd.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            fevd_norm = fevd / row_sums * 100.0

            # Spillover measures
            total_spillover = float(
                (fevd_norm.sum() - np.trace(fevd_norm)) / fevd_norm.sum() * 100.0
            )

            dir_to: Dict[str, float] = {}
            dir_from: Dict[str, float] = {}
            net: Dict[str, float] = {}

            for i, asset in enumerate(assets):
                # Directional TO: column sum minus own
                to_others = float(fevd_norm[:, i].sum() - fevd_norm[i, i])
                # Directional FROM: row sum minus own
                from_others = float(fevd_norm[i, :].sum() - fevd_norm[i, i])

                dir_to[asset] = to_others
                dir_from[asset] = from_others
                net[asset] = to_others - from_others

            return SpilloverResult(
                total_spillover_index=total_spillover,
                directional_to=dir_to,
                directional_from=dir_from,
                net_spillover=net,
                pairwise=fevd_norm,
                assets=assets,
            )

        except Exception as e:
            logger.warning("Spillover computation failed: %s", e)
            return None

    def _fit_var(
        self,
        data: np.ndarray,
        lag: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Fit a VAR(p) model via OLS.

        Args:
            data: (T, k) array of standardized returns.
            lag: VAR lag order.

        Returns:
            Tuple of (list of k×k coefficient matrices, k×k residual covariance).
        """
        T, k = data.shape

        # Build Y and X matrices
        Y = data[lag:]  # (T-lag, k)
        X_parts = [np.ones((T - lag, 1))]  # intercept
        for p in range(1, lag + 1):
            X_parts.append(data[lag - p: T - p])
        X = np.column_stack(X_parts)  # (T-lag, 1 + k*lag)

        # OLS: beta = (X'X)^{-1} X'Y
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]  # (1+k*lag, k)

        # Extract coefficient matrices (skip intercept)
        coef_matrices = []
        for p in range(lag):
            start = 1 + p * k
            coef_matrices.append(beta[start: start + k].T)  # (k, k)

        # Residual covariance
        residuals = Y - X @ beta
        sigma_u = (residuals.T @ residuals) / (T - lag - 1)

        return coef_matrices, sigma_u

    def _generalized_fevd(
        self,
        coef_matrices: List[np.ndarray],
        sigma_u: np.ndarray,
        horizon: int,
        k: int,
    ) -> np.ndarray:
        """Generalized forecast error variance decomposition (Pesaran-Shin).

        Args:
            coef_matrices: VAR coefficient matrices.
            sigma_u: Residual covariance matrix.
            horizon: Forecast horizon.
            k: Number of variables.

        Returns:
            k×k FEVD matrix where entry (i,j) = contribution of j to i's FEV.
        """
        # Compute MA coefficient matrices (VMA representation)
        psi = self._compute_vma(coef_matrices, horizon, k)

        # Generalized FEVD
        sigma_diag = np.diag(sigma_u)
        sigma_diag[sigma_diag == 0] = 1.0

        fevd = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                numerator = 0.0
                denominator = 0.0
                for h in range(horizon):
                    ej = np.zeros(k)
                    ej[j] = 1.0
                    contribution = (psi[h] @ sigma_u @ ej) ** 2 / sigma_diag[j]
                    numerator += contribution[i]

                    ei = np.zeros(k)
                    ei[i] = 1.0
                    total = ei @ psi[h] @ sigma_u @ psi[h].T @ ei
                    denominator += total

                if denominator > 0:
                    fevd[i, j] = numerator / denominator

        return fevd

    def _compute_vma(
        self,
        coef_matrices: List[np.ndarray],
        horizon: int,
        k: int,
    ) -> List[np.ndarray]:
        """Compute Vector Moving Average coefficient matrices.

        Recursively computes Psi_h from VAR coefficient matrices.

        Args:
            coef_matrices: List of VAR(p) coefficient matrices.
            horizon: Number of horizon steps.
            k: Number of variables.

        Returns:
            List of k×k MA coefficient matrices [Psi_0, Psi_1, ..., Psi_{H-1}].
        """
        p = len(coef_matrices)
        psi: List[np.ndarray] = [np.eye(k)]

        for h in range(1, horizon):
            psi_h = np.zeros((k, k))
            for j in range(min(h, p)):
                psi_h += psi[h - j - 1] @ coef_matrices[j]
            psi.append(psi_h)

        return psi

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def _build_network(
        self,
        granger_results: List[GrangerResult],
        spillover: Optional[SpilloverResult],
        assets: List[str],
    ) -> Dict[str, Any]:
        """Build a network graph representation.

        Args:
            granger_results: Pairwise Granger causality results.
            spillover: Spillover decomposition results.
            assets: Asset names.

        Returns:
            Dict with nodes and edges suitable for graph visualization.
        """
        nodes = []
        for asset in assets:
            node_data: Dict[str, Any] = {"id": asset, "label": asset}
            if spillover:
                node_data["net_spillover"] = spillover.net_spillover.get(asset, 0)
                node_data["to_others"] = spillover.directional_to.get(asset, 0)
                node_data["from_others"] = spillover.directional_from.get(asset, 0)
                # Node role: transmitter (net > 0) or receiver (net < 0)
                ns = spillover.net_spillover.get(asset, 0)
                node_data["role"] = "transmitter" if ns > 0 else "receiver"
            nodes.append(node_data)

        edges = []
        for g in granger_results:
            if g.significant:
                edge: Dict[str, Any] = {
                    "source": g.cause,
                    "target": g.effect,
                    "weight": round(1.0 - g.p_value, 4),
                    "lag": g.lag,
                    "f_stat": round(g.f_statistic, 4),
                }
                # Add regime-appropriateness coloring
                is_expected = self._is_expected_link(
                    g.cause, g.effect, self.current_regime or 1
                )
                edge["regime_expected"] = is_expected
                edges.append(edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def _is_expected_link(self, cause: str, effect: str, regime: int) -> bool:
        """Check if a causal link is expected in the current regime.

        Args:
            cause: Causing asset.
            effect: Affected asset.
            regime: Current regime.

        Returns:
            True if this link is typical for the regime.
        """
        expected_links: Dict[int, List[Tuple[str, str]]] = {
            1: [("VIX", "SPX"), ("TLT", "SPX")],
            2: [("VIX", "SPX"), ("VIX", "TLT"), ("VIX", "GLD"),
                ("SPX", "TLT"), ("SPX", "GLD"), ("SPX", "WTI")],
            3: [("WTI", "GLD"), ("DXY", "WTI"), ("GLD", "SPX")],
            4: [("TLT", "SPX"), ("TLT", "GLD"), ("DXY", "GLD")],
        }
        return (cause, effect) in expected_links.get(regime, [])

    def _network_density(
        self,
        granger_results: List[GrangerResult],
        n_assets: int,
    ) -> float:
        """Compute network density (ratio of significant links to possible links).

        Args:
            granger_results: Granger test results.
            n_assets: Number of assets.

        Returns:
            Density value in [0, 1].
        """
        possible = n_assets * (n_assets - 1)
        if possible == 0:
            return 0.0
        significant = sum(1 for g in granger_results if g.significant)
        return round(significant / possible, 4)

    # ------------------------------------------------------------------
    # Contagion detection
    # ------------------------------------------------------------------

    def _detect_contagion(
        self,
        spillover: Optional[SpilloverResult],
    ) -> Dict[str, bool]:
        """Detect contagion anomalies by comparing spillover to regime baseline.

        Args:
            spillover: Current spillover result.

        Returns:
            Dict of contagion flags.
        """
        flags: Dict[str, bool] = {
            "high_systemic_connectedness": False,
            "anomalous_transmission": False,
            "regime_mismatch": False,
        }

        if spillover is None:
            return flags

        regime = self.current_regime or 1
        baseline = self.REGIME_CONNECTEDNESS_BASELINES.get(regime, {})
        expected = baseline.get("total_spillover", 30.0)
        std = baseline.get("std", 10.0)

        total = spillover.total_spillover_index

        # High systemic connectedness
        if total > expected + 2 * std:
            flags["high_systemic_connectedness"] = True

        # Anomalous transmission: any single asset net spillover > 30 pp
        for asset, ns in spillover.net_spillover.items():
            if abs(ns) > 30:
                flags["anomalous_transmission"] = True
                break

        # Regime mismatch: crisis-level connectedness in non-crisis regime
        if regime in (1, 4) and total > 50:
            flags["regime_mismatch"] = True

        return flags

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(
        self,
        spillover: Optional[SpilloverResult],
        contagion_flags: Dict[str, bool],
        granger_results: List[GrangerResult],
    ) -> Tuple[str, float, str]:
        """Generate an aggregate signal from contagion analysis.

        Args:
            spillover: Spillover decomposition.
            contagion_flags: Contagion anomaly flags.
            granger_results: Granger test results.

        Returns:
            Tuple of (signal, strength, explanation).
        """
        regime = self.current_regime or 1
        n_sig = sum(1 for g in granger_results if g.significant)

        if spillover is None:
            return "neutral", 0.0, "Spillover computation unavailable"

        total_spill = spillover.total_spillover_index
        any_contagion = any(contagion_flags.values())

        parts: List[str] = [
            f"Spillover index: {total_spill:.1f}%.",
            f"{n_sig} significant Granger causal links detected.",
        ]

        # Determine top transmitters
        top_transmitter = max(
            spillover.net_spillover, key=spillover.net_spillover.get
        ) if spillover.net_spillover else "N/A"
        parts.append(f"Top transmitter: {top_transmitter}.")

        if any_contagion:
            triggered = [k for k, v in contagion_flags.items() if v]
            parts.append(f"Contagion flags: {', '.join(triggered)}.")

            if regime in (1, 4):
                signal = "bearish"
                strength = min(1.0, total_spill / 60.0)
            else:
                signal = "cautious"
                strength = min(1.0, total_spill / 80.0)
        else:
            if total_spill < 20:
                signal = "bullish" if regime == 1 else "neutral"
                strength = 0.3
            else:
                signal = "neutral"
                strength = min(0.7, total_spill / 100.0)

        return signal, round(strength, 2), " ".join(parts)
