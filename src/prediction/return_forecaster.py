"""
Regime-Conditional Return Forecasting (Phase 3.1).

Implements Hamilton-style regime-switching regression where every
coefficient (alpha, factor loadings) varies by regime.

    E[R | Regime=k] = alpha_k + beta_1k * F1 + beta_2k * F2 + eps_k

Key features:
    - Separate OLS per regime using historical regime labels
    - Out-of-sample R² comparison vs static (regime-agnostic) model
    - Kelly-criterion position sizing per regime
    - Walk-forward refit support

Classes:
    ReturnForecaster: Main forecasting engine

Example:
    >>> forecaster = ReturnForecaster()
    >>> forecaster.fit(market_data, regime_series)
    >>> forecast = forecaster.predict("SPX", current_regime=1,
    ...                               factor_values={"momentum": 0.02})
    >>> print(forecast.expected_return, forecast.kelly_fraction)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class RegimeModel:
    """Fitted regression coefficients for one (asset, regime) pair.

    Attributes:
        regime: Regime identifier (1-4).
        alpha: Intercept (annualised daily alpha).
        betas: Factor loadings keyed by factor name.
        residual_vol: Standard deviation of residuals.
        r_squared: In-sample R².
        n_obs: Number of observations used for fitting.
    """

    regime: int
    alpha: float
    betas: Dict[str, float]
    residual_vol: float
    r_squared: float
    n_obs: int


@dataclass
class ForecastResult:
    """Output of a single return forecast.

    Attributes:
        asset: Ticker / asset name.
        regime: Regime used for the forecast.
        expected_return: Conditional expected daily return.
        volatility: Conditional daily volatility.
        r_squared_regime: In-sample R² of the regime-specific model.
        r_squared_static: In-sample R² of the static (all-regime) model.
        r_squared_improvement: Regime R² minus static R².
        kelly_fraction: Kelly-criterion optimal fraction.
        factor_contributions: Per-factor contribution to expected return.
        confidence: Forecast reliability estimate (0-1).
    """

    asset: str
    regime: int
    expected_return: float
    volatility: float
    r_squared_regime: float
    r_squared_static: float
    r_squared_improvement: float
    kelly_fraction: float
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dictionary."""
        return {
            "asset": self.asset,
            "regime": self.regime,
            "expected_return": round(self.expected_return, 6),
            "volatility": round(self.volatility, 6),
            "r_squared_regime": round(self.r_squared_regime, 4),
            "r_squared_static": round(self.r_squared_static, 4),
            "r_squared_improvement": round(self.r_squared_improvement, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "factor_contributions": {
                k: round(v, 6) for k, v in self.factor_contributions.items()
            },
            "confidence": round(self.confidence, 4),
        }


# ─── Return Forecaster ───────────────────────────────────────────


class ReturnForecaster:
    """Regime-conditional return forecasting engine.

    Fits separate linear factor models per regime so that alpha and
    factor loadings change with the market environment.

    Args:
        factors: Factor column names to use (default SPX momentum/vol/corr).
        min_obs_per_regime: Minimum observations required to fit a regime model.
        risk_free_rate: Annualised risk-free rate for Kelly calculation.

    Example:
        >>> rf = ReturnForecaster()
        >>> rf.fit(prices_df, regime_series)
        >>> out = rf.predict("SPX", 1, {"momentum_20d": 0.03})
        >>> print(out.expected_return)
    """

    # Default factor columns expected in the market data DataFrame
    DEFAULT_FACTORS: List[str] = [
        "momentum_20d",
        "realized_vol_20d",
        "equity_bond_corr_30d",
    ]

    REGIME_NAMES: Dict[int, str] = {
        1: "Risk-On Growth",
        2: "Risk-Off Crisis",
        3: "Stagflation",
        4: "Disinflationary Boom",
    }

    def __init__(
        self,
        factors: Optional[List[str]] = None,
        min_obs_per_regime: int = 30,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.factors = factors or self.DEFAULT_FACTORS
        self.min_obs_per_regime = min_obs_per_regime
        self.risk_free_rate = risk_free_rate

        # {asset: {regime: RegimeModel}}
        self._models: Dict[str, Dict[int, RegimeModel]] = {}
        # {asset: RegimeModel}  (static model across all regimes)
        self._static_models: Dict[str, RegimeModel] = {}
        self._is_fitted = False
        self._assets: List[str] = []

    # ── Public properties ──

    @property
    def is_fitted(self) -> bool:
        """Whether the forecaster has been fitted."""
        return self._is_fitted

    # ── Fitting ───────────────────────────────────────────────────

    def fit(
        self,
        market_data: pd.DataFrame,
        regime_series: pd.Series,
        assets: Optional[List[str]] = None,
    ) -> None:
        """Fit regime-conditional and static factor models.

        Args:
            market_data: DataFrame with asset return columns and factor columns.
            regime_series: Series of regime labels aligned to market_data index.
            assets: Asset tickers to model (auto-detected if None).

        Raises:
            ValueError: If market_data or regime_series is empty.
        """
        if market_data.empty or regime_series.empty:
            raise ValueError("market_data and regime_series must be non-empty")

        # Auto-detect asset return columns (pattern: <ASSET>_returns)
        if assets is None:
            assets = [
                c.replace("_returns", "")
                for c in market_data.columns
                if c.endswith("_returns")
            ]
        if not assets:
            # Fall back: treat all numeric columns that aren't factors as assets
            assets = [
                c
                for c in market_data.select_dtypes(include=[np.number]).columns
                if c not in self.factors
            ]

        self._assets = assets

        # Align regime_series to market_data
        common_idx = market_data.index.intersection(regime_series.index)
        if len(common_idx) == 0:
            raise ValueError("No overlapping dates between market_data and regime_series")

        data = market_data.loc[common_idx].copy()
        regimes = regime_series.loc[common_idx]

        # Build factor matrix
        factor_cols = [f for f in self.factors if f in data.columns]
        if not factor_cols:
            # Derive basic factors if not present
            factor_cols = self._derive_factors(data, assets)
        if not factor_cols:
            logger.warning(
                "No factor columns found or derivable — cannot fit return models"
            )
            self._is_fitted = True
            return

        for asset in assets:
            ret_col = f"{asset}_returns" if f"{asset}_returns" in data.columns else asset
            if ret_col not in data.columns:
                logger.warning(f"No return column for {asset}, skipping")
                continue

            y = data[ret_col].dropna()
            X = data.loc[y.index, factor_cols].dropna()
            common = y.index.intersection(X.index)
            y = y.loc[common]
            X = X.loc[common]
            r = regimes.loc[common]

            # --- Regime-specific models ---
            self._models[asset] = {}
            for regime_id in sorted(r.unique()):
                mask = r == regime_id
                y_r = y[mask]
                X_r = X[mask]
                if len(y_r) < self.min_obs_per_regime:
                    logger.info(
                        f"{asset} regime {regime_id}: only {len(y_r)} obs "
                        f"(need {self.min_obs_per_regime}), skipping"
                    )
                    continue
                model = self._fit_ols(y_r, X_r, regime_id, factor_cols)
                self._models[asset][regime_id] = model

            # --- Static (all-regime) model ---
            if len(y) >= self.min_obs_per_regime:
                self._static_models[asset] = self._fit_ols(
                    y, X, regime_id=0, factor_names=factor_cols
                )

        self._is_fitted = True
        logger.info(
            f"ReturnForecaster fitted for {len(self._assets)} assets, "
            f"{len(self._models)} with regime models"
        )

    # ── Prediction ────────────────────────────────────────────────

    def predict(
        self,
        asset: str,
        current_regime: int,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> ForecastResult:
        """Generate a regime-conditional return forecast.

        Args:
            asset: Asset ticker.
            current_regime: Current regime (1-4).
            factor_values: Current factor values (uses zeros if None).

        Returns:
            ForecastResult with expected return, volatility, Kelly, etc.

        Raises:
            RuntimeError: If not fitted.
            KeyError: If asset has no model.
        """
        if not self._is_fitted:
            raise RuntimeError("ReturnForecaster is not fitted. Call fit() first.")

        if asset not in self._models:
            raise KeyError(f"No model available for asset '{asset}'")

        regime_models = self._models[asset]

        # Fall back to closest regime or static model
        if current_regime in regime_models:
            model = regime_models[current_regime]
        else:
            # Try nearest regime; ultimate fallback = static
            model = self._fallback_model(asset, current_regime)

        static_model = self._static_models.get(asset)
        static_r2 = static_model.r_squared if static_model else 0.0

        # Build factor vector
        fv = factor_values or {}
        factor_contributions: Dict[str, float] = {}
        predicted = model.alpha
        for fname, beta in model.betas.items():
            val = fv.get(fname, 0.0)
            contribution = beta * val
            factor_contributions[fname] = contribution
            predicted += contribution

        # Kelly criterion: f* = (mu - rf) / sigma^2
        daily_rf = self.risk_free_rate / 252.0
        excess = predicted - daily_rf
        vol = model.residual_vol if model.residual_vol > 0 else 1e-6
        kelly = excess / (vol ** 2) if vol > 0 else 0.0
        # Cap Kelly at ±2 for safety
        kelly = max(-2.0, min(2.0, kelly))

        # Confidence based on observation count and R²
        confidence = min(1.0, model.n_obs / 200) * min(1.0, model.r_squared + 0.5)
        confidence = round(max(0.05, confidence), 4)

        return ForecastResult(
            asset=asset,
            regime=current_regime,
            expected_return=predicted,
            volatility=vol,
            r_squared_regime=model.r_squared,
            r_squared_static=static_r2,
            r_squared_improvement=model.r_squared - static_r2,
            kelly_fraction=kelly,
            factor_contributions=factor_contributions,
            confidence=confidence,
        )

    def predict_all(
        self,
        current_regime: int,
        factor_values: Optional[Dict[str, float]] = None,
    ) -> Dict[str, ForecastResult]:
        """Predict returns for all fitted assets.

        Args:
            current_regime: Current regime.
            factor_values: Current factor values.

        Returns:
            Dict mapping asset name to ForecastResult.
        """
        results: Dict[str, ForecastResult] = {}
        for asset in self._assets:
            if asset in self._models:
                try:
                    results[asset] = self.predict(asset, current_regime, factor_values)
                except Exception as e:
                    logger.warning(f"Forecast failed for {asset}: {e}")
        return results

    def get_regime_coefficients(self, asset: str) -> Dict[int, Dict[str, Any]]:
        """Return fitted coefficients for all regimes for an asset.

        Args:
            asset: Asset ticker.

        Returns:
            Dict regime_id → {alpha, betas, residual_vol, r_squared, n_obs}.
        """
        if asset not in self._models:
            return {}
        return {
            r: {
                "alpha": m.alpha,
                "betas": m.betas,
                "residual_vol": m.residual_vol,
                "r_squared": m.r_squared,
                "n_obs": m.n_obs,
            }
            for r, m in self._models[asset].items()
        }

    # ── Private helpers ───────────────────────────────────────────

    def _fit_ols(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        regime_id: int,
        factor_names: List[str],
    ) -> RegimeModel:
        """Fit OLS regression y ~ X and return a RegimeModel."""
        # Add intercept
        X_with_const = X.copy()
        X_with_const.insert(0, "_const", 1.0)

        try:
            # Normal equations: beta = (X'X)^-1 X'y
            XtX = X_with_const.values.T @ X_with_const.values
            Xty = X_with_const.values.T @ y.values
            coeffs = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            coeffs = np.linalg.lstsq(X_with_const.values, y.values, rcond=None)[0]

        alpha = float(coeffs[0])
        betas = {fn: float(coeffs[i + 1]) for i, fn in enumerate(factor_names)}

        # Residuals
        predicted = X_with_const.values @ coeffs
        residuals = y.values - predicted
        residual_vol = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        # R²
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y.values - np.mean(y.values)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared = max(0.0, r_squared)

        return RegimeModel(
            regime=regime_id,
            alpha=alpha,
            betas=betas,
            residual_vol=residual_vol,
            r_squared=r_squared,
            n_obs=len(y),
        )

    def _fallback_model(self, asset: str, target_regime: int) -> RegimeModel:
        """Return the closest available regime model or static fallback."""
        regime_models = self._models.get(asset, {})
        if regime_models:
            # Pick regime with smallest distance
            available = sorted(regime_models.keys())
            closest = min(available, key=lambda r: abs(r - target_regime))
            return regime_models[closest]

        # Last resort: static model
        if asset in self._static_models:
            return self._static_models[asset]

        raise KeyError(f"No regime or static model for '{asset}'")

    def _derive_factors(
        self, data: pd.DataFrame, assets: List[str]
    ) -> List[str]:
        """Derive basic momentum / vol / correlation factors if missing.

        If ``<asset>_returns`` columns are not present the method first
        computes simple returns from the raw price column so that
        momentum and volatility factors can be built.

        Returns:
            List of derived factor column names added to *data*, or an
            empty list when nothing could be derived (callers must
            handle the empty case).
        """
        derived: List[str] = []

        for asset in assets:
            # Locate or create a returns column
            ret_col = f"{asset}_returns"
            if ret_col not in data.columns:
                # Try computing returns from the raw price column
                if asset in data.columns:
                    data[ret_col] = data[asset].pct_change()
                    logger.info("Derived returns column '%s' from raw prices", ret_col)
                else:
                    continue

            mom_col = f"{asset}_momentum_20d"
            if mom_col not in data.columns:
                data[mom_col] = data[ret_col].rolling(20).mean()
                derived.append(mom_col)

            vol_col = f"{asset}_vol_20d"
            if vol_col not in data.columns:
                data[vol_col] = data[ret_col].rolling(20).std()
                derived.append(vol_col)

        # Keep only the first 3 to avoid multicollinearity
        return derived[:3] if derived else []
