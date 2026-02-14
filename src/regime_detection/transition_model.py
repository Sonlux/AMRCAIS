"""
Regime Transition Probability Model for AMRCAIS.

Predicts the probability of transitioning from the current regime to each
other regime within a forward-looking horizon (default 30 trading days).

**Two-pronged approach:**
1. HMM transition matrix — learned regime-to-regime persistence / switching rates.
2. Logistic regression on five leading indicators that historically signal
   regime transitions ahead of time.

The final forward probabilities blend HMM base-rates with indicator-adjusted
probabilities using a confidence-weighted average.

Leading indicator candidates (ranked by historical predictive power):
    1. Disagreement Index trend (rising disagreement → transition)
    2. VIX term structure slope (backwardation → risk-off incoming)
    3. Credit spread momentum (widening → risk-off)
    4. Equity-bond correlation change (decorrelation breaking → regime shift)
    5. Yield curve butterfly movement (curvature → confusion → transition)

Classes:
    RegimeTransitionModel: Forward-looking transition probability estimator.

Example:
    >>> from src.regime_detection.transition_model import RegimeTransitionModel
    >>> model = RegimeTransitionModel()
    >>> model.fit(market_data, regime_history)
    >>> probs = model.predict_transition_probabilities(
    ...     current_regime=1, leading_indicators=indicators
    ... )
    >>> print(probs)
    {1: 0.55, 2: 0.25, 3: 0.12, 4: 0.08}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Regime name mapping (matches ensemble.py)
REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}


@dataclass
class TransitionForecast:
    """Forward-looking regime transition forecast.

    Attributes:
        current_regime: Current regime id (1-4).
        horizon_days: Forward horizon in trading days.
        hmm_probs: Raw HMM-derived transition probabilities {regime_id: prob}.
        indicator_probs: Logistic-regression-adjusted probabilities.
        blended_probs: Final blended probabilities.
        leading_indicators: Snapshot of indicator values used.
        transition_risk: Probability of *any* regime change (1 - persistence).
        most_likely_next: Regime id with highest non-self probability.
        confidence: Overall confidence in the forecast (0-1).
    """

    current_regime: int
    horizon_days: int
    hmm_probs: Dict[int, float]
    indicator_probs: Dict[int, float]
    blended_probs: Dict[int, float]
    leading_indicators: Dict[str, float]
    transition_risk: float
    most_likely_next: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "current_regime": self.current_regime,
            "horizon_days": self.horizon_days,
            "hmm_probs": self.hmm_probs,
            "indicator_probs": self.indicator_probs,
            "blended_probs": self.blended_probs,
            "leading_indicators": self.leading_indicators,
            "transition_risk": self.transition_risk,
            "most_likely_next": self.most_likely_next,
            "most_likely_next_name": REGIME_NAMES.get(self.most_likely_next, "Unknown"),
            "confidence": self.confidence,
        }


class RegimeTransitionModel:
    """Forward-looking regime transition probability estimator.

    Combines HMM-learned transition dynamics with a logistic regression
    model trained on five leading indicator features.

    Args:
        horizon_days: Default forward-looking horizon (trading days).
        hmm_weight: Weight for HMM base-rate probabilities in the blend.
        min_history: Minimum regime history length required for fitting.

    Example:
        >>> model = RegimeTransitionModel(horizon_days=30)
        >>> model.fit(data, regime_series)
        >>> forecast = model.predict(current_regime=1, data=latest_data)
    """

    # Five leading indicators
    INDICATOR_NAMES: List[str] = [
        "disagreement_trend",
        "vix_term_structure_slope",
        "credit_spread_momentum",
        "equity_bond_corr_change",
        "yield_curve_butterfly",
    ]

    def __init__(
        self,
        horizon_days: int = 30,
        hmm_weight: float = 0.40,
        min_history: int = 60,
    ) -> None:
        self.horizon_days = horizon_days
        self.hmm_weight = hmm_weight
        self.indicator_weight = 1.0 - hmm_weight
        self.min_history = min_history

        # Learned parameters
        self._transition_matrix: Optional[np.ndarray] = None  # 4x4
        self._logreg_weights: Optional[np.ndarray] = None     # (4, n_features+1) per from-regime
        self._logreg_models: Dict[int, Dict[str, np.ndarray]] = {}  # per-regime logistic params
        self._is_fitted: bool = False

        # Diagnostics
        self._fit_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    def fit(
        self,
        market_data: pd.DataFrame,
        regime_series: pd.Series,
        disagreement_series: Optional[pd.Series] = None,
    ) -> None:
        """Fit transition model on historical data.

        Args:
            market_data: DataFrame with at least SPX, TLT, VIX, GLD columns.
            regime_series: Series of regime ids (1-4) indexed by date.
            disagreement_series: Optional series of ensemble disagreement values.

        Raises:
            ValueError: If insufficient data for fitting.
        """
        if len(regime_series) < self.min_history:
            raise ValueError(
                f"Need >= {self.min_history} regime observations, got {len(regime_series)}"
            )

        # 1. Build empirical transition matrix
        self._transition_matrix = self._build_transition_matrix(regime_series)

        # 2. Compute leading indicator features
        indicators_df = self._compute_indicators(
            market_data, regime_series, disagreement_series
        )

        # 3. Build per-regime logistic regression models
        self._fit_logistic_models(indicators_df, regime_series)

        self._is_fitted = True
        logger.info(
            "RegimeTransitionModel fitted on %d observations", len(regime_series)
        )

    def predict(
        self,
        current_regime: int,
        market_data: pd.DataFrame,
        disagreement: Optional[float] = None,
        disagreement_series: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
        horizon_days: Optional[int] = None,
    ) -> TransitionForecast:
        """Predict forward transition probabilities.

        Args:
            current_regime: Current regime id (1-4).
            market_data: Recent market data for indicator computation.
            disagreement: Current disagreement index value.
            disagreement_series: Recent disagreement history.
            regime_series: Recent regime history (for indicator computation).
            horizon_days: Override default horizon.

        Returns:
            TransitionForecast with blended probabilities.
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        horizon = horizon_days or self.horizon_days

        # 1. HMM base-rate: raise transition matrix to horizon power
        hmm_probs = self._hmm_forward_probs(current_regime, horizon)

        # 2. Compute current indicators
        indicators = self._compute_current_indicators(
            market_data, disagreement, disagreement_series, regime_series
        )

        # 3. Logistic regression adjustment
        indicator_probs = self._logistic_predict(current_regime, indicators)

        # 4. Blend
        blended = self._blend_probabilities(hmm_probs, indicator_probs)

        # 5. Derive summary metrics
        persistence = blended.get(current_regime, 0.0)
        transition_risk = 1.0 - persistence

        # Most likely *different* regime
        non_self = {r: p for r, p in blended.items() if r != current_regime}
        most_likely_next = max(non_self, key=non_self.get) if non_self else current_regime

        # Confidence: higher when HMM and indicator models agree
        hmm_top = max(hmm_probs, key=hmm_probs.get)
        ind_top = max(indicator_probs, key=indicator_probs.get)
        agreement = 1.0 if hmm_top == ind_top else 0.6
        confidence = min(1.0, agreement * (1.0 - self._entropy(blended) / np.log(4)))

        forecast = TransitionForecast(
            current_regime=current_regime,
            horizon_days=horizon,
            hmm_probs=hmm_probs,
            indicator_probs=indicator_probs,
            blended_probs=blended,
            leading_indicators=indicators,
            transition_risk=round(transition_risk, 4),
            most_likely_next=most_likely_next,
            confidence=round(confidence, 4),
        )

        logger.info(
            "Transition forecast: regime=%d, risk=%.2f, next=%d (conf=%.2f)",
            current_regime,
            transition_risk,
            most_likely_next,
            confidence,
        )

        return forecast

    def get_transition_matrix(self) -> Optional[pd.DataFrame]:
        """Return the learned empirical transition matrix as a DataFrame."""
        if self._transition_matrix is None:
            return None
        return pd.DataFrame(
            self._transition_matrix,
            index=[REGIME_NAMES[i] for i in range(1, 5)],
            columns=[REGIME_NAMES[i] for i in range(1, 5)],
        )

    # ------------------------------------------------------------------
    # Internal: transition matrix
    # ------------------------------------------------------------------

    def _build_transition_matrix(self, regime_series: pd.Series) -> np.ndarray:
        """Count regime-to-regime transitions and normalize to probabilities.

        Args:
            regime_series: Series of regime ids.

        Returns:
            4x4 numpy array of transition probabilities.
        """
        n = 4
        counts = np.zeros((n, n), dtype=float)

        regimes = regime_series.values
        for i in range(len(regimes) - 1):
            from_r = int(regimes[i]) - 1
            to_r = int(regimes[i + 1]) - 1
            if 0 <= from_r < n and 0 <= to_r < n:
                counts[from_r, to_r] += 1

        # Add Laplace smoothing to avoid zero probabilities
        counts += 0.01
        row_sums = counts.sum(axis=1, keepdims=True)
        trans_matrix = counts / row_sums

        logger.debug("Empirical transition matrix built from %d observations", len(regimes))
        return trans_matrix

    def _hmm_forward_probs(self, current_regime: int, horizon: int) -> Dict[int, float]:
        """Compute forward probabilities by raising transition matrix to power.

        Args:
            current_regime: Current regime (1-indexed).
            horizon: Number of steps forward.

        Returns:
            Dict mapping regime id → probability.
        """
        if self._transition_matrix is None:
            # Uniform fallback
            return {r: 0.25 for r in range(1, 5)}

        # Matrix exponentiation for multi-step transition
        multi_step = np.linalg.matrix_power(self._transition_matrix, horizon)
        row = multi_step[current_regime - 1]

        # Normalize (numerical stability)
        row = np.clip(row, 0, None)
        total = row.sum()
        if total > 0:
            row = row / total

        return {r + 1: round(float(row[r]), 6) for r in range(4)}

    # ------------------------------------------------------------------
    # Internal: leading indicators
    # ------------------------------------------------------------------

    def _compute_indicators(
        self,
        market_data: pd.DataFrame,
        regime_series: pd.Series,
        disagreement_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute leading indicator time series for training.

        Args:
            market_data: Historical market data.
            regime_series: Historical regimes.
            disagreement_series: Historical disagreement.

        Returns:
            DataFrame with indicator columns aligned to regime_series index.
        """
        indicators: Dict[str, pd.Series] = {}

        n = len(market_data)

        # 1. Disagreement trend (5-day slope or synthetic)
        if disagreement_series is not None and len(disagreement_series) > 5:
            disagr_trend = disagreement_series.rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0.0,
                raw=False,
            )
            indicators["disagreement_trend"] = disagr_trend
        else:
            # Synthetic: use regime flip frequency as proxy
            regime_arr = regime_series.values
            flips = pd.Series(
                [0.0] + [1.0 if regime_arr[i] != regime_arr[i - 1] else 0.0
                         for i in range(1, len(regime_arr))],
                index=regime_series.index,
            )
            indicators["disagreement_trend"] = flips.rolling(10, min_periods=1).mean()

        # 2. VIX term structure slope (VIX vs realized vol proxy)
        if "VIX" in market_data.columns:
            vix = market_data["VIX"]
            realized = market_data.get("SPX", market_data.iloc[:, 0]).pct_change().rolling(20).std() * np.sqrt(252) * 100
            vts = (vix - realized).fillna(0)
            indicators["vix_term_structure_slope"] = vts
        else:
            indicators["vix_term_structure_slope"] = pd.Series(0.0, index=market_data.index)

        # 3. Credit spread momentum (proxy: VIX 10d momentum)
        if "VIX" in market_data.columns:
            credit_mom = market_data["VIX"].diff(10).fillna(0)
            indicators["credit_spread_momentum"] = credit_mom
        else:
            indicators["credit_spread_momentum"] = pd.Series(0.0, index=market_data.index)

        # 4. Equity-bond correlation change (20d rolling corr delta)
        if "SPX" in market_data.columns and "TLT" in market_data.columns:
            spx_ret = market_data["SPX"].pct_change()
            tlt_ret = market_data["TLT"].pct_change()
            corr_20 = spx_ret.rolling(20).corr(tlt_ret).fillna(0)
            corr_change = corr_20.diff(10).fillna(0)
            indicators["equity_bond_corr_change"] = corr_change
        else:
            indicators["equity_bond_corr_change"] = pd.Series(0.0, index=market_data.index)

        # 5. Yield curve butterfly (proxy: GLD vs TLT relative move)
        if "GLD" in market_data.columns and "TLT" in market_data.columns:
            gld_ret = market_data["GLD"].pct_change().rolling(10).sum().fillna(0)
            tlt_ret = market_data["TLT"].pct_change().rolling(10).sum().fillna(0)
            butterfly = (gld_ret - tlt_ret).fillna(0)
            indicators["yield_curve_butterfly"] = butterfly
        else:
            indicators["yield_curve_butterfly"] = pd.Series(0.0, index=market_data.index)

        result = pd.DataFrame(indicators)

        # Align to regime_series index
        common_idx = result.index.intersection(regime_series.index)
        if len(common_idx) > 0:
            result = result.loc[common_idx]
        else:
            # Fallback: align by position
            min_len = min(len(result), len(regime_series))
            result = result.iloc[:min_len]

        return result.fillna(0.0)

    def _compute_current_indicators(
        self,
        market_data: pd.DataFrame,
        disagreement: Optional[float] = None,
        disagreement_series: Optional[pd.Series] = None,
        regime_series: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute current (latest) indicator values for prediction.

        Args:
            market_data: Recent market data.
            disagreement: Current disagreement value.
            disagreement_series: Recent disagreement history.
            regime_series: Recent regime history.

        Returns:
            Dict mapping indicator name → current value.
        """
        indicators: Dict[str, float] = {}

        # 1. Disagreement trend
        if disagreement_series is not None and len(disagreement_series) >= 5:
            recent = disagreement_series.iloc[-5:].values
            slope = np.polyfit(range(5), recent, 1)[0]
            indicators["disagreement_trend"] = float(slope)
        elif disagreement is not None:
            indicators["disagreement_trend"] = float(disagreement) * 0.1  # scaled proxy
        else:
            indicators["disagreement_trend"] = 0.0

        # 2. VIX term structure slope
        if "VIX" in market_data.columns and len(market_data) > 20:
            vix_now = float(market_data["VIX"].iloc[-1])
            spx_col = "SPX" if "SPX" in market_data.columns else market_data.columns[0]
            realized = float(
                market_data[spx_col].pct_change().iloc[-20:].std() * np.sqrt(252) * 100
            )
            indicators["vix_term_structure_slope"] = vix_now - realized
        else:
            indicators["vix_term_structure_slope"] = 0.0

        # 3. Credit spread momentum (VIX 10d change proxy)
        if "VIX" in market_data.columns and len(market_data) > 10:
            indicators["credit_spread_momentum"] = float(
                market_data["VIX"].iloc[-1] - market_data["VIX"].iloc[-10]
            )
        else:
            indicators["credit_spread_momentum"] = 0.0

        # 4. Equity-bond correlation change
        if (
            "SPX" in market_data.columns
            and "TLT" in market_data.columns
            and len(market_data) > 30
        ):
            spx_ret = market_data["SPX"].pct_change().iloc[-30:]
            tlt_ret = market_data["TLT"].pct_change().iloc[-30:]
            corr_recent = float(spx_ret.iloc[-20:].corr(tlt_ret.iloc[-20:]))
            corr_prior = float(spx_ret.iloc[-30:-10].corr(tlt_ret.iloc[-30:-10]))
            indicators["equity_bond_corr_change"] = corr_recent - corr_prior
        else:
            indicators["equity_bond_corr_change"] = 0.0

        # 5. Yield curve butterfly proxy
        if (
            "GLD" in market_data.columns
            and "TLT" in market_data.columns
            and len(market_data) > 10
        ):
            gld_ret = float(
                market_data["GLD"].pct_change().iloc[-10:].sum()
            )
            tlt_ret = float(
                market_data["TLT"].pct_change().iloc[-10:].sum()
            )
            indicators["yield_curve_butterfly"] = gld_ret - tlt_ret
        else:
            indicators["yield_curve_butterfly"] = 0.0

        return indicators

    # ------------------------------------------------------------------
    # Internal: logistic regression (manual implementation)
    # ------------------------------------------------------------------

    def _fit_logistic_models(
        self,
        indicators_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> None:
        """Fit per-regime logistic regression models.

        For each 'from' regime, train a softmax model predicting which
        regime will follow at the next time step.

        Uses iterative reweighted least squares (manual) to avoid
        sklearn dependency for this simple case.

        Args:
            indicators_df: Leading indicator features.
            regime_series: Regime labels.
        """
        # Align lengths
        min_len = min(len(indicators_df), len(regime_series))
        X_all = indicators_df.iloc[:min_len].values  # (T, 5)
        regimes = regime_series.iloc[:min_len].values

        # Standardize features
        self._indicator_means = np.nanmean(X_all, axis=0)
        self._indicator_stds = np.nanstd(X_all, axis=0)
        self._indicator_stds[self._indicator_stds == 0] = 1.0
        X_std = (X_all - self._indicator_means) / self._indicator_stds

        # Add bias
        X_aug = np.column_stack([np.ones(len(X_std)), X_std])  # (T, 6)

        n_classes = 4
        n_features = X_aug.shape[1]

        for from_regime in range(1, 5):
            # Find timesteps where we're in this regime
            mask = regimes[:-1] == from_regime
            if mask.sum() < 10:
                # Not enough samples — use uniform weights
                self._logreg_models[from_regime] = {
                    "weights": np.zeros((n_classes, n_features)),
                }
                continue

            X_from = X_aug[:-1][mask]
            y_next = regimes[1:][mask] - 1  # 0-indexed targets

            # Simple multinomial logistic via gradient descent
            W = np.zeros((n_classes, n_features))
            lr = 0.01
            for _ in range(200):
                logits = X_from @ W.T  # (N, 4)
                logits -= logits.max(axis=1, keepdims=True)  # stability
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

                # One-hot targets
                one_hot = np.zeros_like(probs)
                for i, y in enumerate(y_next):
                    if 0 <= int(y) < n_classes:
                        one_hot[i, int(y)] = 1.0

                # Gradient
                grad = (probs - one_hot).T @ X_from / len(X_from)  # (4, 6)
                W -= lr * grad

            self._logreg_models[from_regime] = {"weights": W}

        logger.debug(
            "Logistic regression models fitted for %d source regimes",
            len(self._logreg_models),
        )

    def _logistic_predict(
        self,
        current_regime: int,
        indicators: Dict[str, float],
    ) -> Dict[int, float]:
        """Predict transition probabilities using logistic regression.

        Args:
            current_regime: Current regime id (1-4).
            indicators: Current leading indicator values.

        Returns:
            Dict mapping regime id → probability.
        """
        model = self._logreg_models.get(current_regime)
        if model is None or model["weights"].sum() == 0:
            # Fallback to uniform
            return {r: 0.25 for r in range(1, 5)}

        # Build feature vector
        x = np.array([indicators.get(name, 0.0) for name in self.INDICATOR_NAMES])

        # Standardize
        if hasattr(self, "_indicator_means"):
            x = (x - self._indicator_means) / self._indicator_stds

        x_aug = np.concatenate([[1.0], x])  # bias

        W = model["weights"]
        logits = W @ x_aug
        logits -= logits.max()
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        return {r + 1: round(float(probs[r]), 6) for r in range(4)}

    # ------------------------------------------------------------------
    # Internal: blending
    # ------------------------------------------------------------------

    def _blend_probabilities(
        self,
        hmm_probs: Dict[int, float],
        indicator_probs: Dict[int, float],
    ) -> Dict[int, float]:
        """Blend HMM and indicator-based probabilities.

        Args:
            hmm_probs: From HMM transition matrix.
            indicator_probs: From logistic regression.

        Returns:
            Blended probability distribution.
        """
        blended = {}
        for r in range(1, 5):
            blended[r] = (
                self.hmm_weight * hmm_probs.get(r, 0.25)
                + self.indicator_weight * indicator_probs.get(r, 0.25)
            )

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {r: round(p / total, 6) for r, p in blended.items()}

        return blended

    @staticmethod
    def _entropy(probs: Dict[int, float]) -> float:
        """Compute Shannon entropy of a probability distribution."""
        vals = np.array([p for p in probs.values() if p > 0])
        if len(vals) == 0:
            return 0.0
        return float(-np.sum(vals * np.log(vals + 1e-12)))
