"""
Multi-Timeframe Regime Detection for AMRCAIS.

Runs independent regime ensembles at daily, weekly, and monthly timeframes
using appropriate lookback windows and data resampling.  Conflicting
timeframe signals are a high-conviction trade signal:

- **Daily bearish within monthly bullish** → buy-the-dip opportunity.
- **All timeframes agree** → high-confidence directional view.
- **Timeframes diverge** → caution / reduced position sizing.

Classes:
    MultiTimeframeDetector: Orchestrates 3 timeframe-specific ensembles.
    TimeframeResult: Per-timeframe regime result.
    MultiTimeframeResult: Aggregated cross-timeframe view.

Example:
    >>> from src.regime_detection.multi_timeframe import MultiTimeframeDetector
    >>> mtf = MultiTimeframeDetector()
    >>> mtf.fit(market_data)
    >>> result = mtf.predict(market_data)
    >>> print(result.daily.regime_name)      # "Risk-On Growth"
    >>> print(result.monthly.regime_name)    # "Stagflation"
    >>> print(result.conflict_detected)      # True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.regime_detection.ensemble import RegimeEnsemble, EnsembleResult

logger = logging.getLogger(__name__)

# Human-readable regime names
REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}


@dataclass
class TimeframeResult:
    """Regime result for a single timeframe.

    Attributes:
        timeframe: "daily", "weekly", or "monthly".
        regime: Regime id (1-4).
        regime_name: Human-readable name.
        confidence: Ensemble confidence.
        disagreement: Classifier disagreement index.
        transition_warning: Whether disagreement exceeds threshold.
        duration: How long this regime has persisted at this timeframe (periods).
    """

    timeframe: str
    regime: int
    regime_name: str
    confidence: float
    disagreement: float
    transition_warning: bool
    duration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "confidence": round(self.confidence, 4),
            "disagreement": round(self.disagreement, 4),
            "transition_warning": self.transition_warning,
            "duration": self.duration,
        }


@dataclass
class MultiTimeframeResult:
    """Aggregated multi-timeframe regime view.

    Attributes:
        daily: Daily timeframe result.
        weekly: Weekly timeframe result.
        monthly: Monthly timeframe result.
        conflict_detected: True if any two timeframes disagree.
        highest_conviction: Timeframe with highest confidence.
        trade_signal: Interpretive signal from timeframe combination.
        agreement_score: 0-1 score of cross-timeframe agreement.
    """

    daily: TimeframeResult
    weekly: TimeframeResult
    monthly: TimeframeResult
    conflict_detected: bool
    highest_conviction: str
    trade_signal: str
    agreement_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily": self.daily.to_dict(),
            "weekly": self.weekly.to_dict(),
            "monthly": self.monthly.to_dict(),
            "conflict_detected": self.conflict_detected,
            "highest_conviction": self.highest_conviction,
            "trade_signal": self.trade_signal,
            "agreement_score": round(self.agreement_score, 4),
        }


class MultiTimeframeDetector:
    """Orchestrates regime detection across three timeframes.

    Creates three independent ``RegimeEnsemble`` instances with different
    lookback windows and feeds them data resampled to the appropriate
    frequency.

    Lookback windows:
        - Daily:   60 trading days  (~3 months)
        - Weekly:  52 weeks         (~1 year)
        - Monthly: 36 months        (~3 years)

    Args:
        config_path: Path to configuration directory.
        daily_window: Rolling window for daily ensemble.
        weekly_window: Rolling window for weekly ensemble (weeks).
        monthly_window: Rolling window for monthly ensemble (months).

    Example:
        >>> mtf = MultiTimeframeDetector(config_path="config")
        >>> mtf.fit(historical_data)
        >>> result = mtf.predict(recent_data)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        daily_window: int = 60,
        weekly_window: int = 52,
        monthly_window: int = 36,
    ) -> None:
        self.config_path = config_path or "config"
        self.daily_window = daily_window
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window

        # Three independent ensembles
        self._daily_ensemble = RegimeEnsemble(config_path=self.config_path)
        self._weekly_ensemble = RegimeEnsemble(config_path=self.config_path)
        self._monthly_ensemble = RegimeEnsemble(config_path=self.config_path)

        self._is_fitted = False

        # Duration tracking: how long each timeframe has stayed in current regime
        self._prev_regimes: Dict[str, Optional[int]] = {
            "daily": None, "weekly": None, "monthly": None,
        }
        self._regime_durations: Dict[str, int] = {
            "daily": 0, "weekly": 0, "monthly": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, data: pd.DataFrame) -> None:
        """Fit all three timeframe ensembles.

        Args:
            data: Historical market data (daily frequency expected).

        Raises:
            ValueError: If insufficient data.
        """
        if len(data) < 252:
            raise ValueError(
                f"Need at least 252 daily observations, got {len(data)}"
            )

        # Daily: use raw data
        self._daily_ensemble.fit(data)

        # Weekly: resample
        weekly = self._resample_to_weekly(data)
        if len(weekly) >= 52:
            self._weekly_ensemble.fit(weekly)
        else:
            logger.warning("Insufficient weekly data; copying daily ensemble")
            self._weekly_ensemble = self._daily_ensemble

        # Monthly: resample
        monthly = self._resample_to_monthly(data)
        if len(monthly) >= 24:
            self._monthly_ensemble.fit(monthly)
        else:
            logger.warning("Insufficient monthly data; copying daily ensemble")
            self._monthly_ensemble = self._daily_ensemble

        self._is_fitted = True
        logger.info("MultiTimeframeDetector fitted on %d daily observations", len(data))

    def predict(self, data: pd.DataFrame) -> MultiTimeframeResult:
        """Predict regime at all three timeframes.

        Args:
            data: Latest market data (daily frequency).

        Returns:
            MultiTimeframeResult with per-timeframe and aggregate views.

        Raises:
            ValueError: If model not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before predict()")

        # Daily prediction
        daily_data = data.iloc[-self.daily_window:] if len(data) > self.daily_window else data
        daily_res = self._daily_ensemble.predict(daily_data)
        daily_tf = self._make_timeframe_result("daily", daily_res)

        # Weekly prediction
        weekly_data = self._resample_to_weekly(data)
        weekly_window = min(self.weekly_window, len(weekly_data))
        weekly_slice = weekly_data.iloc[-weekly_window:] if weekly_window > 0 else weekly_data
        weekly_res = self._weekly_ensemble.predict(weekly_slice)
        weekly_tf = self._make_timeframe_result("weekly", weekly_res)

        # Monthly prediction
        monthly_data = self._resample_to_monthly(data)
        monthly_window = min(self.monthly_window, len(monthly_data))
        monthly_slice = monthly_data.iloc[-monthly_window:] if monthly_window > 0 else monthly_data
        monthly_res = self._monthly_ensemble.predict(monthly_slice)
        monthly_tf = self._make_timeframe_result("monthly", monthly_res)

        # Aggregate analysis
        regimes = [daily_tf.regime, weekly_tf.regime, monthly_tf.regime]
        conflict = len(set(regimes)) > 1

        # Agreement score: fraction of timeframes that agree
        from collections import Counter
        regime_counts = Counter(regimes)
        most_common_count = regime_counts.most_common(1)[0][1]
        agreement_score = most_common_count / 3.0

        # Highest conviction
        tf_results = {"daily": daily_tf, "weekly": weekly_tf, "monthly": monthly_tf}
        highest = max(tf_results, key=lambda k: tf_results[k].confidence)

        # Trade signal interpretation
        trade_signal = self._interpret_timeframe_stack(daily_tf, weekly_tf, monthly_tf)

        result = MultiTimeframeResult(
            daily=daily_tf,
            weekly=weekly_tf,
            monthly=monthly_tf,
            conflict_detected=conflict,
            highest_conviction=highest,
            trade_signal=trade_signal,
            agreement_score=agreement_score,
        )

        logger.info(
            "Multi-TF: daily=%s, weekly=%s, monthly=%s, conflict=%s, signal=%s",
            daily_tf.regime_name, weekly_tf.regime_name, monthly_tf.regime_name,
            conflict, trade_signal,
        )

        return result

    # ------------------------------------------------------------------
    # Data resampling
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly frequency.

        Uses last observation for prices, sum for volume.

        Args:
            data: Daily DataFrame with DatetimeIndex or date column.

        Returns:
            Weekly DataFrame.
        """
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to infer datetime index
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"], errors="ignore")
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")

        # Filter numeric columns only
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return df

        weekly = numeric.resample("W").last().dropna(how="all")
        return weekly

    @staticmethod
    def _resample_to_monthly(data: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to monthly frequency.

        Args:
            data: Daily DataFrame.

        Returns:
            Monthly DataFrame.
        """
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"], errors="ignore")
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")

        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            return df

        monthly = numeric.resample("ME").last().dropna(how="all")
        return monthly

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_timeframe_result(
        self,
        timeframe: str,
        ensemble_result: EnsembleResult,
    ) -> TimeframeResult:
        """Convert an EnsembleResult to a TimeframeResult with duration tracking.

        Args:
            timeframe: "daily", "weekly", or "monthly".
            ensemble_result: Result from the ensemble predictor.

        Returns:
            TimeframeResult.
        """
        regime = ensemble_result.regime
        prev = self._prev_regimes.get(timeframe)

        if prev == regime:
            self._regime_durations[timeframe] += 1
        else:
            self._regime_durations[timeframe] = 1
            self._prev_regimes[timeframe] = regime

        return TimeframeResult(
            timeframe=timeframe,
            regime=regime,
            regime_name=ensemble_result.regime_name,
            confidence=ensemble_result.confidence,
            disagreement=ensemble_result.disagreement,
            transition_warning=ensemble_result.transition_warning,
            duration=self._regime_durations[timeframe],
        )

    @staticmethod
    def _interpret_timeframe_stack(
        daily: TimeframeResult,
        weekly: TimeframeResult,
        monthly: TimeframeResult,
    ) -> str:
        """Interpret the combination of timeframe regimes into a trade signal.

        Key interpretations:
        - All agree risk-on → strong bullish
        - Daily bearish, monthly bullish → buy-the-dip
        - All bearish → defensive
        - Mixed → reduce exposure, await clarity

        Args:
            daily: Daily timeframe result.
            weekly: Weekly timeframe result.
            monthly: Monthly timeframe result.

        Returns:
            Human-readable trade signal string.
        """
        d, w, m = daily.regime, weekly.regime, monthly.regime

        # All agree
        if d == w == m:
            if d == 1:
                return "strong_bullish_all_timeframes_risk_on"
            elif d == 2:
                return "strong_defensive_all_timeframes_crisis"
            elif d == 3:
                return "caution_all_timeframes_stagflation"
            elif d == 4:
                return "bullish_all_timeframes_disinflationary_boom"

        # Daily diverges from longer timeframes
        if w == m and d != w:
            if d == 2 and w in (1, 4):
                return "buy_the_dip_daily_crisis_longer_bullish"
            elif d in (1, 4) and w == 2:
                return "sell_the_rally_daily_bullish_longer_crisis"
            elif d == 3 and w in (1, 4):
                return "caution_daily_stagflation_longer_growth"
            else:
                return f"daily_divergence_{REGIME_NAMES.get(d, 'unknown').lower().replace(' ', '_')}"

        # Weekly diverges
        if d == m and w != d:
            return "weekly_noise_daily_monthly_agree"

        # All different
        if len({d, w, m}) == 3:
            return "high_uncertainty_all_timeframes_disagree"

        return "mixed_signals_partial_agreement"
