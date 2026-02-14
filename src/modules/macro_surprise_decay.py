"""
Macro Surprise Decay Model for AMRCAIS.

Economic data surprises don't have instant, permanent effects — they decay
over time.  This module models per-indicator exponential decay with
regime-conditional half-lives and maintains a rolling cumulative surprise
index.

Key features:
    - Per-indicator half-lives (NFP decays faster than CPI in equities).
    - Regime-conditional decay: surprises in stagflation decay slower
      (persistent uncertainty) than in risk-on (quickly absorbed).
    - Cumulative surprise index: running weighted sum of all active
      (not yet stale) surprises.
    - Stale detection: flags surprises whose remaining impact is below a
      threshold and removes them from the active set.

Classes:
    SurpriseDecayModel: Per-indicator decay curve calculator.
    DecayingSurprise: Single surprise event with time-varying impact.

Example:
    >>> from src.modules.macro_surprise_decay import SurpriseDecayModel
    >>> model = SurpriseDecayModel()
    >>> model.update_regime(1, 0.85)
    >>> model.add_surprise("NFP", surprise=2.1, date=datetime(2025, 2, 7))
    >>> impact = model.get_current_impact("NFP")  # decayed value
    >>> index = model.get_cumulative_index()       # aggregate
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


@dataclass
class DecayingSurprise:
    """A single macro surprise with time-varying impact.

    Attributes:
        indicator: Indicator name (e.g., "NFP", "CPI").
        surprise: Standardized surprise magnitude (actual - consensus) / std.
        release_date: Date the data was released.
        half_life_days: Decay half-life in trading days.
        initial_weight: Regime-specific importance weight.
        regime_at_release: Regime when the surprise occurred.
    """

    indicator: str
    surprise: float
    release_date: datetime
    half_life_days: float
    initial_weight: float = 1.0
    regime_at_release: int = 1

    def impact_at(self, current_date: datetime) -> float:
        """Compute the decayed impact at a given date.

        Uses exponential decay: impact = surprise * weight * 2^(-t/half_life).

        Args:
            current_date: Date at which to evaluate impact.

        Returns:
            Decayed impact value.
        """
        elapsed = (current_date - self.release_date).days
        if elapsed < 0:
            return 0.0
        if self.half_life_days <= 0:
            return 0.0
        decay_factor = math.pow(2.0, -elapsed / self.half_life_days)
        return self.surprise * self.initial_weight * decay_factor

    def is_stale(self, current_date: datetime, threshold: float = 0.05) -> bool:
        """Check if this surprise has decayed below the stale threshold.

        Args:
            current_date: Current evaluation date.
            threshold: Absolute impact below which surprise is considered stale.

        Returns:
            True if the surprise is too old to be informative.
        """
        return abs(self.impact_at(current_date)) < threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "indicator": self.indicator,
            "surprise": self.surprise,
            "release_date": self.release_date.isoformat(),
            "half_life_days": self.half_life_days,
            "initial_weight": self.initial_weight,
            "regime_at_release": self.regime_at_release,
        }


class SurpriseDecayModel(AnalyticalModule):
    """Models the time-varying impact of macro data surprises.

    Maintains a pool of active surprises and computes a cumulative
    surprise index that reflects the *current* aggregate macro sentiment
    after accounting for decay.

    Regime-adaptive behavior:
        - Risk-On Growth (1): Fast decay — market absorbs surprises quickly.
        - Risk-Off Crisis (2): Medium decay — heightened sensitivity.
        - Stagflation (3): Slow decay — persistent uncertainty.
        - Disinflationary Boom (4): Normal decay.

    Args:
        config_path: Path to YAML configuration directory.
        stale_threshold: Absolute impact below which surprise is pruned.

    Example:
        >>> model = SurpriseDecayModel()
        >>> model.update_regime(3, 0.8)  # Stagflation → slow decay
        >>> model.add_surprise("CPI", 1.8, datetime(2025, 2, 12))
        >>> idx = model.get_cumulative_index()
    """

    # Default half-lives (trading days) per indicator per asset-class sensitivity
    # Format: {indicator: {regime: half_life_days}}
    DEFAULT_HALF_LIVES: Dict[str, Dict[int, float]] = {
        "NFP": {1: 3.0, 2: 5.0, 3: 7.0, 4: 4.0},
        "CPI": {1: 5.0, 2: 7.0, 3: 14.0, 4: 6.0},
        "CORE_CPI": {1: 5.0, 2: 7.0, 3: 14.0, 4: 6.0},
        "PPI": {1: 3.0, 2: 4.0, 3: 10.0, 4: 4.0},
        "PMI": {1: 4.0, 2: 5.0, 3: 8.0, 4: 5.0},
        "GDP": {1: 7.0, 2: 10.0, 3: 14.0, 4: 7.0},
        "FOMC": {1: 5.0, 2: 10.0, 3: 12.0, 4: 7.0},
        "RETAIL_SALES": {1: 2.0, 2: 3.0, 3: 5.0, 4: 3.0},
    }

    # Importance weights per regime (same as MacroEventTracker)
    REGIME_WEIGHTS: Dict[int, Dict[str, float]] = {
        1: {"NFP": 1.2, "CPI": 0.8, "FOMC": 1.0, "PMI": 1.1, "GDP": 0.9,
            "PPI": 0.7, "CORE_CPI": 0.8, "RETAIL_SALES": 0.9},
        2: {"NFP": 0.3, "CPI": 0.5, "FOMC": 1.5, "PMI": 0.4, "GDP": 0.3,
            "PPI": 0.4, "CORE_CPI": 0.5, "RETAIL_SALES": 0.3},
        3: {"NFP": 0.7, "CPI": 1.5, "FOMC": 1.2, "PMI": 0.8, "GDP": 0.6,
            "PPI": 1.3, "CORE_CPI": 1.5, "RETAIL_SALES": 0.5},
        4: {"NFP": 1.0, "CPI": 0.6, "FOMC": 1.3, "PMI": 0.9, "GDP": 1.0,
            "PPI": 0.5, "CORE_CPI": 0.6, "RETAIL_SALES": 0.8},
    }

    # Historical standard deviations (reused from MacroEventTracker)
    HISTORICAL_STDS: Dict[str, float] = {
        "NFP": 75.0,
        "CPI": 0.1,
        "CORE_CPI": 0.1,
        "PPI": 0.2,
        "PMI": 1.5,
        "GDP": 0.5,
        "FOMC": 0.25,
        "RETAIL_SALES": 0.3,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        stale_threshold: float = 0.05,
    ) -> None:
        super().__init__(name="SurpriseDecayModel", config_path=config_path)
        self.stale_threshold = stale_threshold
        self._active_surprises: List[DecayingSurprise] = []
        self._surprise_history: List[DecayingSurprise] = []
        self._half_lives = dict(self.DEFAULT_HALF_LIVES)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_surprise(
        self,
        indicator: str,
        surprise: Optional[float] = None,
        release_date: Optional[datetime] = None,
        actual: Optional[float] = None,
        consensus: Optional[float] = None,
    ) -> DecayingSurprise:
        """Register a new macro data surprise.

        Args:
            indicator: Indicator name (e.g. "NFP", "CPI").
            surprise: Pre-computed standardized surprise. If None, computed
                      from actual/consensus.
            release_date: When the data was released. Defaults to now.
            actual: Actual value (used if surprise is None).
            consensus: Consensus value (used if surprise is None).

        Returns:
            The created DecayingSurprise instance.

        Raises:
            ValueError: If neither surprise nor actual+consensus provided.
        """
        if surprise is None:
            if actual is None or consensus is None:
                raise ValueError(
                    "Provide either `surprise` or both `actual` and `consensus`"
                )
            hist_std = self.HISTORICAL_STDS.get(indicator, 1.0)
            surprise = (actual - consensus) / hist_std

        if release_date is None:
            release_date = datetime.now()

        regime = self.current_regime or 1
        half_life = self._get_half_life(indicator, regime)
        weight = self.REGIME_WEIGHTS.get(regime, {}).get(indicator, 1.0)

        ds = DecayingSurprise(
            indicator=indicator,
            surprise=surprise,
            release_date=release_date,
            half_life_days=half_life,
            initial_weight=weight,
            regime_at_release=regime,
        )

        self._active_surprises.append(ds)
        self._surprise_history.append(ds)

        logger.info(
            "Surprise added: %s = %.2f (half-life=%.1fd, regime=%d)",
            indicator, surprise, half_life, regime,
        )

        return ds

    def get_current_impact(
        self,
        indicator: Optional[str] = None,
        current_date: Optional[datetime] = None,
    ) -> float:
        """Get the current decayed impact for an indicator or all indicators.

        Args:
            indicator: Specific indicator to query. If None returns total.
            current_date: Evaluation date. Defaults to now.

        Returns:
            Aggregate decayed impact.
        """
        if current_date is None:
            current_date = datetime.now()

        self._prune_stale(current_date)

        total = 0.0
        for ds in self._active_surprises:
            if indicator is not None and ds.indicator != indicator:
                continue
            total += ds.impact_at(current_date)

        return total

    def get_cumulative_index(
        self,
        current_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Compute the cumulative surprise index.

        The index is the sum of all active (non-stale) decayed surprises,
        providing a single number summarizing current macro sentiment.

        Args:
            current_date: Evaluation date. Defaults to now.

        Returns:
            Dict with index value, components, and stale count.
        """
        if current_date is None:
            current_date = datetime.now()

        self._prune_stale(current_date)

        components: Dict[str, float] = {}
        for ds in self._active_surprises:
            impact = ds.impact_at(current_date)
            components[ds.indicator] = components.get(ds.indicator, 0.0) + impact

        index_value = sum(components.values())

        # Determine direction
        if index_value > 0.5:
            direction = "positive_surprise_dominant"
        elif index_value < -0.5:
            direction = "negative_surprise_dominant"
        else:
            direction = "neutral"

        return {
            "index": round(index_value, 4),
            "direction": direction,
            "components": {k: round(v, 4) for k, v in components.items()},
            "active_surprises": len(self._active_surprises),
            "total_historical": len(self._surprise_history),
        }

    def get_decay_curves(
        self,
        current_date: Optional[datetime] = None,
        forward_days: int = 30,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get decay curves for all active surprises.

        Args:
            current_date: Start date. Defaults to now.
            forward_days: Number of days to project forward.

        Returns:
            Dict keyed by indicator with list of {day, impact} points.
        """
        if current_date is None:
            current_date = datetime.now()

        curves: Dict[str, List[Dict[str, Any]]] = {}

        for ds in self._active_surprises:
            if ds.indicator not in curves:
                curves[ds.indicator] = []

            for day_offset in range(forward_days + 1):
                eval_date = current_date + timedelta(days=day_offset)
                impact = ds.impact_at(eval_date)
                curves[ds.indicator].append({
                    "day": day_offset,
                    "impact": round(impact, 6),
                    "is_stale": abs(impact) < self.stale_threshold,
                })

        return curves

    def get_half_lives(self) -> Dict[str, Dict[int, float]]:
        """Return the current half-life table.

        Returns:
            Dict mapping indicator → {regime: half_life_days}.
        """
        return dict(self._half_lives)

    # ------------------------------------------------------------------
    # AnalyticalModule interface
    # ------------------------------------------------------------------

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current surprise environment.

        Args:
            data: DataFrame (not directly used; surprises added via add_surprise).

        Returns:
            Analysis result with signal and cumulative index.
        """
        now = datetime.now()
        index_data = self.get_cumulative_index(now)
        index_val = index_data["index"]

        # Determine signal based on index
        regime = self.current_regime or 1
        if regime in (1, 4):
            # Growth regimes: positive surprises → bullish
            if index_val > 0.8:
                signal, strength = "bullish", min(1.0, abs(index_val) / 3.0)
            elif index_val < -0.8:
                signal, strength = "bearish", min(1.0, abs(index_val) / 3.0)
            else:
                signal, strength = "neutral", 0.3
        elif regime == 2:
            # Crisis: any surprise → cautious
            if abs(index_val) > 0.5:
                signal, strength = "cautious", 0.7
            else:
                signal, strength = "neutral", 0.2
        else:
            # Stagflation: positive CPI surprise → bearish
            cpi_impact = index_data["components"].get("CPI", 0) + index_data["components"].get("CORE_CPI", 0)
            if cpi_impact > 0.3:
                signal, strength = "bearish", min(1.0, cpi_impact / 2.0)
            elif index_val < -0.5:
                signal, strength = "bullish", min(1.0, abs(index_val) / 3.0)
            else:
                signal, strength = "neutral", 0.3

        explanation = (
            f"Cumulative surprise index: {index_val:.2f} ({index_data['direction']}). "
            f"{index_data['active_surprises']} active surprises. "
            f"Components: {index_data['components']}"
        )

        module_signal = self.create_signal(
            signal=signal,
            strength=strength,
            explanation=explanation,
            regime_context=f"Regime {regime} decay model",
        )

        return {
            "signal": module_signal,
            "cumulative_index": index_data,
            "active_surprises": [ds.to_dict() for ds in self._active_surprises],
            "decay_curves": self.get_decay_curves(now, forward_days=14),
        }

    def get_regime_parameters(self, regime: int) -> Dict[str, Any]:
        """Get decay parameters for a specific regime.

        Args:
            regime: Regime ID (1-4).

        Returns:
            Half-lives and weights for the regime.
        """
        half_lives = {
            ind: hl.get(regime, 5.0)
            for ind, hl in self._half_lives.items()
        }
        weights = self.REGIME_WEIGHTS.get(regime, {})
        return {
            "half_lives": half_lives,
            "weights": weights,
            "regime": regime,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_half_life(self, indicator: str, regime: int) -> float:
        """Look up the half-life for an indicator in a regime.

        Args:
            indicator: Macro indicator name.
            regime: Current regime id.

        Returns:
            Half-life in trading days.
        """
        indicator_hl = self._half_lives.get(indicator, {})
        if isinstance(indicator_hl, dict):
            return indicator_hl.get(regime, 5.0)
        return float(indicator_hl) if indicator_hl else 5.0

    def _prune_stale(self, current_date: datetime) -> int:
        """Remove surprises that have decayed below the stale threshold.

        Args:
            current_date: Current evaluation date.

        Returns:
            Number of surprises pruned.
        """
        before = len(self._active_surprises)
        self._active_surprises = [
            ds for ds in self._active_surprises
            if not ds.is_stale(current_date, self.stale_threshold)
        ]
        pruned = before - len(self._active_surprises)
        if pruned > 0:
            logger.debug("Pruned %d stale surprises", pruned)
        return pruned
