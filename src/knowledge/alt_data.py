"""
Alternative Data Integration — Phase 5.3.

Extends the AMRCAIS data pipeline with non-traditional signals that
uniquely inform regime detection beyond market-price data.

Signals integrated:
    1. Fed Funds Futures  → Market-implied rate path → regime expectations
    2. MOVE Index         → Bond volatility → fixed income regime
    3. SKEW Index         → Tail risk pricing → crash probability
    4. CDX Indices        → Credit risk → corporate regime
    5. Copper/Gold Ratio  → Growth vs safety → macro regime proxy
    6. High-Yield Spreads → Risk appetite → credit regime
    7. TIPS Breakevens    → Inflation expectations → stagflation detector

Each signal feeds into the regime detection ensemble with its own
classifier weight.  More data → more classifiers → higher confidence →
better disagreement signals.

Classes:
    AltDataSignal: Single alternative data reading with regime signal.
    AltDataIntegrator: Fetches, normalizes, and interprets alt-data.

Example:
    >>> integrator = AltDataIntegrator()
    >>> signals = integrator.get_all_signals(regime=1)
    >>> for s in signals:
    ...     print(f"{s.name}: {s.regime_signal} (weight={s.weight})")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Signal Definitions ──────────────────────────────────────────


class AltSignalType(str, Enum):
    """Alternative data signal types."""

    FED_FUNDS_FUTURES = "fed_funds_futures"
    MOVE_INDEX = "move_index"
    SKEW_INDEX = "skew_index"
    CDX_INDEX = "cdx_index"
    COPPER_GOLD_RATIO = "copper_gold_ratio"
    HY_SPREADS = "hy_spreads"
    TIPS_BREAKEVEN = "tips_breakeven"


@dataclass
class AltDataSignal:
    """Single alternative data reading with regime interpretation.

    Attributes:
        name: Signal name (from AltSignalType).
        value: Raw observed value.
        z_score: Deviation from historical mean in σ.
        regime_signal: Regime interpretation string.
        regime_weight: Weight for regime ensemble voting.
        timestamp: When data was observed.
        source: Data provider.
        confidence: Confidence in the reading (0–1).
    """

    name: str = ""
    value: float = 0.0
    z_score: float = 0.0
    regime_signal: str = "neutral"
    regime_weight: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "z_score": round(self.z_score, 4),
            "regime_signal": self.regime_signal,
            "regime_weight": round(self.regime_weight, 4),
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": round(self.confidence, 4),
        }


# ─── Regime-Conditional Interpretation Rules ─────────────────────


# Regime IDs: 1 = Risk-On Growth, 2 = Risk-Off Crisis,
#              3 = Stagflation,    4 = Disinflationary Boom

# Thresholds: (signal_name, regime) -> interpretation function
# Each function: (value, z_score) -> (regime_signal, weight, confidence)

_REGIME_INTERPRETATION = {
    # ── Fed Funds Futures ─────────────────────────────────────
    AltSignalType.FED_FUNDS_FUTURES: {
        1: lambda v, z: (
            ("hawkish" if z > 1.0 else "neutral" if z > -0.5 else "dovish"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.15),
        ),
        2: lambda v, z: (
            ("easing" if z < -1.0 else "hold" if z < 0.5 else "tightening"),
            0.15,
            min(1.0, 0.5 + abs(z) * 0.15),
        ),
        3: lambda v, z: (
            ("behind_curve" if z > 0.5 else "responsive"),
            0.18,
            min(1.0, 0.5 + abs(z) * 0.15),
        ),
        4: lambda v, z: (
            ("accommodative" if z < -0.5 else "neutral"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.15),
        ),
    },
    # ── MOVE Index ────────────────────────────────────────────
    AltSignalType.MOVE_INDEX: {
        1: lambda v, z: (
            ("bond_stress" if z > 1.5 else "calm" if z < 0.5 else "elevated"),
            0.08,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        2: lambda v, z: (
            ("crisis_vol" if z > 2.0 else "elevated" if z > 0.5 else "subsiding"),
            0.14,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        3: lambda v, z: (
            ("rate_uncertainty" if z > 1.0 else "repricing"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        4: lambda v, z: (
            ("calm" if z < 0.5 else "rising"),
            0.06,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
    },
    # ── SKEW Index ────────────────────────────────────────────
    AltSignalType.SKEW_INDEX: {
        1: lambda v, z: (
            ("tail_risk_high" if z > 1.5 else "normal"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        2: lambda v, z: (
            ("crash_fear" if z > 2.0 else "elevated" if z > 1.0 else "normalizing"),
            0.15,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        3: lambda v, z: (
            ("stagflation_tail" if z > 1.0 else "moderate"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        4: lambda v, z: (
            ("complacency" if z < -1.0 else "benign"),
            0.08,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
    },
    # ── CDX Index ─────────────────────────────────────────────
    AltSignalType.CDX_INDEX: {
        1: lambda v, z: (
            ("credit_tight" if z < -0.5 else "normal"),
            0.08,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        2: lambda v, z: (
            ("credit_stress" if z > 1.5 else "widening" if z > 0.5 else "stable"),
            0.16,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        3: lambda v, z: (
            ("credit_risk" if z > 1.0 else "moderate"),
            0.14,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        4: lambda v, z: (
            ("credit_benign" if z < -0.5 else "normal"),
            0.06,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
    },
    # ── Copper/Gold Ratio ─────────────────────────────────────
    AltSignalType.COPPER_GOLD_RATIO: {
        1: lambda v, z: (
            ("growth_signal" if z > 0.5 else "neutral" if z > -0.5 else "caution"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        2: lambda v, z: (
            ("flight_safety" if z < -1.0 else "risk_off" if z < 0 else "recovering"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        3: lambda v, z: (
            ("industrial_weak" if z < -0.5 else "commodity_bid"),
            0.14,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        4: lambda v, z: (
            ("disinflation_signal" if z > 0.5 else "mixed"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
    },
    # ── High-Yield Spreads ────────────────────────────────────
    AltSignalType.HY_SPREADS: {
        1: lambda v, z: (
            ("tight" if z < -0.5 else "normal" if z < 1.0 else "widening"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        2: lambda v, z: (
            ("blowout" if z > 2.0 else "stress" if z > 1.0 else "elevated"),
            0.18,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        3: lambda v, z: (
            ("stagflation_stress" if z > 1.0 else "moderate"),
            0.14,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
        4: lambda v, z: (
            ("goldilocks" if z < -1.0 else "benign"),
            0.08,
            min(1.0, 0.5 + abs(z) * 0.12),
        ),
    },
    # ── TIPS Breakevens ───────────────────────────────────────
    AltSignalType.TIPS_BREAKEVEN: {
        1: lambda v, z: (
            ("inflation_risk" if z > 1.5 else "anchored" if z < 0.5 else "rising"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        2: lambda v, z: (
            ("deflation_risk" if z < -1.5 else "falling" if z < 0 else "stable"),
            0.10,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        3: lambda v, z: (
            ("inflation_embedded" if z > 1.0 else "rising" if z > 0 else "mixed"),
            0.20,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
        4: lambda v, z: (
            ("disinflation" if z < -0.5 else "low"),
            0.12,
            min(1.0, 0.5 + abs(z) * 0.10),
        ),
    },
}

# Historical reference values for z-score normalisation
# Format: {signal: (mean, std, source_hint)}
_HISTORICAL_BASELINES: Dict[AltSignalType, tuple] = {
    AltSignalType.FED_FUNDS_FUTURES: (2.5, 1.5, "FRED: DFF"),
    AltSignalType.MOVE_INDEX: (90.0, 30.0, "CBOE MOVE"),
    AltSignalType.SKEW_INDEX: (125.0, 10.0, "CBOE SKEW"),
    AltSignalType.CDX_INDEX: (75.0, 40.0, "CDX.NA.IG"),
    AltSignalType.COPPER_GOLD_RATIO: (0.0055, 0.0015, "Copper/Gold ratio"),
    AltSignalType.HY_SPREADS: (400.0, 150.0, "ICE BofA HY OAS"),
    AltSignalType.TIPS_BREAKEVEN: (2.2, 0.5, "FRED: T10YIE"),
}


# ─── AltData Integrator ─────────────────────────────────────────


class AltDataIntegrator:
    """Alternative data signal integration for AMRCAIS.

    Fetches, normalises, and interprets alternative data signals
    through a regime-first lens.  Each signal is compared to its
    historical baseline and interpreted according to the current
    regime — the same reading means different things in risk-on vs
    stagflation environments.

    Args:
        baselines: Custom historical baselines (override defaults).
        data_fetcher: Optional callback to fetch live data.

    Example:
        >>> adi = AltDataIntegrator()
        >>> adi.set_signal_value("move_index", 130.0)
        >>> signals = adi.get_all_signals(regime=2)
        >>> print(signals[0].regime_signal)
        'elevated'
    """

    def __init__(
        self,
        baselines: Optional[Dict[str, tuple]] = None,
        data_fetcher: Optional[Callable] = None,
    ) -> None:
        self._baselines = dict(_HISTORICAL_BASELINES)
        if baselines:
            for k, v in baselines.items():
                try:
                    self._baselines[AltSignalType(k)] = v
                except ValueError:
                    logger.warning(f"Unknown alt signal type: {k}")

        self._data_fetcher = data_fetcher

        # Current raw values (set manually or by fetcher)
        self._current_values: Dict[AltSignalType, float] = {}
        self._last_update: Optional[str] = None

    # ── Setting Values ────────────────────────────────────────

    def set_signal_value(self, signal_name: str, value: float) -> None:
        """Set the current value for an alternative data signal.

        Args:
            signal_name: Signal identifier (from AltSignalType values).
            value: Current observed value.
        """
        try:
            sig_type = AltSignalType(signal_name)
        except ValueError:
            logger.warning(f"Unknown signal: {signal_name}")
            return

        self._current_values[sig_type] = value
        self._last_update = datetime.now().isoformat()

    def set_all_values(self, values: Dict[str, float]) -> None:
        """Batch-set multiple signal values.

        Args:
            values: Dict of signal_name → current value.
        """
        for name, val in values.items():
            self.set_signal_value(name, val)

    # ── Signal Retrieval ──────────────────────────────────────

    def get_signal(
        self, signal_name: str, regime: int
    ) -> Optional[AltDataSignal]:
        """Get a single interpreted signal for the current regime.

        Args:
            signal_name: Signal identifier.
            regime: Current regime ID (1–4).

        Returns:
            AltDataSignal with regime interpretation, or None if no data.
        """
        try:
            sig_type = AltSignalType(signal_name)
        except ValueError:
            return None

        if sig_type not in self._current_values:
            return None

        value = self._current_values[sig_type]
        baseline = self._baselines.get(sig_type, (0.0, 1.0, ""))
        mean, std, source = baseline[0], baseline[1], baseline[2]

        z_score = (value - mean) / std if std > 0 else 0.0

        # Get regime-conditional interpretation
        interp_map = _REGIME_INTERPRETATION.get(sig_type, {})
        interp_fn = interp_map.get(regime)

        if interp_fn:
            regime_signal, weight, confidence = interp_fn(value, z_score)
        else:
            regime_signal, weight, confidence = "neutral", 0.05, 0.3

        return AltDataSignal(
            name=sig_type.value,
            value=value,
            z_score=z_score,
            regime_signal=regime_signal,
            regime_weight=weight,
            source=source,
            confidence=confidence,
            timestamp=self._last_update or datetime.now().isoformat(),
        )

    def get_all_signals(self, regime: int) -> List[AltDataSignal]:
        """Get all available signals interpreted for the current regime.

        Args:
            regime: Current regime ID (1–4).

        Returns:
            List of AltDataSignal with regime-conditional interpretation.
        """
        signals: List[AltDataSignal] = []
        for sig_type in AltSignalType:
            sig = self.get_signal(sig_type.value, regime)
            if sig is not None:
                signals.append(sig)
        return signals

    def get_regime_vote(self, regime: int) -> Dict[str, Any]:
        """Produce a weighted vote for the regime ensemble.

        Aggregates all available signals into a single regime vote
        that can be combined with the core classifier ensemble.

        Args:
            regime: Current suspected regime ID.

        Returns:
            Dict with vote, total_weight, signal_count, details.
        """
        signals = self.get_all_signals(regime)
        if not signals:
            return {
                "vote": "abstain",
                "total_weight": 0.0,
                "signal_count": 0,
                "confidence": 0.0,
                "details": [],
            }

        # Compute weighted confidence
        total_weight = sum(s.regime_weight for s in signals)
        weighted_confidence = (
            sum(s.confidence * s.regime_weight for s in signals) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Determine overall vote direction
        confirming = sum(
            1 for s in signals if s.confidence > 0.6
        )
        contradicting = sum(
            1 for s in signals if s.confidence < 0.4
        )

        if confirming > contradicting:
            vote = "confirm"
        elif contradicting > confirming:
            vote = "contradict"
        else:
            vote = "neutral"

        return {
            "vote": vote,
            "total_weight": round(total_weight, 4),
            "signal_count": len(signals),
            "confidence": round(weighted_confidence, 4),
            "confirming_signals": confirming,
            "contradicting_signals": contradicting,
            "details": [s.to_dict() for s in signals],
        }

    # ── Summary / Status ──────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get integrator status with available signals.

        Returns:
            Dict with signal availability and last update time.
        """
        available = [s.value for s in AltSignalType if s in self._current_values]
        missing = [s.value for s in AltSignalType if s not in self._current_values]

        return {
            "available_signals": available,
            "missing_signals": missing,
            "total_configured": len(AltSignalType),
            "total_available": len(available),
            "last_update": self._last_update,
            "baselines": {
                s.value: {"mean": b[0], "std": b[1], "source": b[2]}
                for s, b in self._baselines.items()
            },
        }

    def get_signal_types(self) -> List[str]:
        """Get all supported signal type names.

        Returns:
            List of signal type identifiers.
        """
        return [s.value for s in AltSignalType]
