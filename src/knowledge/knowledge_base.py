"""
Institutional Memory — Knowledge Base (Phase 5.1).

Indexes regime transitions, anomaly patterns, and macro-surprise impacts.
Every analysis run enriches the knowledge base, creating a compounding
advantage that stateless terminals cannot replicate.

Architecture:
    ┌────────────────┐   record()   ┌──────────────┐
    │ AMRCAIS.analyze│ ──────────→  │ KnowledgeBase│
    └────────────────┘              │ (SQLite)     │
                                    │              │
    ┌────────────────┐   query()    │  transitions │
    │ Pattern Match  │ ←──────────  │  anomalies   │
    │ & Forecasting  │              │  impacts     │
    └────────────────┘              └──────────────┘

Classes:
    RegimeTransitionRecord: Immutable record of a detected transition.
    AnomalyRecord: Cataloged anomaly with outcome tracking.
    PatternMatch: Historical analog with similarity score.
    KnowledgeBase: Core engine — indexing, querying, pattern-matching.

Example:
    >>> kb = KnowledgeBase()
    >>> kb.record_transition(from_regime=1, to_regime=2, confidence=0.78,
    ...     leading_indicators={"disagreement_trend": 0.65, "vix_slope": -0.3})
    >>> matches = kb.find_similar_transitions(current_indicators={"disagreement_trend": 0.6})
    >>> print(matches[0].similarity)
    0.87
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Data Records ─────────────────────────────────────────────────


@dataclass
class RegimeTransitionRecord:
    """Immutable record of a detected regime transition.

    Attributes:
        transition_id: Unique identifier.
        timestamp: When the transition was detected.
        from_regime: Regime ID before transition.
        to_regime: Regime ID after transition.
        confidence: Ensemble confidence at detection time.
        disagreement: Disagreement index at detection time.
        detection_latency_days: Days between actual and detected transition.
        leading_indicators: Dict of indicator name → value at transition.
        classifier_accuracy: Per-classifier correctness at this event.
        post_transition_performance: Performance metrics after the event.
        notes: Human or auto-generated annotations.
    """

    transition_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    from_regime: int = 0
    to_regime: int = 0
    confidence: float = 0.0
    disagreement: float = 0.0
    detection_latency_days: float = 0.0
    leading_indicators: Dict[str, float] = field(default_factory=dict)
    classifier_accuracy: Dict[str, bool] = field(default_factory=dict)
    post_transition_performance: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp,
            "from_regime": self.from_regime,
            "to_regime": self.to_regime,
            "confidence": self.confidence,
            "disagreement": self.disagreement,
            "detection_latency_days": self.detection_latency_days,
            "leading_indicators": self.leading_indicators,
            "classifier_accuracy": self.classifier_accuracy,
            "post_transition_performance": self.post_transition_performance,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeTransitionRecord":
        """Deserialize from dict."""
        return cls(
            transition_id=data.get("transition_id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            from_regime=data.get("from_regime", 0),
            to_regime=data.get("to_regime", 0),
            confidence=data.get("confidence", 0.0),
            disagreement=data.get("disagreement", 0.0),
            detection_latency_days=data.get("detection_latency_days", 0.0),
            leading_indicators=data.get("leading_indicators", {}),
            classifier_accuracy=data.get("classifier_accuracy", {}),
            post_transition_performance=data.get(
                "post_transition_performance", {}
            ),
            notes=data.get("notes", ""),
        )


@dataclass
class AnomalyRecord:
    """Cataloged cross-asset anomaly with outcome tracking.

    Attributes:
        anomaly_id: Unique identifier.
        timestamp: When the anomaly was detected.
        anomaly_type: Category (e.g., "correlation_spike", "vol_divergence").
        asset_pair: Affected assets (e.g., "SPX_TLT").
        regime: Regime at detection time.
        z_score: Deviation from regime baseline in standard deviations.
        expected_value: Baseline value for this regime.
        actual_value: Observed value.
        reversion_days: Days until the anomaly mean-reverted (None = ongoing).
        outcome: What happened after detection (populated retrospectively).
    """

    anomaly_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    anomaly_type: str = ""
    asset_pair: str = ""
    regime: int = 0
    z_score: float = 0.0
    expected_value: float = 0.0
    actual_value: float = 0.0
    reversion_days: Optional[float] = None
    outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "anomaly_id": self.anomaly_id,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "asset_pair": self.asset_pair,
            "regime": self.regime,
            "z_score": round(self.z_score, 4),
            "expected_value": round(self.expected_value, 4),
            "actual_value": round(self.actual_value, 4),
            "reversion_days": self.reversion_days,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyRecord":
        """Deserialize from dict."""
        return cls(
            anomaly_id=data.get("anomaly_id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            anomaly_type=data.get("anomaly_type", ""),
            asset_pair=data.get("asset_pair", ""),
            regime=data.get("regime", 0),
            z_score=data.get("z_score", 0.0),
            expected_value=data.get("expected_value", 0.0),
            actual_value=data.get("actual_value", 0.0),
            reversion_days=data.get("reversion_days"),
            outcome=data.get("outcome", ""),
        )


@dataclass
class PatternMatch:
    """Historical analog found by similarity search.

    Attributes:
        record: The matched transition record.
        similarity: Cosine similarity score (0–1).
        days_ago: How many days ago this pattern occurred.
        outcome_summary: What happened after the matched pattern.
    """

    record: RegimeTransitionRecord
    similarity: float = 0.0
    days_ago: float = 0.0
    outcome_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "transition": self.record.to_dict(),
            "similarity": round(self.similarity, 4),
            "days_ago": round(self.days_ago, 1),
            "outcome_summary": self.outcome_summary,
        }


# ─── Knowledge Base ───────────────────────────────────────────────


class KnowledgeBase:
    """Institutional memory engine for AMRCAIS.

    Stores and indexes regime transitions, anomaly patterns, and
    macro-surprise impacts.  Supports pattern-matching queries to
    find historical analogs for current market conditions.

    The knowledge base is the compounding advantage — every analysis
    run makes AMRCAIS smarter.  Bloomberg's terminal is stateless;
    AMRCAIS learns from every regime transition it observes.

    Args:
        storage_path: Path to the JSON-backed knowledge store.
        max_transitions: Maximum transition records to retain.
        max_anomalies: Maximum anomaly records to retain.

    Example:
        >>> kb = KnowledgeBase("data/knowledge.json")
        >>> kb.record_transition(from_regime=1, to_regime=2, confidence=0.78,
        ...     leading_indicators={"disagreement_trend": 0.65})
        >>> matches = kb.find_similar_transitions(
        ...     current_indicators={"disagreement_trend": 0.6})
    """

    def __init__(
        self,
        storage_path: str = "data/knowledge.json",
        max_transitions: int = 500,
        max_anomalies: int = 2000,
    ) -> None:
        self._storage_path = Path(storage_path)
        self._max_transitions = max_transitions
        self._max_anomalies = max_anomalies

        self._transitions: List[RegimeTransitionRecord] = []
        self._anomalies: List[AnomalyRecord] = []
        self._macro_impacts: Dict[str, Dict[int, List[float]]] = {}
        # macro_impacts[indicator][regime] = [impact1, impact2, ...]

        self._load()

    # ── Transition Management ─────────────────────────────────

    def record_transition(
        self,
        from_regime: int,
        to_regime: int,
        confidence: float = 0.0,
        disagreement: float = 0.0,
        detection_latency_days: float = 0.0,
        leading_indicators: Optional[Dict[str, float]] = None,
        classifier_accuracy: Optional[Dict[str, bool]] = None,
        notes: str = "",
    ) -> RegimeTransitionRecord:
        """Record a regime transition event.

        Args:
            from_regime: Previous regime ID.
            to_regime: New regime ID.
            confidence: Ensemble confidence at detection.
            disagreement: Disagreement index at detection.
            detection_latency_days: Detection delay vs actual.
            leading_indicators: Indicator snapshot at transition.
            classifier_accuracy: Which classifiers got it right.
            notes: Freeform annotation.

        Returns:
            The created RegimeTransitionRecord.
        """
        record = RegimeTransitionRecord(
            from_regime=from_regime,
            to_regime=to_regime,
            confidence=confidence,
            disagreement=disagreement,
            detection_latency_days=detection_latency_days,
            leading_indicators=leading_indicators or {},
            classifier_accuracy=classifier_accuracy or {},
            notes=notes,
        )

        self._transitions.append(record)

        # Trim if necessary
        if len(self._transitions) > self._max_transitions:
            self._transitions = self._transitions[-self._max_transitions :]

        self._save()

        logger.info(
            f"Recorded transition {from_regime}→{to_regime} "
            f"(confidence={confidence:.2f})"
        )
        return record

    def get_transitions(
        self,
        from_regime: Optional[int] = None,
        to_regime: Optional[int] = None,
        limit: int = 50,
    ) -> List[RegimeTransitionRecord]:
        """Query transition history.

        Args:
            from_regime: Filter by source regime.
            to_regime: Filter by destination regime.
            limit: Maximum records to return.

        Returns:
            List of transition records, most recent first.
        """
        result = self._transitions[:]

        if from_regime is not None:
            result = [r for r in result if r.from_regime == from_regime]
        if to_regime is not None:
            result = [r for r in result if r.to_regime == to_regime]

        return list(reversed(result[-limit:]))

    def find_similar_transitions(
        self,
        current_indicators: Dict[str, float],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> List[PatternMatch]:
        """Find historical transitions similar to current conditions.

        Uses cosine similarity over leading-indicator vectors.

        Args:
            current_indicators: Current indicator values.
            top_k: Maximum matches to return.
            min_similarity: Minimum similarity to include.

        Returns:
            List of PatternMatch objects, sorted by similarity desc.
        """
        if not current_indicators or not self._transitions:
            return []

        matches: List[PatternMatch] = []
        now = datetime.now()

        for record in self._transitions:
            if not record.leading_indicators:
                continue

            sim = self._cosine_similarity(
                current_indicators, record.leading_indicators
            )

            if sim < min_similarity:
                continue

            # Calculate days ago
            try:
                ts = datetime.fromisoformat(record.timestamp)
                days_ago = (now - ts).total_seconds() / 86400
            except (ValueError, TypeError):
                days_ago = 0.0

            # Build outcome summary
            perf = record.post_transition_performance
            if perf:
                parts = [f"{k}: {v:+.2f}%" for k, v in perf.items()]
                outcome = f"Post-transition: {', '.join(parts)}"
            else:
                outcome = f"Transition {record.from_regime}→{record.to_regime}"

            matches.append(
                PatternMatch(
                    record=record,
                    similarity=sim,
                    days_ago=days_ago,
                    outcome_summary=outcome,
                )
            )

        # Sort by similarity descending, take top_k
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    def update_transition_outcome(
        self,
        transition_id: str,
        post_transition_performance: Dict[str, float],
        notes: str = "",
    ) -> bool:
        """Update a transition record with post-event performance.

        Args:
            transition_id: ID of the transition to update.
            post_transition_performance: Asset → return after transition.
            notes: Additional annotations.

        Returns:
            True if updated, False if transition not found.
        """
        for record in self._transitions:
            if record.transition_id == transition_id:
                record.post_transition_performance = (
                    post_transition_performance
                )
                if notes:
                    record.notes = (
                        f"{record.notes}; {notes}" if record.notes else notes
                    )
                self._save()
                return True
        return False

    # ── Anomaly Catalog ───────────────────────────────────────

    def record_anomaly(
        self,
        anomaly_type: str,
        asset_pair: str,
        regime: int,
        z_score: float,
        expected_value: float = 0.0,
        actual_value: float = 0.0,
    ) -> AnomalyRecord:
        """Record a cross-asset anomaly detection.

        Args:
            anomaly_type: Category of anomaly.
            asset_pair: Affected asset pair.
            regime: Regime at detection time.
            z_score: Deviation in standard deviations.
            expected_value: Regime baseline.
            actual_value: Observed value.

        Returns:
            The created AnomalyRecord.
        """
        record = AnomalyRecord(
            anomaly_type=anomaly_type,
            asset_pair=asset_pair,
            regime=regime,
            z_score=z_score,
            expected_value=expected_value,
            actual_value=actual_value,
        )

        self._anomalies.append(record)

        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies :]

        self._save()

        logger.debug(
            f"Recorded anomaly: {anomaly_type} on {asset_pair} "
            f"(z={z_score:.2f})"
        )
        return record

    def get_anomalies(
        self,
        anomaly_type: Optional[str] = None,
        asset_pair: Optional[str] = None,
        regime: Optional[int] = None,
        limit: int = 50,
    ) -> List[AnomalyRecord]:
        """Query anomaly catalog.

        Args:
            anomaly_type: Filter by anomaly category.
            asset_pair: Filter by asset pair.
            regime: Filter by regime.
            limit: Maximum records to return.

        Returns:
            List of anomaly records, most recent first.
        """
        result = self._anomalies[:]

        if anomaly_type is not None:
            result = [r for r in result if r.anomaly_type == anomaly_type]
        if asset_pair is not None:
            result = [r for r in result if r.asset_pair == asset_pair]
        if regime is not None:
            result = [r for r in result if r.regime == regime]

        return list(reversed(result[-limit:]))

    def get_anomaly_stats(
        self,
        anomaly_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics on the anomaly catalog.

        Args:
            anomaly_type: Optional filter.

        Returns:
            Dict with count, avg_z_score, avg_reversion_days, etc.
        """
        records = self._anomalies
        if anomaly_type:
            records = [r for r in records if r.anomaly_type == anomaly_type]

        if not records:
            return {
                "total": 0,
                "avg_z_score": 0.0,
                "avg_reversion_days": None,
                "by_regime": {},
                "by_pair": {},
            }

        z_scores = [r.z_score for r in records]
        reversion = [
            r.reversion_days
            for r in records
            if r.reversion_days is not None
        ]

        by_regime: Dict[int, int] = {}
        by_pair: Dict[str, int] = {}
        for r in records:
            by_regime[r.regime] = by_regime.get(r.regime, 0) + 1
            by_pair[r.asset_pair] = by_pair.get(r.asset_pair, 0) + 1

        return {
            "total": len(records),
            "avg_z_score": round(sum(z_scores) / len(z_scores), 4),
            "avg_reversion_days": (
                round(sum(reversion) / len(reversion), 1) if reversion else None
            ),
            "by_regime": by_regime,
            "by_pair": by_pair,
        }

    def update_anomaly_outcome(
        self,
        anomaly_id: str,
        reversion_days: Optional[float] = None,
        outcome: str = "",
    ) -> bool:
        """Update an anomaly record with resolution data.

        Args:
            anomaly_id: ID of the anomaly to update.
            reversion_days: Days until mean reversion.
            outcome: Description of what happened.

        Returns:
            True if updated, False if not found.
        """
        for record in self._anomalies:
            if record.anomaly_id == anomaly_id:
                if reversion_days is not None:
                    record.reversion_days = reversion_days
                if outcome:
                    record.outcome = outcome
                self._save()
                return True
        return False

    # ── Macro Impact Database ─────────────────────────────────

    def record_macro_impact(
        self,
        indicator: str,
        regime: int,
        impact_pct: float,
    ) -> None:
        """Record the market impact of a macro data surprise.

        Args:
            indicator: Macro indicator name (e.g., "NFP", "CPI").
            regime: Regime at time of release.
            impact_pct: Asset return impact in percent.
        """
        if indicator not in self._macro_impacts:
            self._macro_impacts[indicator] = {}
        if regime not in self._macro_impacts[indicator]:
            self._macro_impacts[indicator][regime] = []

        self._macro_impacts[indicator][regime].append(impact_pct)
        self._save()

    def get_macro_impact_stats(
        self,
        indicator: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregate macro impact statistics.

        Args:
            indicator: Optional filter by indicator name.

        Returns:
            Dict of indicator → regime → {count, avg_impact, std_impact}.
        """
        sources = self._macro_impacts
        if indicator:
            sources = {indicator: sources.get(indicator, {})}

        stats: Dict[str, Any] = {}
        for ind, regimes in sources.items():
            stats[ind] = {}
            for regime, impacts in regimes.items():
                if impacts:
                    avg = sum(impacts) / len(impacts)
                    variance = (
                        sum((x - avg) ** 2 for x in impacts) / len(impacts)
                    )
                    stats[ind][regime] = {
                        "count": len(impacts),
                        "avg_impact": round(avg, 4),
                        "std_impact": round(math.sqrt(variance), 4),
                    }

        return stats

    # ── Aggregate Stats ───────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge base state.

        Returns:
            Dict with counts, coverage, and key statistics.
        """
        # Unique transition types
        transition_types = set()
        for t in self._transitions:
            transition_types.add((t.from_regime, t.to_regime))

        # Anomaly types
        anomaly_types = set(a.anomaly_type for a in self._anomalies if a.anomaly_type)

        return {
            "total_transitions": len(self._transitions),
            "unique_transition_types": len(transition_types),
            "transition_types": [
                {"from": f, "to": t} for f, t in sorted(transition_types)
            ],
            "total_anomalies": len(self._anomalies),
            "anomaly_types": sorted(anomaly_types),
            "macro_indicators_tracked": len(self._macro_impacts),
            "total_macro_observations": sum(
                len(impacts)
                for regimes in self._macro_impacts.values()
                for impacts in regimes.values()
            ),
        }

    # ── Persistence ───────────────────────────────────────────

    def _save(self) -> None:
        """Persist knowledge base to JSON file."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "transitions": [t.to_dict() for t in self._transitions],
                "anomalies": [a.to_dict() for a in self._anomalies],
                "macro_impacts": {
                    ind: {str(r): imps for r, imps in regimes.items()}
                    for ind, regimes in self._macro_impacts.items()
                },
                "last_updated": datetime.now().isoformat(),
            }
            self._storage_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error(f"Knowledge base save failed: {exc}")

    def _load(self) -> None:
        """Load knowledge base from JSON file."""
        if not self._storage_path.exists():
            logger.info(
                f"No existing knowledge base at {self._storage_path}; "
                "starting fresh"
            )
            return

        try:
            raw = json.loads(
                self._storage_path.read_text(encoding="utf-8")
            )

            self._transitions = [
                RegimeTransitionRecord.from_dict(t)
                for t in raw.get("transitions", [])
            ]
            self._anomalies = [
                AnomalyRecord.from_dict(a) for a in raw.get("anomalies", [])
            ]

            # Restore macro impacts with int regime keys
            self._macro_impacts = {}
            for ind, regimes in raw.get("macro_impacts", {}).items():
                self._macro_impacts[ind] = {
                    int(r): imps for r, imps in regimes.items()
                }

            logger.info(
                f"Loaded knowledge base: {len(self._transitions)} transitions, "
                f"{len(self._anomalies)} anomalies"
            )
        except Exception as exc:
            logger.error(f"Knowledge base load failed: {exc}")

    # ── Internal Helpers ──────────────────────────────────────

    @staticmethod
    def _cosine_similarity(
        a: Dict[str, float], b: Dict[str, float]
    ) -> float:
        """Compute cosine similarity between two indicator vectors.

        Args:
            a: First vector as dict.
            b: Second vector as dict.

        Returns:
            Similarity score between 0 and 1.
        """
        # Use shared keys only
        shared = set(a.keys()) & set(b.keys())
        if not shared:
            return 0.0

        dot = sum(a[k] * b[k] for k in shared)
        norm_a = math.sqrt(sum(a[k] ** 2 for k in shared))
        norm_b = math.sqrt(sum(b[k] ** 2 for k in shared))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm_a * norm_b)))
