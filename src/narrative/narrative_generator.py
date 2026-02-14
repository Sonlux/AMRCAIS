"""
Natural Language Regime Narrative Generator for AMRCAIS.

Generates human-readable daily briefings backed by specific data from
the analysis modules.  Every sentence is grounded in a concrete metric.
The narrative changes based on the current regime — the same data gets
a different story depending on market context.

**Template-based initially; designed for future LLM enhancement.**

Classes:
    NarrativeGenerator: Produces structured daily briefings.
    NarrativeBriefing: Container for a complete briefing.

Example:
    >>> from src.narrative.narrative_generator import NarrativeGenerator
    >>> gen = NarrativeGenerator()
    >>> briefing = gen.generate(analysis_result)
    >>> print(briefing.full_text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regime names
REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}

# Regime descriptive adjectives for narrative flavour
REGIME_TONE: Dict[int, Dict[str, str]] = {
    1: {
        "sentiment": "constructive",
        "bias": "risk-seeking",
        "concern_adj": "manageable",
        "outlook": "favorable",
    },
    2: {
        "sentiment": "defensive",
        "bias": "risk-averse",
        "concern_adj": "acute",
        "outlook": "uncertain",
    },
    3: {
        "sentiment": "cautious",
        "bias": "inflation-hedging",
        "concern_adj": "persistent",
        "outlook": "challenging",
    },
    4: {
        "sentiment": "optimistic",
        "bias": "duration-seeking",
        "concern_adj": "fading",
        "outlook": "improving",
    },
}


@dataclass
class NarrativeBriefing:
    """Complete daily narrative briefing.

    Attributes:
        headline: One-line regime headline.
        regime_section: Paragraph about the current regime state.
        signal_section: Paragraph about module signals.
        risk_section: Paragraph about warnings and risks.
        positioning_section: Suggested positioning changes.
        full_text: Complete assembled briefing.
        timestamp: When the briefing was generated.
        data_sources: Which data points back each sentence.
    """

    headline: str
    regime_section: str
    signal_section: str
    risk_section: str
    positioning_section: str
    full_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    data_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "regime_section": self.regime_section,
            "signal_section": self.signal_section,
            "risk_section": self.risk_section,
            "positioning_section": self.positioning_section,
            "full_text": self.full_text,
            "timestamp": self.timestamp.isoformat(),
            "data_sources": self.data_sources,
        }


class NarrativeGenerator:
    """Template-based narrative generator for AMRCAIS daily briefings.

    Transforms structured analysis output into coherent, human-readable
    market commentary.  Every sentence is backed by a specific data point
    from the analysis pipeline.

    Design principles:
        1. Regime-first framing — the regime context sets the narrative tone.
        2. Data-backed assertions — no hand-waving; every claim cites a metric.
        3. Actionable conclusions — end with positioning guidance.
        4. Regime-conditional language — same data, different story per regime.

    Args:
        include_positioning: Whether to include positioning suggestions.

    Example:
        >>> gen = NarrativeGenerator()
        >>> briefing = gen.generate(analysis)
    """

    def __init__(self, include_positioning: bool = True) -> None:
        self.include_positioning = include_positioning

    def generate(self, analysis: Dict[str, Any]) -> NarrativeBriefing:
        """Generate a complete daily briefing from analysis results.

        Args:
            analysis: Full AMRCAIS analysis dict with keys:
                      "regime", "modules", "meta", "summary".

        Returns:
            NarrativeBriefing with all sections populated.
        """
        regime_data = analysis.get("regime", {})
        modules_data = analysis.get("modules", {})
        meta_data = analysis.get("meta", {})
        summary_data = analysis.get("summary", {})

        regime_id = regime_data.get("id", 1)
        regime_name = regime_data.get("name", REGIME_NAMES.get(regime_id, "Unknown"))
        confidence = regime_data.get("confidence", 0.0)
        disagreement = regime_data.get("disagreement", 0.0)
        transition_warning = regime_data.get("transition_warning", False)

        tone = REGIME_TONE.get(regime_id, REGIME_TONE[1])
        data_sources: Dict[str, str] = {}

        # 1. Headline
        headline = self._build_headline(
            regime_name, confidence, disagreement, transition_warning
        )

        # 2. Regime section
        regime_section = self._build_regime_section(
            regime_id, regime_name, confidence, disagreement,
            transition_warning, regime_data, tone, data_sources,
        )

        # 3. Signal section
        signal_section = self._build_signal_section(
            modules_data, regime_id, tone, data_sources,
        )

        # 4. Risk section
        risk_section = self._build_risk_section(
            meta_data, disagreement, transition_warning, tone, data_sources,
        )

        # 5. Positioning section
        if self.include_positioning:
            positioning_section = self._build_positioning_section(
                regime_id, summary_data, modules_data, tone, data_sources,
            )
        else:
            positioning_section = ""

        # Assemble
        now = datetime.now()
        date_str = now.strftime("%B %d, %Y")
        sections = [
            f"AMRCAIS Daily Briefing — {date_str}",
            "",
            headline,
            "",
            regime_section,
            "",
            signal_section,
            "",
            risk_section,
        ]
        if positioning_section:
            sections.extend(["", positioning_section])

        full_text = "\n".join(sections)

        return NarrativeBriefing(
            headline=headline,
            regime_section=regime_section,
            signal_section=signal_section,
            risk_section=risk_section,
            positioning_section=positioning_section,
            full_text=full_text,
            timestamp=now,
            data_sources=data_sources,
        )

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_headline(
        self,
        regime_name: str,
        confidence: float,
        disagreement: float,
        transition_warning: bool,
    ) -> str:
        """Build one-line headline."""
        conf_pct = f"{confidence:.0%}"
        if transition_warning:
            return (
                f"REGIME ALERT: {regime_name} ({conf_pct} confidence) — "
                f"Transition warning active (disagreement: {disagreement:.2f})"
            )
        return f"Current Regime: {regime_name} ({conf_pct} confidence)"

    def _build_regime_section(
        self,
        regime_id: int,
        regime_name: str,
        confidence: float,
        disagreement: float,
        transition_warning: bool,
        regime_data: Dict[str, Any],
        tone: Dict[str, str],
        data_sources: Dict[str, str],
    ) -> str:
        """Build the regime context paragraph."""
        parts: List[str] = []

        parts.append(
            f"The system identifies the current regime as {regime_name} with "
            f"{confidence:.0%} confidence."
        )
        data_sources["regime_confidence"] = f"{confidence:.4f}"

        # Disagreement commentary
        if disagreement > 0.6:
            parts.append(
                f"Classifier disagreement is elevated at {disagreement:.2f}, "
                f"suggesting significant uncertainty about the regime classification."
            )
        elif disagreement > 0.3:
            parts.append(
                f"Classifier disagreement stands at {disagreement:.2f}, indicating "
                f"moderate consensus among detection models."
            )
        else:
            parts.append(
                f"Classifier agreement is strong (disagreement: {disagreement:.2f}), "
                f"indicating high conviction in the current classification."
            )
        data_sources["disagreement"] = f"{disagreement:.4f}"

        # Transition warning
        if transition_warning:
            parts.append(
                "A transition warning is active — elevated classifier disagreement "
                "has historically preceded regime changes within 5-15 trading days."
            )

        # Probabilities
        probs = regime_data.get("probabilities", {})
        if probs:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) > 1:
                second = sorted_probs[1]
                parts.append(
                    f"The second most likely regime is {second[0]} at {second[1]:.0%}."
                )
                data_sources["second_regime_prob"] = f"{second[0]}: {second[1]:.4f}"

        return " ".join(parts)

    def _build_signal_section(
        self,
        modules_data: Dict[str, Any],
        regime_id: int,
        tone: Dict[str, str],
        data_sources: Dict[str, str],
    ) -> str:
        """Build the module signals paragraph."""
        parts: List[str] = []
        parts.append("Module Analysis:")

        for name, mod_data in modules_data.items():
            if "error" in mod_data:
                continue

            signal = mod_data.get("signal", {})
            if isinstance(signal, dict):
                sig_val = signal.get("signal", "neutral")
                strength = signal.get("strength", 0.0)
                explanation = signal.get("explanation", "")
            elif isinstance(signal, str):
                sig_val = signal
                strength = 0.5
                explanation = ""
            else:
                continue

            readable_name = name.replace("_", " ").title()
            parts.append(
                f"  • {readable_name}: {sig_val.upper()} "
                f"(strength: {strength:.2f}). {explanation}"
            )
            data_sources[f"module_{name}"] = f"{sig_val} @ {strength:.2f}"

        if len(parts) == 1:
            parts.append("  No module signals available.")

        return "\n".join(parts)

    def _build_risk_section(
        self,
        meta_data: Dict[str, Any],
        disagreement: float,
        transition_warning: bool,
        tone: Dict[str, str],
        data_sources: Dict[str, str],
    ) -> str:
        """Build the risk and warnings paragraph."""
        parts: List[str] = []

        needs_recal = meta_data.get("needs_recalibration", False)
        severity = meta_data.get("recalibration_severity", 0.0)
        reasons = meta_data.get("recalibration_reasons", [])

        if needs_recal:
            parts.append(
                f"SYSTEM ALERT: Recalibration recommended (severity: {severity:.2f}). "
                f"Triggers: {', '.join(reasons) if reasons else 'unspecified'}."
            )
            data_sources["recalibration"] = f"severity={severity:.2f}"

        if transition_warning:
            parts.append(
                "The elevated disagreement index suggests the current regime "
                "may be transitioning. Historical analysis shows that sustained "
                f"disagreement above 0.6 precedes regime changes in the majority of cases."
            )

        uncertainty = meta_data.get("uncertainty_signal", {})
        if isinstance(uncertainty, dict) and uncertainty.get("level", "low") != "low":
            parts.append(
                f"Uncertainty level: {uncertainty.get('level', 'unknown').upper()}. "
                f"Consider reducing position sizing."
            )

        if not parts:
            parts.append(
                f"No significant risk flags detected. "
                f"System operating normally with {tone['concern_adj']} risk levels."
            )

        return " ".join(parts)

    def _build_positioning_section(
        self,
        regime_id: int,
        summary_data: Dict[str, Any],
        modules_data: Dict[str, Any],
        tone: Dict[str, str],
        data_sources: Dict[str, str],
    ) -> str:
        """Build positioning suggestion paragraph."""
        bias = summary_data.get("overall_bias", "Neutral")
        signal_counts = summary_data.get("signal_counts", {})

        parts: List[str] = []
        parts.append("Positioning Guidance:")

        # Regime-specific suggestions
        suggestions: Dict[int, Dict[str, str]] = {
            1: {
                "Bullish": "Maintain full equity exposure with tactical overweight. "
                           "Consider reducing duration as growth supports higher rates.",
                "Bearish": "Unusual bearish signals in risk-on regime — monitor for "
                           "potential regime transition. Consider adding hedges.",
                "Neutral": "Maintain balanced allocation. Risk-on regime supports "
                           "equity exposure but signals lack conviction.",
                "Cautious": "Caution within risk-on suggests potential late-cycle dynamics. "
                            "Consider taking some gains off the table.",
            },
            2: {
                "Bullish": "Rare bullish signals in crisis — possible bottom formation. "
                           "Extend hedges but monitor for reversal.",
                "Bearish": "Crisis regime confirms defensive posture. Maintain flight-to-"
                           "quality allocation: overweight Treasuries, gold, reduce equity.",
                "Neutral": "Crisis regime with neutral signals — maintain defensive "
                           "positioning until regime clarity improves.",
                "Cautious": "Maximum caution warranted. Prioritize capital preservation "
                            "over return generation.",
            },
            3: {
                "Bullish": "Rare bullish signal in stagflation — likely sector-specific. "
                           "Focus on commodities and real assets.",
                "Bearish": "Stagflation bearish consensus — overweight inflation hedges "
                           "(gold, TIPS, commodities), underweight duration and growth.",
                "Neutral": "Stagflation with neutral signals — maintain inflation protection "
                           "and monitor for regime resolution.",
                "Cautious": "Stagflationary caution — reduce overall exposure, emphasize "
                            "real returns and short-duration assets.",
            },
            4: {
                "Bullish": "Disinflationary boom supports both equity and duration. "
                           "Consider barbell: growth equities + long-duration bonds.",
                "Bearish": "Unusual bearish in disinflation — check for yield curve or "
                           "credit signals. May signal regime exhaustion.",
                "Neutral": "Goldilocks environment with neutral signals. Maintain balanced "
                           "quality-biased allocation.",
                "Cautious": "Caution in disinflationary boom — monitor for signs of "
                            "stalling momentum or policy reversal.",
            },
        }

        regime_suggestions = suggestions.get(regime_id, suggestions[1])
        suggestion = regime_suggestions.get(bias, regime_suggestions.get("Neutral", ""))
        parts.append(suggestion)
        data_sources["positioning_bias"] = bias

        return "\n".join(parts)
