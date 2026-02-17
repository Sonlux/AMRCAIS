"""
Research Publication Pipeline — Phase 5.4.

Turns AMRCAIS institutional memory into publishable research artifacts.
Auto-generates regime transition case studies, backtest reports, and
factor exposure analysis papers.

Published research creates credibility → attracts users → more data →
better model → more research.  The research flywheel.

Classes:
    ResearchReport: Generated report with title, sections, and metadata.
    ResearchPublisher: Engine for auto-generating regime research.

Example:
    >>> publisher = ResearchPublisher(knowledge_base=kb)
    >>> report = publisher.generate_transition_case_study(
    ...     from_regime=1, to_regime=2)
    >>> print(report.title)
    'Regime Transition Case Study: Risk-On Growth → Risk-Off Crisis'
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regime name mapping for human-readable reports
REGIME_NAMES: Dict[int, str] = {
    1: "Risk-On Growth",
    2: "Risk-Off Crisis",
    3: "Stagflation",
    4: "Disinflationary Boom",
}


@dataclass
class ReportSection:
    """Single section of a research report.

    Attributes:
        heading: Section title.
        content: Section body text.
        data: Optional structured data (tables, charts).
    """

    heading: str = ""
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "heading": self.heading,
            "content": self.content,
            "data": self.data,
        }


@dataclass
class ResearchReport:
    """Generated research report.

    Attributes:
        report_id: Unique identifier.
        title: Report title.
        report_type: Category (case_study, backtest, factor_analysis).
        generated_at: ISO timestamp.
        sections: Ordered list of report sections.
        metadata: Additional context (regime, date range, etc.).
        summary: Executive summary.
    """

    report_id: str = ""
    title: str = ""
    report_type: str = ""
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API / JSON."""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "report_type": self.report_type,
            "generated_at": self.generated_at,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        """Render report as Markdown text.

        Returns:
            Markdown-formatted report string.
        """
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {self.generated_at}*",
            f"*Type: {self.report_type}*",
            "",
        ]

        if self.summary:
            lines.extend(["## Executive Summary", "", self.summary, ""])

        for section in self.sections:
            lines.extend([f"## {section.heading}", "", section.content, ""])
            if section.data:
                lines.append("```json")
                lines.append(json.dumps(section.data, indent=2))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


class ResearchPublisher:
    """Engine for auto-generating regime-intelligence research.

    Takes the knowledge base (transitions, anomalies, macro impacts)
    and turns it into structured reports suitable for publication.

    Args:
        knowledge_base: KnowledgeBase instance for data retrieval.
        output_dir: Directory to save generated reports.

    Example:
        >>> publisher = ResearchPublisher(knowledge_base=kb)
        >>> report = publisher.generate_transition_case_study(
        ...     from_regime=1, to_regime=2)
    """

    def __init__(
        self,
        knowledge_base: Any = None,
        output_dir: str = "data/reports",
    ) -> None:
        self._kb = knowledge_base
        self._output_dir = Path(output_dir)
        self._reports: List[ResearchReport] = []

    # ── Report Generation ─────────────────────────────────────

    def generate_transition_case_study(
        self,
        from_regime: Optional[int] = None,
        to_regime: Optional[int] = None,
        limit: int = 20,
    ) -> ResearchReport:
        """Generate a regime transition case study.

        Analyzes historical transitions of the specified type and
        produces a structured report with leading indicators,
        classifier accuracy, and post-transition performance.

        Args:
            from_regime: Source regime filter.
            to_regime: Destination regime filter.
            limit: Maximum transitions to include.

        Returns:
            ResearchReport with transition analysis.
        """
        import uuid

        from_name = REGIME_NAMES.get(from_regime, f"Regime {from_regime}") if from_regime else "Any"
        to_name = REGIME_NAMES.get(to_regime, f"Regime {to_regime}") if to_regime else "Any"

        title = f"Regime Transition Case Study: {from_name} → {to_name}"
        report_id = uuid.uuid4().hex[:12]

        sections: List[ReportSection] = []

        # Section 1: Overview
        transitions = []
        if self._kb:
            transitions = self._kb.get_transitions(
                from_regime=from_regime, to_regime=to_regime, limit=limit
            )

        sections.append(
            ReportSection(
                heading="Overview",
                content=(
                    f"This report analyzes {len(transitions)} historical "
                    f"regime transitions from {from_name} to {to_name}. "
                    f"Each transition is evaluated for leading indicators, "
                    f"detection latency, and post-transition performance."
                ),
                data={"transition_count": len(transitions)},
            )
        )

        # Section 2: Leading Indicators Analysis
        indicator_stats = self._analyze_leading_indicators(transitions)
        sections.append(
            ReportSection(
                heading="Leading Indicators",
                content=(
                    f"Analysis of leading indicators across "
                    f"{len(transitions)} transitions reveals the "
                    f"following patterns."
                ),
                data=indicator_stats,
            )
        )

        # Section 3: Classifier Performance
        clf_stats = self._analyze_classifier_accuracy(transitions)
        sections.append(
            ReportSection(
                heading="Classifier Performance",
                content=(
                    "Performance of individual classifiers in detecting "
                    "this transition type."
                ),
                data=clf_stats,
            )
        )

        # Section 4: Detection Latency
        latency_stats = self._analyze_detection_latency(transitions)
        sections.append(
            ReportSection(
                heading="Detection Latency",
                content=(
                    "Analysis of how quickly the ensemble detected the "
                    "transition relative to the actual regime change."
                ),
                data=latency_stats,
            )
        )

        # Section 5: Post-Transition Performance
        perf_stats = self._analyze_post_performance(transitions)
        sections.append(
            ReportSection(
                heading="Post-Transition Performance",
                content=(
                    "Asset performance following the detected transition."
                ),
                data=perf_stats,
            )
        )

        # Build summary
        summary = (
            f"Analyzed {len(transitions)} transitions from "
            f"{from_name} to {to_name}. "
        )
        if latency_stats.get("avg_latency") is not None:
            summary += (
                f"Average detection latency: "
                f"{latency_stats['avg_latency']:.1f} days. "
            )
        if clf_stats.get("best_classifier"):
            summary += (
                f"Most reliable classifier: "
                f"{clf_stats['best_classifier']}. "
            )

        report = ResearchReport(
            report_id=report_id,
            title=title,
            report_type="transition_case_study",
            sections=sections,
            metadata={
                "from_regime": from_regime,
                "to_regime": to_regime,
                "sample_size": len(transitions),
            },
            summary=summary,
        )

        self._reports.append(report)
        return report

    def generate_backtest_report(
        self,
        backtest_results: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """Generate a backtest results report.

        Args:
            backtest_results: Dict with backtest metrics.

        Returns:
            ResearchReport with backtest analysis.
        """
        import uuid

        results = backtest_results or {}
        report_id = uuid.uuid4().hex[:12]

        sections: List[ReportSection] = []

        # Overview
        sections.append(
            ReportSection(
                heading="Backtest Overview",
                content=(
                    "Walk-forward validation backtest results for the "
                    "AMRCAIS regime detection system."
                ),
                data={
                    "method": "walk-forward",
                    "periods": results.get("periods", 0),
                    "total_return": results.get("total_return", 0.0),
                },
            )
        )

        # Regime Detection Accuracy
        sections.append(
            ReportSection(
                heading="Regime Detection Accuracy",
                content="Accuracy metrics by regime type.",
                data={
                    "overall_accuracy": results.get("accuracy", 0.0),
                    "by_regime": results.get("accuracy_by_regime", {}),
                },
            )
        )

        # Alpha Attribution
        sections.append(
            ReportSection(
                heading="Alpha Attribution",
                content=(
                    "Decomposition of returns into regime-detection alpha "
                    "and market beta."
                ),
                data={
                    "regime_alpha": results.get("regime_alpha", 0.0),
                    "market_beta": results.get("market_beta", 0.0),
                    "transition_alpha": results.get("transition_alpha", 0.0),
                },
            )
        )

        summary = (
            f"Backtest over {results.get('periods', 0)} periods. "
            f"Overall accuracy: {results.get('accuracy', 0.0):.1%}. "
            f"Regime alpha: {results.get('regime_alpha', 0.0):+.2%}."
        )

        report = ResearchReport(
            report_id=report_id,
            title="AMRCAIS Backtest Report",
            report_type="backtest_report",
            sections=sections,
            metadata={"results_summary": results},
            summary=summary,
        )

        self._reports.append(report)
        return report

    def generate_factor_analysis(
        self,
        factor_data: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """Generate a regime-conditional factor exposure analysis.

        Args:
            factor_data: Factor exposure data by regime.

        Returns:
            ResearchReport with factor analysis.
        """
        import uuid

        data = factor_data or {}
        report_id = uuid.uuid4().hex[:12]

        sections: List[ReportSection] = []

        sections.append(
            ReportSection(
                heading="Factor Exposure Overview",
                content=(
                    "Regime-conditional factor exposures across major "
                    "risk factors. Factor behaviour changes significantly "
                    "across regimes — a key AMRCAIS insight."
                ),
                data=data.get("exposures", {}),
            )
        )

        sections.append(
            ReportSection(
                heading="Regime-Conditional Returns",
                content=(
                    "Factor return distribution by regime. Some factors "
                    "reverse sign across regimes."
                ),
                data=data.get("returns_by_regime", {}),
            )
        )

        sections.append(
            ReportSection(
                heading="Disagreement Index Analysis",
                content=(
                    "Empirical analysis of the disagreement index as a "
                    "leading indicator of regime transitions."
                ),
                data=data.get("disagreement_analysis", {}),
            )
        )

        summary = (
            "Factor exposure analysis across regime states. "
            f"Factors analyzed: {len(data.get('exposures', {}))}."
        )

        report = ResearchReport(
            report_id=report_id,
            title="Regime-Conditional Factor Exposure Analysis",
            report_type="factor_analysis",
            sections=sections,
            metadata={"factor_data_summary": list(data.keys())},
            summary=summary,
        )

        self._reports.append(report)
        return report

    # ── Report Management ─────────────────────────────────────

    def get_reports(
        self,
        report_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[ResearchReport]:
        """Get generated reports.

        Args:
            report_type: Filter by type.
            limit: Maximum reports to return.

        Returns:
            List of reports, most recent first.
        """
        result = self._reports
        if report_type:
            result = [r for r in result if r.report_type == report_type]
        return list(reversed(result[-limit:]))

    def save_report(self, report: ResearchReport) -> str:
        """Save a report to the output directory as markdown.

        Args:
            report: Report to save.

        Returns:
            Path to the saved file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{report.report_type}_{report.report_id}.md"
        path = self._output_dir / filename
        path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(f"Report saved: {path}")
        return str(path)

    def get_summary(self) -> Dict[str, Any]:
        """Get publisher status summary.

        Returns:
            Dict with report counts and types.
        """
        by_type: Dict[str, int] = {}
        for r in self._reports:
            by_type[r.report_type] = by_type.get(r.report_type, 0) + 1

        return {
            "total_reports": len(self._reports),
            "by_type": by_type,
            "output_dir": str(self._output_dir),
        }

    # ── Internal Analysis Helpers ─────────────────────────────

    def _analyze_leading_indicators(
        self, transitions: list,
    ) -> Dict[str, Any]:
        """Aggregate leading indicator patterns across transitions."""
        all_indicators: Dict[str, List[float]] = {}

        for t in transitions:
            for ind, val in getattr(t, "leading_indicators", {}).items():
                if ind not in all_indicators:
                    all_indicators[ind] = []
                all_indicators[ind].append(val)

        stats: Dict[str, Any] = {}
        for ind, values in all_indicators.items():
            if values:
                avg = sum(values) / len(values)
                stats[ind] = {
                    "observations": len(values),
                    "avg_value": round(avg, 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }

        return {"indicators": stats, "unique_indicators": len(all_indicators)}

    def _analyze_classifier_accuracy(
        self, transitions: list,
    ) -> Dict[str, Any]:
        """Aggregate classifier accuracy across transitions."""
        clf_correct: Dict[str, int] = {}
        clf_total: Dict[str, int] = {}

        for t in transitions:
            for clf, correct in getattr(t, "classifier_accuracy", {}).items():
                clf_total[clf] = clf_total.get(clf, 0) + 1
                if correct:
                    clf_correct[clf] = clf_correct.get(clf, 0) + 1

        accuracy: Dict[str, float] = {}
        for clf in clf_total:
            accuracy[clf] = round(
                clf_correct.get(clf, 0) / clf_total[clf], 4
            )

        best = max(accuracy, key=accuracy.get) if accuracy else None  # type: ignore[arg-type]

        return {
            "classifiers": accuracy,
            "best_classifier": best,
            "sample_sizes": clf_total,
        }

    def _analyze_detection_latency(
        self, transitions: list,
    ) -> Dict[str, Any]:
        """Aggregate detection latency statistics."""
        latencies = [
            t.detection_latency_days
            for t in transitions
            if hasattr(t, "detection_latency_days")
            and t.detection_latency_days != 0
        ]

        if not latencies:
            return {"avg_latency": None, "sample_size": 0}

        import math

        avg = sum(latencies) / len(latencies)
        variance = sum((x - avg) ** 2 for x in latencies) / len(latencies)

        return {
            "avg_latency": round(avg, 2),
            "std_latency": round(math.sqrt(variance), 2),
            "min_latency": round(min(latencies), 2),
            "max_latency": round(max(latencies), 2),
            "sample_size": len(latencies),
        }

    def _analyze_post_performance(
        self, transitions: list,
    ) -> Dict[str, Any]:
        """Aggregate post-transition performance statistics."""
        all_perf: Dict[str, List[float]] = {}

        for t in transitions:
            for asset, ret in getattr(
                t, "post_transition_performance", {}
            ).items():
                if asset not in all_perf:
                    all_perf[asset] = []
                all_perf[asset].append(ret)

        stats: Dict[str, Any] = {}
        for asset, returns in all_perf.items():
            if returns:
                avg = sum(returns) / len(returns)
                stats[asset] = {
                    "observations": len(returns),
                    "avg_return": round(avg, 4),
                    "min_return": round(min(returns), 4),
                    "max_return": round(max(returns), 4),
                }

        return {"assets": stats, "assets_tracked": len(all_perf)}
