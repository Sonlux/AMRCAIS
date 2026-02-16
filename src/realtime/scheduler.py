"""
Periodic analysis scheduler for AMRCAIS (Phase 4.1).

Runs regime re-analysis at configurable intervals during market
hours and publishes results to the event bus.

Architecture:
    ┌──────────┐    tick     ┌──────────┐   event    ┌──────────┐
    │ Scheduler│ ─────────→  │ AMRCAIS  │ ────────→  │ EventBus │
    │ (asyncio)│  every N    │ analyze()│            │ pub/sub  │
    └──────────┘   minutes   └──────────┘            └──────────┘

Features:
    - Configurable interval (default 15 min)
    - Market-hours-only scheduling (9:30–16:00 ET weekdays)
    - Automatic regime change detection → REGIME_CHANGE event
    - Transition warning detection → TRANSITION_WARNING event
    - Graceful start / stop with asyncio Task

Classes:
    AnalysisScheduler: Periodic analysis runner.

Example:
    >>> scheduler = AnalysisScheduler(system, bus, interval_seconds=900)
    >>> await scheduler.start()    # begin periodic analysis
    >>> await scheduler.stop()     # graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from src.realtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger(__name__)

# US Eastern Time offset (simplified — production should use pytz/zoneinfo)
_ET_UTC_OFFSET_HOURS = -5  # EST; DST would be -4


class AnalysisScheduler:
    """Periodic regime analysis scheduler.

    Runs ``system.analyze()`` at regular intervals, compares with the
    previous result, and publishes appropriate events to the bus.

    Args:
        system: AMRCAIS instance (must be initialized).
        event_bus: EventBus for publishing analysis events.
        interval_seconds: Seconds between analysis runs (default 900 = 15 min).
        market_hours_only: If True, only run during US equity market hours.
        on_analysis_complete: Optional callback after each run.

    Example:
        >>> scheduler = AnalysisScheduler(system, bus, 900)
        >>> await scheduler.start()
    """

    def __init__(
        self,
        system: Any,
        event_bus: EventBus,
        interval_seconds: int = 900,
        market_hours_only: bool = True,
        on_analysis_complete: Optional[Callable] = None,
    ) -> None:
        self._system = system
        self._bus = event_bus
        self._interval = interval_seconds
        self._market_hours_only = market_hours_only
        self._on_complete = on_analysis_complete

        self._task: Optional[asyncio.Task] = None
        self._is_running = False
        self._last_regime: Optional[int] = None
        self._last_analysis: Optional[Dict] = None
        self._run_count = 0
        self._error_count = 0
        self._last_run_time: Optional[float] = None

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        """Start the periodic analysis loop.

        Creates an asyncio task that runs until ``stop()`` is called.
        """
        if self._is_running:
            logger.warning("Scheduler already running")
            return
        self._is_running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Analysis scheduler started (interval={self._interval}s, "
            f"market_hours_only={self._market_hours_only})"
        )

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._is_running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"Analysis scheduler stopped after {self._run_count} runs"
        )

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is actively looping."""
        return self._is_running

    @property
    def run_count(self) -> int:
        """Number of completed analysis runs."""
        return self._run_count

    @property
    def error_count(self) -> int:
        """Number of failed analysis runs."""
        return self._error_count

    @property
    def last_run_time(self) -> Optional[float]:
        """Epoch time of last completed analysis."""
        return self._last_run_time

    @property
    def last_analysis(self) -> Optional[Dict]:
        """Most recent analysis result."""
        return self._last_analysis

    # ── Core loop ─────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduling loop."""
        while self._is_running:
            try:
                if self._should_run():
                    await self._run_analysis()
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._error_count += 1
                logger.error(f"Scheduler loop error: {exc}")
                self._bus.publish(
                    Event(
                        EventType.SYSTEM_ERROR,
                        {"error": str(exc), "component": "scheduler"},
                        source="scheduler",
                    )
                )
                await asyncio.sleep(min(self._interval, 60))

    def _should_run(self) -> bool:
        """Check if we should run now (market hours filter)."""
        if not self._market_hours_only:
            return True

        now = datetime.now(timezone.utc)
        # Convert to ET (simplified)
        et_hour = (now.hour + _ET_UTC_OFFSET_HOURS) % 24
        weekday = now.weekday()  # 0=Mon, 6=Sun

        # Market hours: Mon–Fri, 9:00–16:30 ET (with 30 min buffer)
        if weekday >= 5:
            return False
        if et_hour < 9 or et_hour > 16:
            return False
        return True

    async def _run_analysis(self) -> None:
        """Execute a single analysis cycle and publish events."""
        start = time.monotonic()
        try:
            # Run analysis in executor to avoid blocking event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._system.analyze
            )
            elapsed = time.monotonic() - start
            self._run_count += 1
            self._last_run_time = time.time()
            self._last_analysis = result

            logger.info(
                f"Analysis #{self._run_count} completed in {elapsed:.2f}s"
            )

            # Detect regime changes
            regime_data = result.get("regime", {})
            current_regime = regime_data.get("id")
            confidence = regime_data.get("confidence", 0.0)
            disagreement = regime_data.get("disagreement", 0.0)

            if (
                self._last_regime is not None
                and current_regime is not None
                and current_regime != self._last_regime
            ):
                self._bus.publish(
                    Event(
                        EventType.REGIME_CHANGE,
                        {
                            "previous_regime": self._last_regime,
                            "new_regime": current_regime,
                            "regime_name": regime_data.get("name", ""),
                            "confidence": confidence,
                            "disagreement": disagreement,
                        },
                        source="scheduler",
                    )
                )

            # Transition warnings
            if disagreement > 0.6:
                self._bus.publish(
                    Event(
                        EventType.TRANSITION_WARNING,
                        {
                            "regime": current_regime,
                            "disagreement": disagreement,
                            "confidence": confidence,
                            "transition_warning": regime_data.get(
                                "transition_warning", False
                            ),
                        },
                        source="scheduler",
                    )
                )

            # Correlation anomalies
            modules_data = result.get("modules", {})
            corr_data = modules_data.get("correlations", {})
            corr_signal = corr_data.get("signal", {})
            if isinstance(corr_signal, dict):
                corr_strength = corr_signal.get("strength", 0)
                if corr_strength > 0.5:
                    self._bus.publish(
                        Event(
                            EventType.CORRELATION_ANOMALY,
                            {
                                "strength": corr_strength,
                                "details": corr_data.get("details", {}),
                            },
                            source="scheduler",
                        )
                    )

            # General analysis complete event
            self._bus.publish(
                Event(
                    EventType.ANALYSIS_COMPLETE,
                    {
                        "regime": current_regime,
                        "confidence": confidence,
                        "disagreement": disagreement,
                        "run_number": self._run_count,
                        "elapsed_seconds": round(elapsed, 3),
                    },
                    source="scheduler",
                )
            )

            self._last_regime = current_regime

            if self._on_complete:
                self._on_complete(result)

        except Exception as exc:
            self._error_count += 1
            elapsed = time.monotonic() - start
            logger.error(
                f"Analysis #{self._run_count + 1} failed "
                f"after {elapsed:.2f}s: {exc}"
            )
            self._bus.publish(
                Event(
                    EventType.SYSTEM_ERROR,
                    {
                        "error": str(exc),
                        "component": "analysis",
                        "elapsed_seconds": round(elapsed, 3),
                    },
                    source="scheduler",
                )
            )

    # ── Manual trigger ────────────────────────────────────────

    async def trigger_now(self) -> Optional[Dict]:
        """Manually trigger an immediate analysis run.

        Returns:
            Analysis result dict, or None if analysis fails.
        """
        await self._run_analysis()
        return self._last_analysis

    def get_status(self) -> Dict[str, Any]:
        """Return scheduler status summary."""
        return {
            "is_running": self._is_running,
            "interval_seconds": self._interval,
            "market_hours_only": self._market_hours_only,
            "run_count": self._run_count,
            "error_count": self._error_count,
            "last_run_time": self._last_run_time,
            "last_regime": self._last_regime,
        }
