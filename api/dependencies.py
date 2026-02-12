"""
Dependency injection for the AMRCAIS Dashboard API.

Provides:
- AMRCAIS singleton instance (initialized on startup)
- Startup/shutdown lifespan manager
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from src.main import AMRCAIS

logger = logging.getLogger(__name__)

# ─── Singleton state ──────────────────────────────────────────────

_system: Optional[AMRCAIS] = None
_startup_time: Optional[float] = None
_last_analysis: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage AMRCAIS lifecycle — initialize on startup, cleanup on shutdown."""
    global _system, _startup_time

    logger.info("Starting AMRCAIS system initialization…")
    _startup_time = time.time()

    _system = AMRCAIS(config_path="config")
    try:
        _system.initialize(lookback_days=365)
        logger.info("AMRCAIS initialized successfully")
    except Exception as exc:
        logger.warning(f"AMRCAIS init with live data failed ({exc}); "
                       "system available in degraded mode")
        # Keep the partially initialized system so the API can still serve
        # structural / static endpoints (regime names, config params, etc.)

    yield  # ── app is running ──

    logger.info("Shutting down AMRCAIS…")
    _system = None
    _startup_time = None


# ─── Dependency getters ──────────────────────────────────────────


def get_system() -> AMRCAIS:
    """Return the singleton AMRCAIS instance.

    Raises:
        RuntimeError: If the system has not been initialized yet.
    """
    if _system is None:
        raise RuntimeError("AMRCAIS system is not initialized")
    return _system


def get_startup_time() -> float:
    """Return the epoch timestamp when the server started."""
    return _startup_time or time.time()


def get_cached_analysis() -> Optional[dict]:
    """Return the most recent full analysis result (may be None)."""
    return _last_analysis


def set_cached_analysis(result: dict) -> None:
    """Cache the latest full analysis result."""
    global _last_analysis
    _last_analysis = result
