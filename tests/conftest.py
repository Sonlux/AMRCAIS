"""
Test configuration for AMRCAIS.
"""

import pytest
import sys
import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── API Test Helpers ─────────────────────────────────────────────


def _create_mock_system():
    """Create a fully configured mock AMRCAIS instance for API tests."""
    system = MagicMock()
    system._is_initialized = True
    system._current_regime = 1

    # -- Modules --
    mock_modules = {}
    for name in ["macro", "yield_curve", "options", "factors", "correlations"]:
        mod = MagicMock()
        mod.get_regime_parameters.return_value = {
            "threshold": 0.5,
            "sensitivity": "medium",
        }
        mock_modules[name] = mod
    system.modules = mock_modules

    # -- Ensemble --
    ensemble = MagicMock()
    ensemble.weights = {
        "hmm": 0.30,
        "ml": 0.25,
        "correlation": 0.25,
        "volatility": 0.20,
    }
    ensemble._prediction_history = []
    ensemble._disagreement_history = []
    system.ensemble = ensemble

    # -- Meta learner --
    meta = MagicMock()
    meta.enable_adaptive_weights = True
    meta._recalibration_count = 2
    meta._last_recalibration = None

    # Performance metrics mock
    perf = MagicMock()
    perf.regime_stability_score = 0.82
    perf.stability_rating = "stable"
    perf.transition_count = 5
    perf.avg_disagreement = 0.18
    perf.high_disagreement_days = 3
    perf.total_classifications = 100
    perf.regime_distribution = {1: 45, 2: 20, 3: 15, 4: 20}
    meta.get_performance_metrics.return_value = perf

    meta.get_adaptive_weights.return_value = {
        "hmm": 0.30,
        "ml": 0.25,
        "correlation": 0.25,
        "volatility": 0.20,
    }

    # Recalibration decision
    recal = MagicMock()
    recal.should_recalibrate = False
    recal.reasons = []
    recal.severity = 0.0
    recal.urgency_level = "none"
    recal.recommendations = []
    meta.check_recalibration_needed.return_value = recal

    # Health report
    meta.get_system_health_report.return_value = {
        "system_status": "healthy",
        "recalibration": {
            "needed": False,
            "urgency": "none",
            "severity": 0.0,
            "reasons": [],
        },
        "performance_30d": {"stability": 0.82},
        "performance_7d": {"stability": 0.85},
        "alerts": {},
        "adaptive_weights": {
            "hmm": 0.30,
            "ml": 0.25,
            "correlation": 0.25,
            "volatility": 0.20,
        },
    }
    system.meta_learner = meta

    # -- analyze() --
    system.analyze.return_value = {
        "regime": {
            "id": 1,
            "name": "Risk-On Growth",
            "confidence": 0.85,
            "disagreement": 0.15,
            "individual_predictions": {
                "hmm": 1,
                "ml": 1,
                "correlation": 2,
                "volatility": 1,
            },
            "probabilities": {"1": 0.6, "2": 0.2, "3": 0.1, "4": 0.1},
            "transition_warning": False,
        },
        "modules": {
            "macro": {
                "signal": {
                    "signal": "bullish",
                    "strength": 0.7,
                    "confidence": 0.8,
                    "explanation": "GDP growth strong",
                    "regime_context": "Risk-on favors equities",
                },
                "details": {"gdp_growth": 2.5},
            },
            "yield_curve": {
                "signal": {
                    "signal": "neutral",
                    "strength": 0.3,
                    "confidence": 0.6,
                    "explanation": "Curve flat",
                    "regime_context": "Watching closely",
                },
                "details": {"steepness": 0.1},
            },
            "options": {
                "signal": {
                    "signal": "bearish",
                    "strength": 0.5,
                    "confidence": 0.7,
                    "explanation": "VIX elevated",
                    "regime_context": "Risk hedging advised",
                },
                "details": {"vix": 22.5},
            },
            "factors": {
                "signal": {
                    "signal": "bullish",
                    "strength": 0.6,
                    "confidence": 0.75,
                    "explanation": "Quality rotating in",
                    "regime_context": "Growth factors leading",
                },
                "details": {"momentum_factor": 1.2},
            },
            "correlations": {
                "signal": {
                    "signal": "cautious",
                    "strength": 0.4,
                    "confidence": 0.55,
                    "explanation": "Cross-asset correlations rising",
                    "regime_context": "Regime uncertainty",
                },
                "details": {"avg_correlation": 0.6},
            },
        },
    }

    # -- get_current_state --
    system.get_current_state.return_value = {
        "is_initialized": True,
        "regime": 1,
        "confidence": 0.85,
        "disagreement": 0.15,
    }

    # -- Market data (None → endpoints return empty / backtest uses fallback) --
    system.market_data = None

    return system


@pytest.fixture
def mock_system():
    """Configured mock AMRCAIS instance."""
    return _create_mock_system()


@pytest.fixture
def api_client(mock_system):
    """FastAPI TestClient with mocked AMRCAIS system.

    The lifespan is bypassed — mock system is injected directly
    into the dependencies module to avoid heavy initialization
    and conflicts with gevent monkey-patching (locust).

    Rate limiting middleware is also bypassed to avoid 429s
    when running many tests in rapid succession.
    """
    import api.dependencies as deps
    from contextlib import asynccontextmanager

    # Save originals
    orig_system = deps._system
    orig_time = deps._startup_time
    orig_analysis = deps._last_analysis

    # Inject mock directly
    deps._system = mock_system
    deps._startup_time = time.time()
    deps._last_analysis = None

    from api.main import app

    # Replace lifespan with a no-op so TestClient doesn't trigger
    # the real AMRCAIS initialization.
    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    saved_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan

    # Bypass rate limiting for tests — prevents 429s from burst limits
    from api.middleware import RateLimitMiddleware, CSRFMiddleware
    _orig_dispatch = RateLimitMiddleware.dispatch
    _orig_csrf_dispatch = CSRFMiddleware.dispatch

    async def _passthrough(self, request, call_next):
        return await call_next(request)

    RateLimitMiddleware.dispatch = _passthrough
    CSRFMiddleware.dispatch = _passthrough

    from starlette.testclient import TestClient

    with TestClient(app) as client:
        yield client

    # Restore everything
    RateLimitMiddleware.dispatch = _orig_dispatch
    CSRFMiddleware.dispatch = _orig_csrf_dispatch
    app.router.lifespan_context = saved_lifespan
    deps._system = orig_system
    deps._startup_time = orig_time
    deps._last_analysis = orig_analysis