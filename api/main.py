"""
AMRCAIS Dashboard API — FastAPI application entry point.

Run in development mode:
    uvicorn api.main:app --reload --port 8000

Swagger docs available at:
    http://localhost:8000/docs
"""

import logging
import os
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import lifespan, get_startup_time
from api.middleware import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    register_exception_handlers,
)
from api.routes import regime, modules, data, meta, backtest
from api.schemas import HealthCheckResponse, StatusResponse
from api.dependencies import get_system

logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AMRCAIS Dashboard API",
    description=(
        "Backend API for the Adaptive Multi-Regime Cross-Asset "
        "Intelligence System dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Security ─────────────────────────────────────────────────────
# Global exception handlers — prevent stack trace leakage in responses
register_exception_handlers(app)

# Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting — 120 requests / 60 s per IP, burst limit 30 / 5 s
app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)

# Request body size limit — 1 MB
app.add_middleware(RequestSizeLimitMiddleware)

# ─── CORS ─────────────────────────────────────────────────────────
# Origins sourced from env var for production flexibility.
_cors_origins = os.getenv(
    "AMRCAIS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ─── Routers ──────────────────────────────────────────────────────

app.include_router(regime.router, prefix="/api/regime", tags=["Regime"])
app.include_router(modules.router, prefix="/api/modules", tags=["Modules"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(meta.router, prefix="/api/meta", tags=["Meta-Learning"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])


# ─── System Routes ────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Simple health probe — always returns 200 if the process is alive."""
    return HealthCheckResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now(),
    )


@app.get("/api/status", response_model=StatusResponse)
async def system_status():
    """Return AMRCAIS initialization state and current regime."""
    try:
        system = get_system()
        state = system.get_current_state()
        modules_loaded = list(system.modules.keys()) if system.modules else []
        uptime = time.time() - get_startup_time()

        return StatusResponse(
            is_initialized=state.get("is_initialized", False),
            current_regime=state.get("regime"),
            confidence=state.get("confidence", 0.0),
            disagreement=state.get("disagreement", 0.0),
            modules_loaded=modules_loaded,
            uptime_seconds=round(uptime, 1),
        )
    except RuntimeError:
        return StatusResponse(is_initialized=False)
