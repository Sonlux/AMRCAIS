"""
AMRCAIS Dashboard API — FastAPI application entry point.

Run in development mode:
    uvicorn api.main:app --reload --port 8000

Swagger docs available at:
    http://localhost:8000/docs
"""

import logging
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import lifespan, get_startup_time
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

# ─── CORS ─────────────────────────────────────────────────────────
# Allow the Next.js dev server (port 3000) and any localhost origin.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
