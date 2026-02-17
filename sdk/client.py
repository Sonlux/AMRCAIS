"""
AMRCAIS Python SDK (Phase 4.4).

A clean, ergonomic Python client for the AMRCAIS REST API.
Provides typed methods for all API endpoints across all phases.

Installation:
    pip install amrcais  (when published)
    # Or:
    from sdk.client import AMRCAISClient

Usage:
    >>> client = AMRCAISClient("http://localhost:8000", api_key="your-key")
    >>> regime = client.get_current_regime()
    >>> print(f"Regime: {regime['regime_name']} ({regime['confidence']:.0%})")

    >>> signals = client.get_module_signals()
    >>> for s in signals['signals']:
    ...     print(f"  {s['module']}: {s['signal']} ({s['strength']:.2f})")

    >>> allocation = client.get_portfolio_allocation(method="black_litterman")
    >>> print(allocation['blended_weights'])

Classes:
    AMRCAISClient: Main SDK client class.

Dependencies:
    requests (pip install requests)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AMRCAISClient:
    """Python SDK client for the AMRCAIS API.

    Provides typed, ergonomic access to all AMRCAIS API endpoints
    with built-in error handling, retry logic, and convenience methods.

    Args:
        base_url: AMRCAIS API base URL.
        api_key: API authentication key.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.

    Example:
        >>> client = AMRCAISClient("http://localhost:8000")
        >>> regime = client.get_current_regime()
        >>> print(regime['regime_name'])
        'Risk-On Growth'
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

        try:
            import requests
            self._session = requests.Session()
        except ImportError:
            raise ImportError(
                "requests package required. Install with: pip install requests"
            )

        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    # ── Core Request Method ───────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
    ) -> Any:
        """Make an API request with retry logic.

        Args:
            method: HTTP method ("GET", "POST", etc.).
            path: API endpoint path (e.g., "/api/v1/regime/current").
            params: Query parameters.
            json_body: JSON request body.

        Returns:
            Parsed JSON response.

        Raises:
            AMRCAISError: If the request fails after all retries.
        """
        import time

        url = f"{self._base_url}{path}"
        last_error = None

        for attempt in range(self._max_retries):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()

            except Exception as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}): {exc}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise AMRCAISError(f"Request to {url} failed after {self._max_retries} attempts: {last_error}")

    def _get(self, path: str, **params) -> Any:
        """GET request helper."""
        clean_params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=clean_params)

    def _post(self, path: str, body: Optional[Dict] = None) -> Any:
        """POST request helper."""
        return self._request("POST", path, json_body=body)

    # ── Phase 0: System Status ────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """Check API health and connectivity.

        Returns:
            Health status dict.
        """
        return self._get("/health")

    def get_system_status(self) -> Dict[str, Any]:
        """Get AMRCAIS system status.

        Returns:
            System status with component states.
        """
        return self._get("/api/v1/status")

    # ── Phase 1: Regime Detection ─────────────────────────────

    def get_current_regime(self) -> Dict[str, Any]:
        """Get the current market regime classification.

        Returns:
            Dict with regime, regime_name, confidence, probabilities.

        Example:
            >>> regime = client.get_current_regime()
            >>> print(f"Regime {regime['regime']}: {regime['regime_name']}")
        """
        return self._get("/api/v1/regime/current")

    def get_regime_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get regime classification history.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Dict with history list and metadata.
        """
        return self._get(
            "/api/v1/regime/history",
            start_date=start_date,
            end_date=end_date,
        )

    def get_classifier_votes(self) -> Dict[str, Any]:
        """Get individual classifier votes.

        Returns:
            Dict with per-classifier votes and ensemble result.
        """
        return self._get("/api/v1/regime/classifier-votes")

    def get_transition_matrix(self) -> Dict[str, Any]:
        """Get the regime transition probability matrix.

        Returns:
            Dict with 4x4 transition matrix and regime names.
        """
        return self._get("/api/v1/regime/transition-matrix")

    def get_disagreement(self) -> Dict[str, Any]:
        """Get regime disagreement index time series.

        Returns:
            Dict with disagreement series and statistics.
        """
        return self._get("/api/v1/regime/disagreement")

    # ── Phase 2: Module Signals ───────────────────────────────

    def get_module_signals(self) -> Dict[str, Any]:
        """Get all analytical module signals.

        Returns:
            Dict with signals array and regime context.

        Example:
            >>> signals = client.get_module_signals()
            >>> for s in signals['signals']:
            ...     print(f"{s['module']}: {s['signal']}")
        """
        return self._get("/api/v1/modules/summary")

    def get_module_analysis(self, module_name: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific module.

        Args:
            module_name: Module identifier (e.g., "macro", "yield_curve").

        Returns:
            Full analysis result with raw metrics.
        """
        return self._get(f"/api/v1/modules/{module_name}")

    def get_signal_history(
        self,
        module_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get historical signal data for a module.

        Args:
            module_name: Module identifier.
            start_date: Start date.
            end_date: End date.

        Returns:
            Signal history with dates and values.
        """
        return self._get(
            f"/api/v1/modules/{module_name}/history",
            start_date=start_date,
            end_date=end_date,
        )

    # ── Phase 3: Predictions ──────────────────────────────────

    def get_return_forecast(self, horizon: int = 5) -> Dict[str, Any]:
        """Get return forecasts for tracked assets.

        Args:
            horizon: Forecast horizon in trading days.

        Returns:
            Forecast results per asset.
        """
        return self._get("/api/v1/prediction/forecast", horizon=horizon)

    def get_portfolio_allocation(
        self,
        method: str = "mean_variance",
    ) -> Dict[str, Any]:
        """Get optimal portfolio allocation.

        Args:
            method: Optimization method ("mean_variance" or "black_litterman").

        Returns:
            OptimizationResult with blended weights.

        Example:
            >>> alloc = client.get_portfolio_allocation("black_litterman")
            >>> print(alloc['blended_weights'])
        """
        return self._get("/api/v1/prediction/allocation", method=method)

    def get_transition_forecast(self) -> Dict[str, Any]:
        """Get regime transition probability forecast.

        Returns:
            Forecast transition probabilities.
        """
        return self._get("/api/v1/prediction/transition-forecast")

    # ── Phase 4: Real-Time ────────────────────────────────────

    def get_alerts(
        self,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get system alerts.

        Args:
            alert_type: Filter by type.
            severity: Filter by severity.
            limit: Maximum number of alerts.

        Returns:
            List of alert dicts.
        """
        return self._get(
            "/api/v1/alerts",
            alert_type=alert_type,
            severity=severity,
            limit=limit,
        )

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            Acknowledgement confirmation.
        """
        return self._post(f"/api/v1/alerts/{alert_id}/acknowledge")

    def get_paper_trading_status(self) -> Dict[str, Any]:
        """Get paper trading engine status.

        Returns:
            Trading engine status with portfolio summary.
        """
        return self._get("/api/v1/paper-trading/status")

    def submit_paper_trade(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
    ) -> Dict[str, Any]:
        """Submit a paper trade.

        Args:
            symbol: Ticker symbol.
            qty: Number of shares.
            side: "buy" or "sell".

        Returns:
            Trade execution result.
        """
        return self._post(
            "/api/v1/paper-trading/trade",
            body={"symbol": symbol, "qty": qty, "side": side},
        )

    # ── Phase 5: Knowledge Base ───────────────────────────────

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get knowledge base summary.

        Returns:
            Summary with transition count, anomalies, etc.
        """
        return self._get("/api/v1/knowledge/summary")

    def find_similar_transitions(
        self,
        indicators: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find historical transitions similar to current conditions.

        Args:
            indicators: Current indicator values.
            top_k: Number of matches to return.

        Returns:
            List of PatternMatch dicts.
        """
        return self._post(
            "/api/v1/knowledge/similar-transitions",
            body={"indicators": indicators, "top_k": top_k},
        )

    def get_alt_data_signals(self, regime: int = 1) -> Dict[str, Any]:
        """Get alternative data signals.

        Args:
            regime: Current regime for interpretation.

        Returns:
            Alt data signals with regime interpretation.
        """
        return self._get("/api/v1/knowledge/alt-data", regime=regime)

    # ── Convenience Methods ───────────────────────────────────

    def get_full_analysis(self) -> Dict[str, Any]:
        """Get a complete market analysis snapshot.

        Combines regime, signals, and predictions into a single response.

        Returns:
            Dict with regime, signals, forecast, and allocation.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "regime": self.get_current_regime(),
            "signals": self.get_module_signals(),
            "forecast": self.get_return_forecast(),
            "allocation": self.get_portfolio_allocation(),
        }
        return result

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> "AMRCAISClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AMRCAISClient(base_url='{self._base_url}')"


class AMRCAISError(Exception):
    """Error from AMRCAIS API operations."""
    pass
