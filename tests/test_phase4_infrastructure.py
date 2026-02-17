"""
Tests for Phase 4+ production infrastructure.

Covers:
    - Black-Litterman optimizer
    - Walk-forward backtesting harness
    - Redis EventBus (fallback mode)
    - WebSocket feed (simulated mode)
    - Alert delivery manager
    - Alpaca paper broker (dry-run mode)
    - Alt data live fetcher
    - Python SDK client
"""

import json
import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ═════════════════════════════════════════════════════════════════
# Black-Litterman Tests
# ═════════════════════════════════════════════════════════════════


class TestBlackLitterman:
    """Tests for the Black-Litterman portfolio optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        data = pd.DataFrame(
            {
                "SPX_returns": np.random.normal(0.0004, 0.01, 500),
                "TLT_returns": np.random.normal(0.0001, 0.005, 500),
                "GLD_returns": np.random.normal(0.0002, 0.008, 500),
                "WTI_returns": np.random.normal(0.0003, 0.015, 500),
                "DXY_returns": np.random.normal(0.0001, 0.003, 500),
            },
            index=dates,
        )
        regimes = pd.Series(
            np.random.choice([1, 2, 3, 4], size=500, p=[0.4, 0.2, 0.2, 0.2]),
            index=dates,
        )
        return data, regimes

    def test_fit(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)
        assert bl.is_fitted
        assert len(bl._assets) == 5

    def test_optimize_regime_1(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        result = bl.optimize(regime=1, confidence=0.8)
        assert result.regime == 1
        assert result.method == "black_litterman"
        assert len(result.optimal_weights) == 5
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01
        assert all(w >= 0 for w in result.optimal_weights.values())

    def test_optimize_all_regimes(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        for regime in [1, 2, 3, 4]:
            result = bl.optimize(regime=regime, confidence=0.7)
            assert result.regime == regime
            assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01

    def test_blended_views(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        result = bl.optimize(
            regime=1,
            confidence=0.75,
            transition_probs={1: 0.65, 2: 0.20, 3: 0.10, 4: 0.05},
        )
        assert len(result.optimal_weights) == 5
        assert abs(sum(result.optimal_weights.values()) - 1.0) < 0.01

    def test_custom_views(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        result = bl.optimize(
            regime=1,
            confidence=0.9,
            custom_views={"SPX": 0.15, "TLT": -0.05},
        )
        assert result.method == "black_litterman"
        assert len(result.optimal_weights) == 5

    def test_to_dict(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        result = bl.optimize(regime=2, confidence=0.8)
        d = result.to_dict()
        assert "method" in d
        assert "posterior_returns" in d
        assert "optimal_weights" in d

    def test_to_optimization_result(self, sample_data):
        from src.prediction.black_litterman import BlackLitterman

        data, regimes = sample_data
        bl = BlackLitterman()
        bl.fit(data, regimes)

        bl_result = bl.optimize(regime=1, confidence=0.7)
        opt_result = bl.to_optimization_result(bl_result)

        assert opt_result.current_regime == 1
        assert abs(sum(opt_result.blended_weights.values()) - 1.0) < 0.01
        assert opt_result.expected_volatility >= 0

    def test_not_fitted_raises(self):
        from src.prediction.black_litterman import BlackLitterman

        bl = BlackLitterman()
        with pytest.raises(RuntimeError, match="not fitted"):
            bl.optimize(regime=1)


# ═════════════════════════════════════════════════════════════════
# Walk-Forward Tests
# ═════════════════════════════════════════════════════════════════


class TestWalkForward:
    """Tests for the walk-forward backtesting harness."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with target column."""
        np.random.seed(42)
        n = 1000
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
                "feature_3": np.random.randn(n),
                "target": np.random.choice([0, 1, 2, 3], size=n),
            },
            index=dates,
        )
        return data

    def _simple_model_fn(self, train_df):
        """Trivial model: return the most common target."""
        from collections import Counter
        c = Counter(train_df["target"].values)
        return c.most_common(1)[0][0]

    def _simple_predict_fn(self, model, test_df):
        """Trivial predict: always predict the mode."""
        return np.full(len(test_df), model)

    def test_anchored_walk_forward(self, sample_data):
        from src.prediction.walk_forward import WalkForwardConfig, WalkForwardHarness

        config = WalkForwardConfig(n_folds=4, min_train_size=200, test_size=50)
        harness = WalkForwardHarness(config)

        result = harness.run(
            data=sample_data,
            model_fn=self._simple_model_fn,
            predict_fn=self._simple_predict_fn,
            target_col="target",
            model_name="test_model",
        )
        assert len(result.folds) > 0
        assert result.model_name == "test_model"
        assert result.total_time_seconds > 0

    def test_rolling_walk_forward(self, sample_data):
        from src.prediction.walk_forward import WalkForwardConfig, WalkForwardHarness

        config = WalkForwardConfig(
            n_folds=3, min_train_size=200, test_size=50, anchored=False
        )
        harness = WalkForwardHarness(config)

        result = harness.run(
            data=sample_data,
            model_fn=self._simple_model_fn,
            predict_fn=self._simple_predict_fn,
            target_col="target",
            model_name="rolling_test",
        )
        assert len(result.folds) > 0

    def test_fold_metrics(self, sample_data):
        from src.prediction.walk_forward import WalkForwardConfig, WalkForwardHarness

        config = WalkForwardConfig(n_folds=3, min_train_size=200, test_size=50)
        harness = WalkForwardHarness(config)

        result = harness.run(
            data=sample_data,
            model_fn=self._simple_model_fn,
            predict_fn=self._simple_predict_fn,
            target_col="target",
        )

        for fold in result.folds:
            assert fold.train_samples > 0
            assert fold.test_samples > 0
            assert fold.in_sample_accuracy >= 0
            assert fold.oos_accuracy >= 0

    def test_result_to_dict(self, sample_data):
        from src.prediction.walk_forward import WalkForwardConfig, WalkForwardHarness

        config = WalkForwardConfig(n_folds=3, min_train_size=200, test_size=50)
        harness = WalkForwardHarness(config)

        result = harness.run(
            data=sample_data,
            model_fn=self._simple_model_fn,
            predict_fn=self._simple_predict_fn,
            target_col="target",
        )

        d = result.to_dict()
        assert "verdict" in d
        assert "mean_oos_r2" in d
        assert "overfit_ratio" in d
        assert "folds" in d

    def test_r_squared_metric(self):
        from src.prediction.walk_forward import WalkForwardHarness

        # Perfect prediction
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        r2 = WalkForwardHarness._r_squared(y_true, y_true)
        assert abs(r2 - 1.0) < 1e-10

        # Mean prediction (R² ≈ 0)
        y_pred = np.full(4, np.mean(y_true))
        r2 = WalkForwardHarness._r_squared(y_true, y_pred)
        assert abs(r2) < 1e-10


# ═════════════════════════════════════════════════════════════════
# Redis EventBus Tests
# ═════════════════════════════════════════════════════════════════


class TestRedisEventBus:
    """Tests for the Redis-backed EventBus (fallback mode)."""

    def test_fallback_to_memory(self):
        """When Redis is not available, should fall back to in-process."""
        from src.realtime.redis_event_bus import RedisEventBus

        bus = RedisEventBus(redis_url="redis://nonexistent:6379/0")
        assert not bus.is_redis_connected
        # Should still work as in-process bus
        handler_called = [False]

        def handler(event):
            handler_called[0] = True

        from src.realtime.event_bus import EventType
        bus.subscribe(EventType.DATA_UPDATED, handler)

        from src.realtime.event_bus import Event
        bus.publish(Event(EventType.DATA_UPDATED, {"test": 1}))
        assert handler_called[0]
        bus.reset()

    def test_create_factory_memory(self):
        """Factory should create memory bus by default."""
        from src.realtime.redis_event_bus import RedisEventBus
        from src.realtime.event_bus import EventBus

        bus = RedisEventBus.create()
        assert isinstance(bus, EventBus)

    def test_create_factory_redis_fallback(self):
        """Factory with redis backend should fall back when unavailable."""
        from src.realtime.redis_event_bus import RedisEventBus
        from src.realtime.event_bus import EventBus

        os.environ["EVENT_BUS_BACKEND"] = "redis"
        os.environ["REDIS_URL"] = "redis://nonexistent:6379/0"
        try:
            bus = RedisEventBus.create()
            assert isinstance(bus, EventBus)
        finally:
            del os.environ["EVENT_BUS_BACKEND"]
            del os.environ["REDIS_URL"]


# ═════════════════════════════════════════════════════════════════
# WebSocket Feed Tests
# ═════════════════════════════════════════════════════════════════


class TestWebSocketFeed:
    """Tests for the WebSocket data feed."""

    def test_init(self):
        from src.realtime.websocket_feed import WebSocketFeed, QuoteUpdate
        from src.realtime.event_bus import EventBus

        bus = EventBus()
        feed = WebSocketFeed(bus, provider="simulated", symbols=["SPY", "TLT"])
        assert not feed._is_running

    def test_quote_update(self):
        from src.realtime.websocket_feed import QuoteUpdate

        q = QuoteUpdate(symbol="SPY", price=520.50, volume=1000, source="test")
        d = q.to_dict()
        assert d["symbol"] == "SPY"
        assert d["price"] == 520.50
        assert d["volume"] == 1000

    def test_status(self):
        from src.realtime.websocket_feed import WebSocketFeed
        from src.realtime.event_bus import EventBus

        bus = EventBus()
        feed = WebSocketFeed(bus, provider="simulated")
        status = feed.get_status()
        assert status["provider"] == "simulated"
        assert status["is_running"] is False
        assert status["message_count"] == 0

    def test_quote_callback(self):
        from src.realtime.websocket_feed import WebSocketFeed, QuoteUpdate
        from src.realtime.event_bus import EventBus

        bus = EventBus()
        feed = WebSocketFeed(bus, provider="simulated")

        quotes_received = []
        feed.add_quote_callback(lambda q: quotes_received.append(q))

        # Manually dispatch a quote
        q = QuoteUpdate(symbol="SPY", price=520.0, source="test")
        feed._dispatch_quote(q)

        assert len(quotes_received) == 1
        assert quotes_received[0].symbol == "SPY"


# ═════════════════════════════════════════════════════════════════
# Alert Delivery Tests
# ═════════════════════════════════════════════════════════════════


class TestAlertDelivery:
    """Tests for the alert delivery manager."""

    def test_delivery_manager_no_channels(self):
        from src.realtime.alert_delivery import AlertDeliveryManager

        manager = AlertDeliveryManager()
        result = manager.deliver({
            "title": "Test Alert",
            "message": "Test message",
            "severity": "info",
            "alert_type": "test",
        })
        assert len(result) == 0

    def test_delivery_manager_status(self):
        from src.realtime.alert_delivery import AlertDeliveryManager

        manager = AlertDeliveryManager()
        status = manager.get_status()
        assert status["total_deliveries"] == 0
        assert status["total_failures"] == 0
        assert len(status["channels"]) == 0

    def test_email_no_recipients(self):
        from src.realtime.alert_delivery import EmailDelivery

        email = EmailDelivery(to_addrs=[])
        result = email.send({"title": "Test", "severity": "info"})
        assert result is False

    def test_slack_no_webhook(self):
        from src.realtime.alert_delivery import SlackDelivery

        slack = SlackDelivery(webhook_url="")
        result = slack.send({"title": "Test", "severity": "info"})
        assert result is False

    def test_telegram_no_config(self):
        from src.realtime.alert_delivery import TelegramDelivery

        tg = TelegramDelivery(bot_token="", chat_id="")
        result = tg.send({"title": "Test", "severity": "info"})
        assert result is False

    def test_from_env_no_config(self):
        """Without env vars, no channels should be configured."""
        from src.realtime.alert_delivery import AlertDeliveryManager

        manager = AlertDeliveryManager.from_env()
        assert len(manager._channels) == 0

    def test_disabled_channel_skipped(self):
        from src.realtime.alert_delivery import (
            AlertDeliveryManager,
            DeliveryChannel,
        )

        class FakeChannel(DeliveryChannel):
            def send(self, alert_dict):
                return True

        manager = AlertDeliveryManager()
        ch = FakeChannel("fake", enabled=False)
        manager.add_channel(ch)

        result = manager.deliver({"title": "Test"})
        assert result["fake"] is False


# ═════════════════════════════════════════════════════════════════
# Alpaca Paper Broker Tests
# ═════════════════════════════════════════════════════════════════


class TestAlpacaBroker:
    """Tests for the Alpaca paper trading broker (dry-run mode)."""

    def test_dry_run_account(self):
        from src.realtime.alpaca_broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        assert not broker.is_connected
        account = broker.get_account()
        assert account["equity"] == 100_000.0
        assert account["status"] == "DRY_RUN"

    def test_dry_run_order(self):
        from src.realtime.alpaca_broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        result = broker.submit_order("SPY", qty=10, side="buy")
        assert result["status"] == "DRY_RUN"
        assert result["symbol"] == "SPY"
        assert result["qty"] == 10

    def test_dry_run_positions(self):
        from src.realtime.alpaca_broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        positions = broker.get_positions()
        assert positions == []

    def test_status(self):
        from src.realtime.alpaca_broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
        status = broker.get_status()
        assert status["connected"] is False
        assert "account" in status


# ═════════════════════════════════════════════════════════════════
# Alt Data Fetcher Tests
# ═════════════════════════════════════════════════════════════════


class TestAltDataFetcher:
    """Tests for the live alt data fetcher."""

    def test_init(self):
        from src.knowledge.alt_data_fetcher import LiveAltDataFetcher

        fetcher = LiveAltDataFetcher()
        assert fetcher._cache_minutes == 30

    def test_cache(self):
        from src.knowledge.alt_data_fetcher import LiveAltDataFetcher

        fetcher = LiveAltDataFetcher(cache_minutes=60)
        fetcher._set_cached("test_signal", 42.0)
        assert fetcher._get_cached("test_signal") == 42.0

    def test_cache_expiry(self):
        from src.knowledge.alt_data_fetcher import LiveAltDataFetcher

        fetcher = LiveAltDataFetcher(cache_minutes=0)  # immediate expiry
        fetcher._cache["test_signal"] = (42.0, time.time() - 120)
        assert fetcher._get_cached("test_signal") is None

    def test_status(self):
        from src.knowledge.alt_data_fetcher import LiveAltDataFetcher

        fetcher = LiveAltDataFetcher()
        status = fetcher.get_status()
        assert "fred_configured" in status
        assert "fred_series" in status
        assert "yfinance_tickers" in status

    def test_clear_cache(self):
        from src.knowledge.alt_data_fetcher import LiveAltDataFetcher

        fetcher = LiveAltDataFetcher()
        fetcher._set_cached("test", 1.0)
        fetcher.clear_cache()
        assert fetcher._get_cached("test") is None


# ═════════════════════════════════════════════════════════════════
# SDK Client Tests
# ═════════════════════════════════════════════════════════════════


class TestSDKClient:
    """Tests for the Python SDK client."""

    def test_init(self):
        from sdk.client import AMRCAISClient

        client = AMRCAISClient("http://localhost:8000", api_key="test-key")
        assert client._base_url == "http://localhost:8000"
        assert repr(client) == "AMRCAISClient(base_url='http://localhost:8000')"

    def test_init_strips_trailing_slash(self):
        from sdk.client import AMRCAISClient

        client = AMRCAISClient("http://localhost:8000/")
        assert client._base_url == "http://localhost:8000"

    def test_context_manager(self):
        from sdk.client import AMRCAISClient

        with AMRCAISClient("http://localhost:8000") as client:
            assert client._base_url == "http://localhost:8000"

    def test_amrcais_error(self):
        from sdk.client import AMRCAISError

        err = AMRCAISError("test error")
        assert str(err) == "test error"
