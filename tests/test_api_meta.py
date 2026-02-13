"""
Tests for AMRCAIS API — meta-learning endpoints.

Covers:
    GET /api/meta/performance         — Classifier performance metrics
    GET /api/meta/weights             — Current ensemble weights
    GET /api/meta/weights/history     — Weight evolution over time
    GET /api/meta/recalibrations      — Recalibration event log
    GET /api/meta/health              — System health summary
    GET /api/meta/accuracy            — Rolling classifier accuracy
    GET /api/meta/disagreement        — Disagreement time-series
"""

import pytest
from unittest.mock import MagicMock, patch


# ─── Performance ─────────────────────────────────────────────────


class TestPerformance:
    """GET /api/meta/performance"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/performance")
        assert resp.status_code == 200

    def test_has_stability_score(self, api_client):
        data = api_client.get("/api/meta/performance").json()
        assert "stability_score" in data
        assert isinstance(data["stability_score"], float)

    def test_has_stability_rating(self, api_client):
        data = api_client.get("/api/meta/performance").json()
        assert "stability_rating" in data
        assert data["stability_rating"] in ("stable", "moderate", "unstable")

    def test_has_transition_count(self, api_client):
        data = api_client.get("/api/meta/performance").json()
        assert "transition_count" in data
        assert isinstance(data["transition_count"], int)

    def test_has_disagreement_stats(self, api_client):
        data = api_client.get("/api/meta/performance").json()
        assert "avg_disagreement" in data
        assert "high_disagreement_days" in data

    def test_has_regime_distribution(self, api_client):
        data = api_client.get("/api/meta/performance").json()
        assert "regime_distribution" in data
        assert isinstance(data["regime_distribution"], dict)

    def test_no_meta_learner_returns_503(self, api_client, mock_system):
        """When meta_learner is None, endpoint returns 503."""
        mock_system.meta_learner = None
        resp = api_client.get("/api/meta/performance")
        assert resp.status_code == 503


# ─── Weights ─────────────────────────────────────────────────────


class TestWeights:
    """GET /api/meta/weights"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/weights")
        assert resp.status_code == 200

    def test_has_weights_dict(self, api_client):
        data = api_client.get("/api/meta/weights").json()
        assert "weights" in data
        assert isinstance(data["weights"], dict)

    def test_weights_have_4_classifiers(self, api_client):
        data = api_client.get("/api/meta/weights").json()
        assert len(data["weights"]) == 4

    def test_is_adaptive_flag(self, api_client):
        data = api_client.get("/api/meta/weights").json()
        assert "is_adaptive" in data
        assert isinstance(data["is_adaptive"], bool)

    def test_weights_sum_approximately_1(self, api_client):
        data = api_client.get("/api/meta/weights").json()
        total = sum(data["weights"].values())
        assert abs(total - 1.0) < 0.05

    def test_fallback_when_no_meta_learner(self, api_client, mock_system):
        """When meta_learner is None, should fall back to ensemble weights."""
        mock_system.meta_learner = None
        resp = api_client.get("/api/meta/weights")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_adaptive"] is False


# ─── Weight History ──────────────────────────────────────────────


class TestWeightHistory:
    """GET /api/meta/weights/history"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/weights/history")
        assert resp.status_code == 200

    def test_has_history_list(self, api_client):
        data = api_client.get("/api/meta/weights/history").json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_history_entry_schema(self, api_client):
        data = api_client.get("/api/meta/weights/history").json()
        if data["history"]:
            entry = data["history"][0]
            assert "date" in entry
            assert "weights" in entry
            assert isinstance(entry["weights"], dict)


# ─── Recalibrations ──────────────────────────────────────────────


class TestRecalibrations:
    """GET /api/meta/recalibrations"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/recalibrations")
        assert resp.status_code == 200

    def test_has_events_list(self, api_client):
        data = api_client.get("/api/meta/recalibrations").json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_has_total_recalibrations(self, api_client):
        data = api_client.get("/api/meta/recalibrations").json()
        assert "total_recalibrations" in data
        assert isinstance(data["total_recalibrations"], int)

    def test_has_last_recalibration(self, api_client):
        data = api_client.get("/api/meta/recalibrations").json()
        assert "last_recalibration" in data
        # Can be None

    def test_recalibration_needed_shows_event(self, api_client, mock_system):
        """When recalibration is needed, an event should be present."""
        recal = MagicMock()
        recal.should_recalibrate = True
        recal.reasons = ["High disagreement for 15 days"]
        recal.severity = 0.8
        recal.urgency_level = "high"
        recal.recommendations = ["Re-fit HMM classifier"]
        mock_system.meta_learner.check_recalibration_needed.return_value = recal

        data = api_client.get("/api/meta/recalibrations").json()
        assert len(data["events"]) >= 1
        event = data["events"][0]
        assert "trigger_reason" in event
        assert event["urgency"] == "high"


# ─── System Health ───────────────────────────────────────────────


class TestSystemHealth:
    """GET /api/meta/health"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/health")
        assert resp.status_code == 200

    def test_has_system_status(self, api_client):
        data = api_client.get("/api/meta/health").json()
        assert "system_status" in data
        assert data["system_status"] == "healthy"

    def test_has_recalibration_flags(self, api_client):
        data = api_client.get("/api/meta/health").json()
        assert "needs_recalibration" in data
        assert isinstance(data["needs_recalibration"], bool)
        assert "urgency" in data
        assert "severity" in data

    def test_has_performance_data(self, api_client):
        data = api_client.get("/api/meta/health").json()
        assert "performance_30d" in data
        assert "performance_7d" in data

    def test_has_adaptive_weights(self, api_client):
        data = api_client.get("/api/meta/health").json()
        assert "adaptive_weights" in data

    def test_degraded_when_no_meta_learner(self, api_client, mock_system):
        """When meta_learner is None, system_status should be 'degraded'."""
        mock_system.meta_learner = None
        data = api_client.get("/api/meta/health").json()
        assert data["system_status"] == "degraded"
        assert data["needs_recalibration"] is False


# ─── Classifier Accuracy ─────────────────────────────────────────


class TestClassifierAccuracy:
    """GET /api/meta/accuracy"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/accuracy")
        assert resp.status_code == 200

    def test_has_classifiers_list(self, api_client):
        data = api_client.get("/api/meta/accuracy").json()
        assert "classifiers" in data
        assert isinstance(data["classifiers"], list)
        assert len(data["classifiers"]) >= 4

    def test_has_series(self, api_client):
        data = api_client.get("/api/meta/accuracy").json()
        assert "series" in data
        assert isinstance(data["series"], list)

    def test_has_window(self, api_client):
        data = api_client.get("/api/meta/accuracy").json()
        assert "window" in data
        assert isinstance(data["window"], int)

    def test_custom_window(self, api_client):
        resp = api_client.get("/api/meta/accuracy", params={"window": 60})
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == 60

    def test_series_entry_schema(self, api_client):
        data = api_client.get("/api/meta/accuracy").json()
        if data["series"]:
            entry = data["series"][0]
            assert "date" in entry
            assert "accuracy" in entry
            assert "classifier" in entry

    def test_accuracy_values_in_range(self, api_client):
        data = api_client.get("/api/meta/accuracy").json()
        for entry in data["series"]:
            assert 0.0 <= entry["accuracy"] <= 1.0

    def test_fallback_when_no_meta_learner(self, api_client, mock_system):
        """Even without meta_learner, should return synthetic data."""
        mock_system.meta_learner = None
        resp = api_client.get("/api/meta/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["series"]) > 0


# ─── Disagreement Time-Series ────────────────────────────────────


class TestMetaDisagreement:
    """GET /api/meta/disagreement"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/meta/disagreement")
        assert resp.status_code == 200

    def test_has_series(self, api_client):
        data = api_client.get("/api/meta/disagreement").json()
        assert "series" in data
        assert isinstance(data["series"], list)

    def test_has_threshold(self, api_client):
        data = api_client.get("/api/meta/disagreement").json()
        assert "threshold" in data
        assert isinstance(data["threshold"], float)

    def test_series_entry_schema(self, api_client):
        data = api_client.get("/api/meta/disagreement").json()
        if data["series"]:
            entry = data["series"][0]
            assert "date" in entry
            assert "disagreement" in entry

    def test_disagreement_in_range(self, api_client):
        data = api_client.get("/api/meta/disagreement").json()
        for entry in data["series"]:
            assert 0.0 <= entry["disagreement"] <= 1.0

    def test_fallback_when_no_meta_learner(self, api_client, mock_system):
        """Synthetic fallback when no ensemble history available."""
        mock_system.meta_learner = None
        mock_system.regime_ensemble._disagreement_history = []
        resp = api_client.get("/api/meta/disagreement")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["series"]) > 0
