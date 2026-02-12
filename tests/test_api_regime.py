"""
Tests for AMRCAIS API — regime detection endpoints.

Covers:
    GET /api/regime/current       — Current regime classification
    GET /api/regime/history       — Regime history
    GET /api/regime/classifiers   — Individual classifier votes
    GET /api/regime/transitions   — Transition matrix
    GET /api/regime/disagreement  — Disagreement time series
"""

import pytest


# ─── Current Regime ──────────────────────────────────────────────


class TestCurrentRegime:
    """GET /api/regime/current"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/regime/current")
        assert resp.status_code == 200

    def test_has_regime_id(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "regime" in data
        assert 1 <= data["regime"] <= 4

    def test_has_regime_name(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "regime_name" in data
        assert isinstance(data["regime_name"], str)

    def test_has_confidence(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert 0 <= data["confidence"] <= 1

    def test_has_disagreement(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert 0 <= data["disagreement"] <= 1

    def test_has_classifier_votes(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "classifier_votes" in data
        assert isinstance(data["classifier_votes"], dict)

    def test_has_probabilities(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "probabilities" in data
        assert isinstance(data["probabilities"], dict)

    def test_has_transition_warning(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "transition_warning" in data
        assert isinstance(data["transition_warning"], bool)

    def test_has_timestamp(self, api_client):
        data = api_client.get("/api/regime/current").json()
        assert "timestamp" in data


# ─── Regime History ──────────────────────────────────────────────


class TestRegimeHistory:
    """GET /api/regime/history"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/regime/history")
        assert resp.status_code == 200

    def test_has_history_list(self, api_client):
        data = api_client.get("/api/regime/history").json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_has_total_points(self, api_client):
        data = api_client.get("/api/regime/history").json()
        assert "total_points" in data
        assert data["total_points"] == len(data["history"])

    def test_has_date_range(self, api_client):
        data = api_client.get("/api/regime/history").json()
        assert "start_date" in data
        assert "end_date" in data

    def test_custom_date_range(self, api_client):
        resp = api_client.get(
            "/api/regime/history",
            params={"start": "2023-01-01", "end": "2023-12-31"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_date"] == "2023-01-01"
        assert data["end_date"] == "2023-12-31"

    def test_empty_history_when_no_data(self, api_client):
        """With a fresh mock ensemble (empty prediction history), expect 0 points."""
        data = api_client.get("/api/regime/history").json()
        assert data["total_points"] == 0


# ─── Classifier Votes ────────────────────────────────────────────


class TestClassifierVotes:
    """GET /api/regime/classifiers"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/regime/classifiers")
        assert resp.status_code == 200

    def test_has_votes_list(self, api_client):
        data = api_client.get("/api/regime/classifiers").json()
        assert "votes" in data
        assert isinstance(data["votes"], list)

    def test_vote_entry_schema(self, api_client):
        data = api_client.get("/api/regime/classifiers").json()
        for vote in data["votes"]:
            assert "classifier" in vote
            assert "regime" in vote
            assert "confidence" in vote
            assert "weight" in vote

    def test_has_ensemble_regime(self, api_client):
        data = api_client.get("/api/regime/classifiers").json()
        assert "ensemble_regime" in data
        assert 1 <= data["ensemble_regime"] <= 4

    def test_has_weights_dict(self, api_client):
        data = api_client.get("/api/regime/classifiers").json()
        assert "weights" in data
        assert isinstance(data["weights"], dict)


# ─── Transition Matrix ───────────────────────────────────────────


class TestTransitionMatrix:
    """GET /api/regime/transitions"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/regime/transitions")
        assert resp.status_code == 200

    def test_matrix_is_4x4(self, api_client):
        data = api_client.get("/api/regime/transitions").json()
        matrix = data["matrix"]
        assert len(matrix) == 4
        for row in matrix:
            assert len(row) == 4

    def test_has_regime_names(self, api_client):
        data = api_client.get("/api/regime/transitions").json()
        assert len(data["regime_names"]) == 4

    def test_has_total_transitions(self, api_client):
        data = api_client.get("/api/regime/transitions").json()
        assert "total_transitions" in data
        assert isinstance(data["total_transitions"], int)


# ─── Disagreement Series ─────────────────────────────────────────


class TestDisagreementSeries:
    """GET /api/regime/disagreement"""

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/regime/disagreement")
        assert resp.status_code == 200

    def test_has_series_list(self, api_client):
        data = api_client.get("/api/regime/disagreement").json()
        assert "series" in data
        assert isinstance(data["series"], list)

    def test_has_avg_disagreement(self, api_client):
        data = api_client.get("/api/regime/disagreement").json()
        assert "avg_disagreement" in data

    def test_has_max_disagreement(self, api_client):
        data = api_client.get("/api/regime/disagreement").json()
        assert "max_disagreement" in data

    def test_has_threshold(self, api_client):
        data = api_client.get("/api/regime/disagreement").json()
        assert data["threshold"] == 0.6

    def test_custom_date_range(self, api_client):
        resp = api_client.get(
            "/api/regime/disagreement",
            params={"start": "2023-06-01", "end": "2023-12-31"},
        )
        assert resp.status_code == 200
