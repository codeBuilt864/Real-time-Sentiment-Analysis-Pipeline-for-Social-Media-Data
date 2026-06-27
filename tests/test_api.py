"""Tests for src/api.py — uses TestClient, no real Kafka or DB needed."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api import app
from src.sentiment import SentimentLabel, SentimentResult


MOCK_RESULT = SentimentResult(
    label=SentimentLabel.POSITIVE,
    positive=0.8,
    negative=0.0,
    neutral=0.2,
    compound=0.75,
    model_used="vader",
)


@pytest.fixture(scope="module")
def client():
    with patch("src.api.init_db"), \
         patch("src.api.SentimentService") as MockSvc:
        MockSvc.return_value.score.return_value = MOCK_RESULT
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_valid_text_returns_200(self, client):
        resp = client.post("/predict", json={"text": "I love this product!"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/predict", json={"text": "Great news today!"})
        data = resp.json()
        for field in ("label", "compound", "positive", "negative", "neutral", "model_used"):
            assert field in data

    def test_empty_text_returns_422(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch_returns_list(self, client):
        resp = client.post("/predict/batch", json={"texts": ["good", "bad", "ok"]})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 3
