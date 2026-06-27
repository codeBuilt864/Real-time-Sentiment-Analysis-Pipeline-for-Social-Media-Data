"""Tests for src/sentiment.py."""

import pytest
from src.sentiment import SentimentService, SentimentLabel


@pytest.fixture(scope="module")
def svc():
    return SentimentService(backend="vader")


class TestSentimentService:
    def test_positive_text(self, svc):
        result = svc.score("I absolutely love this, it is amazing and wonderful!")
        assert result.label == SentimentLabel.POSITIVE
        assert result.compound > 0

    def test_negative_text(self, svc):
        result = svc.score("This is terrible and I hate it completely.")
        assert result.label == SentimentLabel.NEGATIVE
        assert result.compound < 0

    def test_neutral_text(self, svc):
        result = svc.score("The weather today is partly cloudy.")
        assert result.label == SentimentLabel.NEUTRAL

    def test_empty_text_returns_neutral(self, svc):
        result = svc.score("")
        assert result.label == SentimentLabel.NEUTRAL
        assert result.compound == 0.0

    def test_scores_sum_to_one(self, svc):
        result = svc.score("I love building things.")
        total = round(result.positive + result.negative + result.neutral, 1)
        assert total == 1.0

    def test_batch_returns_correct_length(self, svc):
        texts = ["great!", "terrible!", "ok"]
        results = svc.score_batch(texts)
        assert len(results) == len(texts)

    def test_backend_name(self, svc):
        assert svc.backend_name == "vader"
