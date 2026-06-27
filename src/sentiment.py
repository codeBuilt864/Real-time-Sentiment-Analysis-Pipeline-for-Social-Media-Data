"""
Sentiment scoring service.

Strategy: start with VADER (fast, no GPU needed, great for social media slang).
          The `SentimentService` class exposes a unified `.score()` interface
          so the backend (VADER vs fine-tuned BERT) can be swapped without
          touching any other module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config import get_settings


# ── Result model ─────────────────────────────────────────────────────────────

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


@dataclass
class SentimentResult:
    label: SentimentLabel
    positive: float
    negative: float
    neutral: float
    compound: float          # VADER compound score  -1.0 … +1.0
    model_used: str


# ── VADER backend ─────────────────────────────────────────────────────────────

class VADERBackend:
    """Rule-based sentiment using VADER — optimised for social-media text."""

    # Compound thresholds recommended by the VADER authors
    POSITIVE_THRESHOLD =  0.05
    NEGATIVE_THRESHOLD = -0.05

    def __init__(self) -> None:
        self._analyser = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyser initialised")

    def score(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                compound=0.0,
                model_used="vader",
            )

        scores = self._analyser.polarity_scores(text)
        compound = scores["compound"]

        if compound >= self.POSITIVE_THRESHOLD:
            label = SentimentLabel.POSITIVE
        elif compound <= self.NEGATIVE_THRESHOLD:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentResult(
            label=label,
            positive=round(scores["pos"], 4),
            negative=round(scores["neg"], 4),
            neutral=round(scores["neu"], 4),
            compound=round(compound, 4),
            model_used="vader",
        )


# ── BERT backend (Phase 3 upgrade) ───────────────────────────────────────────

class BERTBackend:
    """
    Fine-tuned DistilBERT sentiment classifier.

    Loaded lazily — only imported when explicitly requested so the service
    starts fast even without a GPU.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    LABEL_MAP  = {"POSITIVE": SentimentLabel.POSITIVE,
                  "NEGATIVE": SentimentLabel.NEGATIVE}

    def __init__(self) -> None:
        from transformers import pipeline  # lazy import
        logger.info(f"Loading BERT model: {self.MODEL_NAME}")
        self._pipe = pipeline(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            truncation=True,
            max_length=512,
        )
        logger.info("BERT model loaded")

    def score(self, text: str) -> SentimentResult:
        if not text:
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                compound=0.0,
                model_used="bert",
            )

        result = self._pipe(text)[0]
        label = self.LABEL_MAP.get(result["label"], SentimentLabel.NEUTRAL)
        conf  = round(result["score"], 4)

        return SentimentResult(
            label=label,
            positive=conf if label == SentimentLabel.POSITIVE else round(1 - conf, 4),
            negative=conf if label == SentimentLabel.NEGATIVE else round(1 - conf, 4),
            neutral=0.0,
            compound=conf if label == SentimentLabel.POSITIVE else -conf,
            model_used="bert",
        )


# ── Unified service ───────────────────────────────────────────────────────────

class SentimentService:
    """
    Unified interface — callers never touch the backend directly.

    Usage:
        svc = SentimentService()            # defaults to VADER
        svc = SentimentService("bert")      # upgrades to BERT

        result = svc.score("I love this!")
        print(result.label, result.compound)
    """

    def __init__(self, backend: str = "vader") -> None:
        self._backend_name = backend.lower()
        if self._backend_name == "bert":
            self._backend = BERTBackend()
        else:
            self._backend = VADERBackend()
        logger.info(f"SentimentService initialised with backend='{self._backend_name}'")

    def score(self, text: str) -> SentimentResult:
        """Score a single piece of text and return a SentimentResult."""
        return self._backend.score(text)

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Score a list of texts. Falls back to loop for VADER."""
        return [self._backend.score(t) for t in texts]

    @property
    def backend_name(self) -> str:
        return self._backend_name
