"""
FastAPI sentiment API.

Endpoints:
    GET  /health                → liveness check
    POST /predict               → score a single text
    POST /predict/batch         → score multiple texts
    GET  /stats                 → sentiment counts from DB (last N hours)
    GET  /metrics               → Prometheus metrics
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from sqlalchemy import func, text
from starlette.responses import Response

from src.config import get_settings
from src.database import SentimentScore, SessionLocal, init_db
from src.preprocessing import preprocess_text
from src.sentiment import SentimentResult, SentimentService

settings = get_settings()

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("api_requests_total",   "Total API requests",  ["endpoint"])
SENTIMENT_COUNT = Counter("sentiment_labels_total","Sentiment label counts", ["label"])
LATENCY         = Histogram("api_request_duration_seconds", "Request latency", ["endpoint"])

# ── Global sentiment service (loaded once at startup) ─────────────────────────
_sentiment_svc: SentimentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _sentiment_svc
    logger.info("Starting up — initialising DB and sentiment service…")
    init_db()
    _sentiment_svc = SentimentService(backend="vader")
    logger.info("API ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Real-time Sentiment Analysis API",
    description="Scores social-media text with VADER (upgradeable to BERT).",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to score")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)


class PredictResponse(BaseModel):
    original_text: str
    clean_text:    str
    label:         str
    compound:      float
    positive:      float
    negative:      float
    neutral:       float
    model_used:    str
    latency_ms:    float


class StatsResponse(BaseModel):
    window_hours: int
    total:        int
    positive:     int
    negative:     int
    neutral:      int
    avg_compound: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictResponse, tags=["sentiment"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score a single piece of text and return full sentiment detail."""
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    t0 = time.perf_counter()

    clean   = preprocess_text(req.text)
    result  = _sentiment_svc.score(clean)
    latency = (time.perf_counter() - t0) * 1000

    SENTIMENT_COUNT.labels(label=result.label.value).inc()

    return PredictResponse(
        original_text = req.text,
        clean_text    = clean,
        label         = result.label.value,
        compound      = result.compound,
        positive      = result.positive,
        negative      = result.negative,
        neutral       = result.neutral,
        model_used    = result.model_used,
        latency_ms    = round(latency, 2),
    )


@app.post("/predict/batch", tags=["sentiment"])
def predict_batch(req: BatchPredictRequest) -> list[PredictResponse]:
    """Score up to 100 texts in a single request."""
    REQUEST_COUNT.labels(endpoint="/predict/batch").inc()

    responses = []
    for text in req.texts:
        clean  = preprocess_text(text)
        result = _sentiment_svc.score(clean)
        SENTIMENT_COUNT.labels(label=result.label.value).inc()
        responses.append(
            PredictResponse(
                original_text = text,
                clean_text    = clean,
                label         = result.label.value,
                compound      = result.compound,
                positive      = result.positive,
                negative      = result.negative,
                neutral       = result.neutral,
                model_used    = result.model_used,
                latency_ms    = 0.0,
            )
        )
    return responses


@app.get("/stats", response_model=StatsResponse, tags=["analytics"])
def stats(hours: int = Query(default=24, ge=1, le=168)) -> StatsResponse:
    """Return sentiment counts and average compound score for the last N hours."""
    REQUEST_COUNT.labels(endpoint="/stats").inc()
    since = datetime.utcnow() - timedelta(hours=hours)

    db = SessionLocal()
    try:
        rows = (
            db.query(
                SentimentScore.label,
                func.count(SentimentScore.post_id).label("cnt"),
                func.avg(SentimentScore.compound).label("avg_compound"),
            )
            .filter(SentimentScore.created_at >= since)
            .group_by(SentimentScore.label)
            .all()
        )
    finally:
        db.close()

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_compound = 0.0
    total          = 0

    for row in rows:
        counts[row.label] = row.cnt
        total             += row.cnt
        total_compound    += (row.avg_compound or 0) * row.cnt

    return StatsResponse(
        window_hours = hours,
        total        = total,
        positive     = counts["positive"],
        negative     = counts["negative"],
        neutral      = counts["neutral"],
        avg_compound = round(total_compound / total, 4) if total else 0.0,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Prometheus metrics endpoint — scraped by prometheus.yml."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
