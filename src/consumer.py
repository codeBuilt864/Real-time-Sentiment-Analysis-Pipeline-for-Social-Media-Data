"""
Kafka consumer — reads from `raw-social-posts`, scores sentiment,
writes enriched records to PostgreSQL and re-publishes to `enriched-sentiment`.

Run:
    python -m src.consumer
"""

from __future__ import annotations

import json
import signal
import sys
from datetime import datetime

from kafka import KafkaConsumer
from kafka import KafkaProducer
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.database import SentimentScore, get_db, init_db
from src.sentiment import SentimentService

settings = get_settings()


# ── Build Kafka clients ───────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def build_consumer() -> KafkaConsumer:
    consumer = KafkaConsumer(
        settings.kafka_topic_raw,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=-1,
    )
    logger.info(f"Consumer connected → topic='{settings.kafka_topic_raw}'")
    return consumer


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


# ── Processing ────────────────────────────────────────────────────────────────

def process_message(
    payload: dict,
    sentiment_svc: SentimentService,
) -> dict:
    """Score a post and return the enriched payload dict."""
    clean_text = payload.get("clean_text", "")
    result     = sentiment_svc.score(clean_text)

    return {
        **payload,
        "label":      result.label.value,
        "compound":   result.compound,
        "positive":   result.positive,
        "negative":   result.negative,
        "neutral":    result.neutral,
        "model_used": result.model_used,
        "scored_at":  datetime.utcnow().isoformat(),
    }


def persist(enriched: dict) -> None:
    """Write an enriched post to PostgreSQL."""
    with get_db() as db:
        created_at = (
            datetime.fromisoformat(enriched["created_at"])
            if enriched.get("created_at")
            else datetime.utcnow()
        )
        record = SentimentScore(
            post_id    = enriched["post_id"],
            source     = enriched["source"],
            created_at = created_at,
            raw_text   = enriched.get("raw_text", ""),
            clean_text = enriched.get("clean_text", ""),
            label      = enriched["label"],
            compound   = enriched["compound"],
            positive   = enriched["positive"],
            negative   = enriched["negative"],
            neutral    = enriched["neutral"],
            model_used = enriched["model_used"],
            author     = enriched.get("author"),
            subreddit  = enriched.get("subreddit"),
        )
        db.merge(record)   # upsert — safe on reprocessing


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_consumer(backend: str = "vader") -> None:
    init_db()

    consumer       = build_consumer()
    producer       = build_producer()
    sentiment_svc  = SentimentService(backend=backend)
    running        = True

    def shutdown(*_):
        nonlocal running
        logger.info("Shutting down consumer…")
        running = False
        consumer.close()
        producer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Consumer loop started (backend='{backend}')")

    for message in consumer:
        if not running:
            break
        try:
            payload  = message.value
            enriched = process_message(payload, sentiment_svc)

            # Write to PostgreSQL
            persist(enriched)

            # Re-publish to enriched topic
            producer.send(settings.kafka_topic_enriched, value=enriched)

            logger.info(
                f"post_id={enriched['post_id']} | "
                f"label={enriched['label']} | "
                f"compound={enriched['compound']:.3f}"
            )

        except Exception as exc:
            logger.error(f"Error processing message: {exc}")


if __name__ == "__main__":
    run_consumer()
