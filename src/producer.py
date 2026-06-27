"""
Kafka producer — ingests raw posts from Twitter (v2) or Reddit
and publishes them to the `raw-social-posts` topic.

Run:
    python -m src.producer --source twitter
    python -m src.producer --source reddit
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import Iterator

import click
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.preprocessing import post_from_reddit, post_from_twitter

settings = get_settings()


# ── Producer factory ──────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def build_producer() -> KafkaProducer:
    """Create a KafkaProducer with JSON serialisation. Retries on connection failure."""
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
        max_in_flight_requests_per_connection=1,
    )
    logger.info(f"Kafka producer connected to {settings.kafka_bootstrap_servers}")
    return producer


# ── Twitter stream ────────────────────────────────────────────────────────────

def twitter_stream(query: str = "AI OR machine learning lang:en") -> Iterator[dict]:
    """
    Yield raw tweet dicts from Twitter API v2 filtered stream.
    Falls back to mock data if no bearer token is set (useful for local dev).
    """
    if not settings.twitter_bearer_token:
        logger.warning("No TWITTER_BEARER_TOKEN — yielding mock tweets for dev")
        yield from _mock_tweets()
        return

    import httpx  # lazy import to keep startup fast without token

    headers = {"Authorization": f"Bearer {settings.twitter_bearer_token}"}
    stream_url = (
        "https://api.twitter.com/2/tweets/search/stream"
        "?tweet.fields=id,text,author_id,created_at,lang"
    )

    with httpx.stream("GET", stream_url, headers=headers, timeout=None) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                try:
                    yield json.loads(line).get("data", {})
                except json.JSONDecodeError:
                    continue


def _mock_tweets() -> Iterator[dict]:
    """Generate fake tweets for local development without API credentials."""
    samples = [
        {"id": "1", "text": "I absolutely love the new AI features! 🚀",
         "author_id": "u1", "created_at": "2026-06-01T10:00:00.000Z", "lang": "en"},
        {"id": "2", "text": "This is terrible, nothing works anymore #frustrated",
         "author_id": "u2", "created_at": "2026-06-01T10:01:00.000Z", "lang": "en"},
        {"id": "3", "text": "Interesting research paper on NLP sentiment analysis",
         "author_id": "u3", "created_at": "2026-06-01T10:02:00.000Z", "lang": "en"},
    ]
    while True:
        for tweet in samples:
            yield tweet
            time.sleep(2)


# ── Reddit stream ─────────────────────────────────────────────────────────────

def reddit_stream(subreddit: str = "technology") -> Iterator[dict]:
    """Poll Reddit for new submissions every 30 seconds."""
    import praw  # pip install praw (add to requirements if using Reddit)

    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )
    sub = reddit.subreddit(subreddit)
    for submission in sub.stream.submissions(skip_existing=True):
        yield {
            "id":          submission.id,
            "title":       submission.title,
            "selftext":    submission.selftext,
            "author":      str(submission.author),
            "subreddit":   submission.subreddit.display_name,
            "created_utc": submission.created_utc,
        }


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_producer(source: str) -> None:
    producer = build_producer()
    topic    = settings.kafka_topic_raw
    running  = True

    def shutdown(*_):
        nonlocal running
        logger.info("Shutting down producer…")
        running = False
        producer.flush()
        producer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Producer started → topic='{topic}', source='{source}'")

    stream = twitter_stream() if source == "twitter" else reddit_stream()

    for raw in stream:
        if not running:
            break
        try:
            post = (
                post_from_twitter(raw)
                if source == "twitter"
                else post_from_reddit(raw)
            )
            payload = {
                "post_id":    post.post_id,
                "source":     post.source,
                "raw_text":   post.raw_text,
                "clean_text": post.clean_text,
                "author":     post.author,
                "subreddit":  post.subreddit,
                "created_at": post.created_at.isoformat() if post.created_at else None,
            }
            producer.send(topic, key=post.post_id, value=payload)
            logger.debug(f"Published post_id={post.post_id}")
        except Exception as exc:
            logger.error(f"Failed to publish post: {exc}")


@click.command()
@click.option("--source", default="twitter", type=click.Choice(["twitter", "reddit"]))
def main(source: str) -> None:
    run_producer(source)


if __name__ == "__main__":
    main()
