"""
Text preprocessing — one canonical function that the entire pipeline uses.

Design principle: all downstream components (VADER, BERT, storage) receive
text that has passed through `preprocess_text()`.  Nothing else calls
clean/normalise steps directly, so the behaviour is always reproducible.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SocialPost:
    """Represents a single social-media post after ingestion."""

    post_id: str
    source: str                  # "twitter" | "reddit"
    raw_text: str
    clean_text: str = ""
    author: Optional[str] = None
    subreddit: Optional[str] = None
    created_at: Optional[datetime] = None
    lang: str = "en"

    def __post_init__(self) -> None:
        if not self.clean_text:
            self.clean_text = preprocess_text(self.raw_text)


# ── Core preprocessing ────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    One canonical preprocessing function that everything uses.

    Steps applied in order:
        1. Unicode normalisation (NFKC)
        2. Remove URLs
        3. Remove @mentions
        4. Remove hashtag symbol (keep word)
        5. Remove RT prefix
        6. Collapse repeated characters  (loooool → lool)
        7. Strip non-ASCII except common punctuation
        8. Collapse extra whitespace
        9. Lower-case

    Args:
        text: Raw social-media post text.

    Returns:
        Cleaned, normalised string ready for sentiment scoring.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Unicode normalise
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 3. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 4. Remove # symbol but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Remove RT prefix
    text = re.sub(r"^RT\s*:?\s*", "", text, flags=re.IGNORECASE)

    # 6. Collapse repeated characters (3+ → 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 7. Keep letters, digits, common punctuation; strip the rest
    text = re.sub(r"[^\w\s.,!?'\";\-]", " ", text)

    # 8. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 9. Lower-case
    return text.lower()


def preprocess_batch(texts: list[str]) -> list[str]:
    """Apply `preprocess_text` to a list of strings."""
    return [preprocess_text(t) for t in texts]


# ── Post factory helpers ───────────────────────────────────────────────────────

def post_from_twitter(tweet: dict) -> SocialPost:
    """Build a SocialPost from a Twitter API v2 tweet object."""
    return SocialPost(
        post_id=tweet.get("id", ""),
        source="twitter",
        raw_text=tweet.get("text", ""),
        author=tweet.get("author_id"),
        created_at=_parse_twitter_date(tweet.get("created_at")),
        lang=tweet.get("lang", "en"),
    )


def post_from_reddit(submission: dict) -> SocialPost:
    """Build a SocialPost from a Reddit submission dict."""
    body = submission.get("selftext") or submission.get("title", "")
    return SocialPost(
        post_id=submission.get("id", ""),
        source="reddit",
        raw_text=body,
        author=submission.get("author"),
        subreddit=submission.get("subreddit"),
        created_at=_parse_reddit_date(submission.get("created_utc")),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_twitter_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return None


def _parse_reddit_date(timestamp: Optional[float]) -> Optional[datetime]:
    if timestamp is None:
        return None
    try:
        return datetime.utcfromtimestamp(float(timestamp))
    except (ValueError, OSError):
        return None
