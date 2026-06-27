"""
Database layer — SQLAlchemy models + session factory.

Uses TimescaleDB hypertable for `sentiment_scores` to enable fast
time-series queries (last hour, last 24h, rolling averages).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import get_settings


# ── Engine setup ──────────────────────────────────────────────────────────────

settings = get_settings()

engine = create_engine(
    settings.postgres_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=(settings.app_env == "development"),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── ORM models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class SentimentScore(Base):
    """One row per scored social-media post."""

    __tablename__ = "sentiment_scores"

    post_id    = Column(String(64),  primary_key=True)
    source     = Column(String(16),  nullable=False)          # twitter | reddit
    created_at = Column(DateTime,    nullable=False, default=datetime.utcnow, index=True)
    raw_text   = Column(Text,        nullable=False)
    clean_text = Column(Text,        nullable=False)
    label      = Column(String(16),  nullable=False)          # positive | negative | neutral
    compound   = Column(Float,       nullable=False)
    positive   = Column(Float,       nullable=False)
    negative   = Column(Float,       nullable=False)
    neutral    = Column(Float,       nullable=False)
    model_used = Column(String(16),  nullable=False, default="vader")
    author     = Column(String(128), nullable=True)
    subreddit  = Column(String(64),  nullable=True)

    __table_args__ = (
        Index("ix_sentiment_source_ts", "source", "created_at"),
    )


# ── Session helpers ───────────────────────────────────────────────────────────

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Yield a database session, rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables and convert sentiment_scores to a TimescaleDB hypertable."""
    logger.info("Initialising database schema…")
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        try:
            conn.execute(
                text(
                    "SELECT create_hypertable('sentiment_scores', 'created_at', "
                    "if_not_exists => TRUE);"
                )
            )
            conn.commit()
            logger.info("TimescaleDB hypertable configured on sentiment_scores.created_at")
        except Exception as exc:
            logger.warning(f"Hypertable creation skipped (non-TimescaleDB?): {exc}")

    logger.info("Database ready")
