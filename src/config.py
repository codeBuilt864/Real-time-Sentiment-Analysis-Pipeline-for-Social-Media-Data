"""Centralised configuration — loaded once from environment / .env file."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Application
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_port: int = Field(default=8000, alias="API_PORT")
    model_path: str = Field(default="models/sentiment_model.pkl", alias="MODEL_PATH")

    # Kafka
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS"
    )
    kafka_topic_raw: str = Field(
        default="raw-social-posts", alias="KAFKA_TOPIC_RAW"
    )
    kafka_topic_enriched: str = Field(
        default="enriched-sentiment", alias="KAFKA_TOPIC_ENRICHED"
    )
    kafka_consumer_group: str = Field(
        default="sentiment-consumer-group", alias="KAFKA_CONSUMER_GROUP"
    )

    # PostgreSQL
    postgres_url: str = Field(
        default="postgresql://sentiment_user:sentiment_pass@localhost:5432/sentimentdb",
        alias="POSTGRES_URL",
    )

    # Twitter
    twitter_bearer_token: str = Field(default="", alias="TWITTER_BEARER_TOKEN")

    # Reddit
    reddit_client_id: str = Field(default="", alias="REDDIT_CLIENT_ID")
    reddit_client_secret: str = Field(default="", alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(
        default="SentimentBot/1.0", alias="REDDIT_USER_AGENT"
    )

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_bucket: str = Field(default="sentiment-models", alias="MINIO_BUCKET")

    class Config:
        populate_by_name = True
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
