-- Initialise TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Main sentiment scores table
CREATE TABLE IF NOT EXISTS sentiment_scores (
    post_id     VARCHAR(64)   PRIMARY KEY,
    source      VARCHAR(16)   NOT NULL,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    raw_text    TEXT          NOT NULL,
    clean_text  TEXT          NOT NULL,
    label       VARCHAR(16)   NOT NULL,
    compound    FLOAT         NOT NULL,
    positive    FLOAT         NOT NULL,
    negative    FLOAT         NOT NULL,
    neutral     FLOAT         NOT NULL,
    model_used  VARCHAR(16)   NOT NULL DEFAULT 'vader',
    author      VARCHAR(128),
    subreddit   VARCHAR(64)
);

-- Convert to TimescaleDB hypertable (time-series optimisation)
SELECT create_hypertable('sentiment_scores', 'created_at', if_not_exists => TRUE);

-- Indexes for dashboard queries
CREATE INDEX IF NOT EXISTS ix_source_ts      ON sentiment_scores (source, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_label_ts       ON sentiment_scores (label, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_compound       ON sentiment_scores (compound);
