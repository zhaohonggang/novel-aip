-- Novel-AIP: Database Initialization Matrix
-- Target: PostgreSQL 16 + pgvector

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS novel_catalog (
    id                   SERIAL PRIMARY KEY,
    title                VARCHAR(512) NOT NULL,
    author               VARCHAR(256),
    main_characters      TEXT[],
    key_locations        TEXT[],
    genres               TEXT[],
    tropes               TEXT[],
    one_liner_summary    TEXT,
    summary_embedding    VECTOR(768),
    raw_json_payload     JSONB,
    processed_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- GIN index for swift string containment tests on array columns
CREATE INDEX IF NOT EXISTS idx_novel_genres_gin
    ON novel_catalog USING GIN (genres);
CREATE INDEX IF NOT EXISTS idx_novel_tropes_gin
    ON novel_catalog USING GIN (tropes);
CREATE INDEX IF NOT EXISTS idx_novel_main_characters_gin
    ON novel_catalog USING GIN (main_characters);
CREATE INDEX IF NOT EXISTS idx_novel_key_locations_gin
    ON novel_catalog USING GIN (key_locations);

-- HNSW graph proximity index (cosine distance) over dense summary vectors
CREATE INDEX IF NOT EXISTS idx_novel_hnsw
    ON novel_catalog USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMIT;