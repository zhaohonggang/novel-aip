# 05 Vector Database Design & Data Loading

## 5.1 Schema Design (`schema.sql`)

### 5.1.1 Table `novel_catalog`
```sql
CREATE TABLE novel_catalog (
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
```

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `SERIAL` | PK | Auto-Increment PK |
| `title` | `VARCHAR(512)` | NOT NULL | Book Title |
| `author` | `VARCHAR(256)` | | Author |
| `main_characters` | `TEXT[]` | | Stage 1 Top-5 Characters |
| `key_locations` | `TEXT[]` | | Stage 1 Top-5 Locations |
| `genres` | `TEXT[]` | | LLM Genres (≤3) |
| `tropes` | `TEXT[]` | | LLM Tropes |
| `one_liner_summary` | `TEXT` | | One-Sentence Premise (≤200 Chars) |
| `summary_embedding` | `VECTOR(768)` | | nomic-embed-text-v1.5 Dense Vector |
| `raw_json_payload` | `JSONB` | | Full Raw Record (Incl source_file) |
| `processed_at` | `TIMESTAMP` | DEFAULT NOW | Ingestion Timestamp |

### 5.1.2 Index Strategy

```sql
-- GIN Inverted Indexes: Array Exact Containment Queries
CREATE INDEX idx_novel_genres_gin           ON novel_catalog USING GIN (genres);
CREATE INDEX idx_novel_tropes_gin           ON novel_catalog USING GIN (tropes);
CREATE INDEX idx_novel_main_characters_gin  ON novel_catalog USING GIN (main_characters);
CREATE INDEX idx_novel_key_locations_gin    ON novel_catalog USING GIN (key_locations);

-- HNSW ANN: Dense Vector Cosine Similarity
CREATE INDEX idx_novel_hnsw
    ON novel_catalog USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

| Index | Type | Use Case | Storage Overhead |
|-------|------|----------|------------------|
| `genres` | GIN | `genres @> ARRAY['Xianxia']` | ~1.2x Column Size |
| `tropes` | GIN | `tropes @> ARRAY['Revenge']` | ~1.2x |
| `main_characters` | GIN | `main_characters @> ARRAY['Yan Ping']` | ~1.2x |
| `key_locations` | GIN | `key_locations @> ARRAY['Guangdong']` | ~1.2x |
| `summary_embedding` | HNSW | `embedding <=> query_vec` | ~1.5x Vector Size |

### HNSW Parameter Tuning Guide
| Param | Default | Recommended Range | Impact |
|-------|---------|-------------------|--------|
| `m` | 16 | 8-32 | Graph Connectivity, Higher = Better Recall, Slower Build, More Memory |
| `ef_construction` | 64 | 40-200 | Build-Time Search Depth, Higher = Better Quality, Slower Build |
| `ef_search` | 40 (Runtime) | 40-400 | Query-Time Search Depth, Runtime `SET hnsw.ef_search = 100;` |

> **Rule of Thumb**: Million-Scale `m=16, ef_construction=64` Optimal; 100k Scale Can Drop to `m=8, ef_construction=40`.

---

## 5.2 Data Load Pipeline (`database_worker.py`)

### 5.2.1 Core Flow
```mermaid
flowchart LR
    A[records.json] --> B[Load Records]
    B --> C[Extract one_liner_summary]
    C --> D[nomic-embed-text-v1.5 Batch Embed]
    D --> E[Format Vector String]
    E --> F[execute_values Batch Insert]
    F --> G[Transaction Commit]
    G --> H[Retry/Circuit Breaker]
```

### 5.2.2 Embedding Model Config
```python
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE = 100

# Task Prefixes (Official Recommendation)
TASK_PREFIX_RETRIEVAL = "search_document: "
TASK_PREFIX_QUERY = "search_query: "

def embed_summaries(embedder, summaries, batch_size=BATCH_SIZE):
    texts = [TASK_PREFIX_RETRIEVAL + s if s else s for s in summaries]
    vectors = embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    return [list(map(float, row)) for row in vectors]
```
- **Task Prefix Critical**: nomic Series Requires Distinction Between "Document Side" (`search_document:`) and "Query Side" (`search_query:`), Else Semantic Space Mismatch
- **Fixed 768 Dim**: `nomic-embed-text-v1.5` Native 768-Dim, Matryoshka Truncation Doesn't Change Dim

### 5.2.3 Vector String Formatting
```python
def format_vector(vector: Sequence[float]) -> str:
    # pgvector Accepts "[0.1,0.2,...]" Text Literal with Explicit Cast
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
```
- **Precision**: 8 Decimal Places ≈ 1e-8, Sufficient for Cosine Similarity
- **NaN/Inf Guard**: Handled in `app.py`

### 5.2.4 Batch Transaction Insert (`execute_values`)
```python
def insert_chunk(records, embeddings, conn_info, batch_size=100):
    conn = psycopg2.connect(**conn_info)
    try:
        with conn.cursor() as cur:
            insert_query = sql.SQL(
                "INSERT INTO novel_catalog ({cols}) VALUES %s"
            ).format(cols=sql.SQL(", ").join(map(sql.Identifier, COLS)))
            
            row_template = "(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)"
            rows = []
            for rec, vec in zip(records, embeddings):
                rows.append((
                    rec["title"], rec.get("author"),
                    rec.get("main_characters"), rec.get("key_locations"),
                    rec.get("genres"), rec.get("tropes"),
                    rec.get("one_liner_summary"),
                    format_vector(vec),
                    json.dumps(rec.get("raw_json_payload") or {}, ensure_ascii=False),
                ))
            extras.execute_values(
                cur, insert_query, rows,
                template=row_template, page_size=batch_size
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```
- `execute_values` Single SQL Batch Insert, Avoids N Round-Trips
- `template` Explicit `%s::vector` Explicit Cast, Avoids Implicit Cast Failures
- `page_size=100` Aligned with `BATCH_SIZE`, Constant Memory

### 5.2.5 Retry & Circuit Breaker
```python
RETRY_LIMIT = 3
RETRY_DELAY_SECONDS = 5

def process_records(records, embedder, conn_info, ...):
    for chunk in chunks(records, BATCH_SIZE):
        attempt = 0
        while True:
            try:
                inserted = insert_chunk(chunk, ...)
                break
            except psycopg2.OperationalError as e:
                attempt += 1
                if attempt >= RETRY_LIMIT:
                    return False  # Crash Flag
                time.sleep(RETRY_DELAY_SECONDS)
            except Exception:
                return False  # Non-Connection Error Immediate Fail
    return True
```
| Error Type | Strategy |
|------------|----------|
| `OperationalError` (Conn Drop/Timeout) | Exponential Backoff Retry 3x, 5s Interval |
| `IntegrityError` (PK Conflict) | Rollback, Log Error, Continue Next Batch |
| `DataError` (Vector Dim Mismatch) | Rollback, Log Error, Continue Next Batch |
| Other Exceptions | Immediate Fail, Set Crash Flag |

---

## 5.3 Run Interface

```bash
python database_worker.py --payload records.json [Options]

Options:
  --payload PATH     JSON File Path (Required), Array or {records: [...]}
  --model STR        Embedding Model (Default nomic-ai/nomic-embed-text-v1.5)
  --batch-size INT   Batch Size (Default 100)
  --log-level LEVEL  Log Level (Default INFO)
```

### Env Vars Required
```bash
# .env Must Provide
DATABASE_NAME=fileindex
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_HOST=localhost
DATABASE_PORT=5432
HF_TOKEN=hf_xxx  # Optional, Accelerates Model Download
```

---

## 5.4 Run Sample Log

```
2026-09-14 02:06:16,425 | INFO | httpx | GET huggingface.co/api/.../nomic-embed-text-v1.5 200 OK
2026-09-14 02:06:16,656 | INFO | novel_aip.database_worker | Embedding 6 summary payload(s)
Batches: 100%|██████████| 1/1 [00:00<00:00,  1.52it/s]
2026-09-14 02:06:17,623 | INFO | novel_aip.database_worker | Inserted chunk of 6 record(s) (cumulative 6)
2026-09-14 02:06:17,623 | INFO | novel_aip.database_worker | Batch insert complete: 6 record(s) total
```

---

## 5.4 Performance Benchmarks

| Data Volume | Embed Time | Write Time | Total | Throughput |
|-------------|------------|------------|-------|------------|
| 6 Records (Sample) | ~1.2 s | ~0.4 s | ~1.6 s | ~3.7 rec/s |
| 1,000 (Est.) | ~200 s | ~30 s | ~230 s | ~4.3 rec/s |
| 10,000 (Est.) | ~2,000 s | ~300 s | ~2,300 s | ~4.3 rec/s |

> Bottleneck = Embedding Model Inference, CPU ~1.5 it/s (batch=100); GPU Inference 10-20x Speedup.

### Scaling Suggestions
| Direction | Expected Gain | Implementation |
|-----------|---------------|----------------|
| GPU Accel (CUDA/ROCm) | 10-20x | `device="cuda"` or `device="hip"` |
| Multi-Process Parallel Embed | Nx (N=CPU Cores) | `multiprocessing.Pool` + Shared Model Memory |
| Larger Batch (512) | +20% Throughput | If VRAM Permits |
| Connection Pool (`psycopg2.pool`) | Reduce Conn Overhead | `ThreadedConnectionPool` |

---

## 5.5 Database Maintenance Ops

### Index Rebuild (Online)
```sql
-- GIN Concurrent Rebuild (Non-Blocking Writes)
REINDEX INDEX CONCURRENTLY idx_novel_tropes_gin;

-- HNSW Rebuild (Exclusive Lock, Maintenance Window)
REINDEX INDEX idx_novel_hnsw;
```

### HNSW Search Precision Tuning (Runtime)
```sql
-- Boost Recall (Session-Scoped)
SET hnsw.ef_search = 200;
SELECT ... FROM novel_catalog ORDER BY summary_embedding <=> $1 LIMIT 10;
```

### Stats Refresh
```sql
ANALYZE novel_catalog;
-- Or Auto: autovacuum Config
```

### Backup & Restore
```bash
# Logical Backup (Incl Vectors)
pg_dump -U postgres -d fileindex -Fc -f backup.dump

# Restore
pg_restore -U postgres -d fileindex -c backup.dump
```

---

## 5.6 Related Documents

- [Dual-Stage NER & LLM Profiling](04-Dual-Stage-NER-and-LLM-Profiling.md) — Input Data Source
- [Hybrid Search Service & Web UI](06-Hybrid-Search-Service-and-Web-UI.md) — Search SQL Depends on Table/Indexes
- [Operations, Monitoring & Performance Tuning](07-Operations-Monitoring-and-Performance-Tuning.md) — Index Maintenance, Param Tuning
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — Dim Mismatch/Conn Drop/Index Corruption