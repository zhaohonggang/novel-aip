# 06 Hybrid Search Service & Web UI

## 6.1 Service Architecture Overview

```mermaid
flowchart TB
    subgraph Client["🌐 Client"]
        Browser["Browser\nhttp://localhost:8501"]
    end

    subgraph Service["🔧 Search Service (app.py)"]
        UI[Streamlit UI\nst.text_input + st.button]
        Embed[nomic-embed-text-v1.5\nsearch_query: Prefix]
        Search[Hybrid Search SQL\npgvector <=> + GIN Filter]
        Render[Card Rendering\nst.container(border=True)]
        Sidebar[Sidebar Real-time Metrics\nCOUNT / pg_size_pretty / perf_counter]
    end

    subgraph DB["💾 PostgreSQL + pgvector"]
        Table[(novel_catalog)]
        HNSW[idx_novel_hnsw]
        GINs[GIN ×4]
    end

    Browser --> UI
    UI --> Embed
    Embed --> Search
    Search --> DB
    DB --> Table
    Table --> HNSW
    Table --> GINs
    Search --> Render
    Search --> Sidebar
    Render --> UI
    Sidebar --> UI
```

---

## 6.2 Core Retrieval Logic

### 6.2.1 Query Vectorization
```python
QUERY_TASK_PREFIX = "search_query: "

def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    vector = embedder.encode([QUERY_TASK_PREFIX + query], convert_to_numpy=True)
    return list(map(float, vector[0]))
```
- **Prefix Critical**: `search_query:` Pairs with Ingestion's `search_document:`, nomic Dual-Tower Architecture Requires This
- **Dimension**: 768-Dim `float32` → Python `list[float]`

### 6.2.2 Hybrid Search SQL
```python
def search_novels(query_vector: Sequence[float], limit: int = 5):
    sql = """
        SELECT
            title, author, main_characters, key_locations,
            genres, tropes, one_liner_summary,
            1 - (summary_embedding <=> %s::vector) AS similarity_score
        FROM novel_catalog
        WHERE summary_embedding IS NOT NULL
        ORDER BY similarity_score DESC
        LIMIT %s
    """
    start = time.perf_counter()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (format_vector(query_vector), limit))
            rows = cur.fetchall()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return rows, latency_ms
```

### Key Points
| Component | Description |
|-----------|-------------|
| `<=>` | pgvector Cosine Distance Operator, Returns `[0, 2]`, Smaller = More Similar |
| `1 - (emb <=> q)` | Convert to Similarity Score `[0, 1]`, Larger = More Similar |
| `%s::vector` | Explicit Cast, Avoids Implicit Cast Errors |
| `WHERE embedding IS NOT NULL` | Filters Non-Vectorized Dirty Data |
| `LIMIT 5` | Fixed Top-K=5, Configurable |

### Vector Formatting
```python
def format_vector(vector: Sequence[float]) -> str:
    cleaned = [0.0 if (v != v or v == float('inf') or v == float('-inf')) else v for v in vector]
    return "[" + ",".join(f"{v:.8f}" for v in cleaned) + "]"
```
- **NaN/Inf Guard**: Prevents pgvector Errors from Model Inference Anomalies
- **Precision**: 8 Decimal Places ≈ 1e-8, Sufficient for Cosine Similarity

---

## 6.3 Streamlit UI Implementation

### 6.3.1 Page Config
```python
st.set_page_config(
    page_title="Novel-AIP",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Novel-AIP — Semantic Novel Discovery Platform")
st.caption("Real-Time Hybrid Retrieval: Local nomic-embed-text-v1.5 Vectorization + PostgreSQL pgvector Cosine Similarity")
```

### 6.3.2 Search Interaction
```python
query = st.text_input(
    "Search Your Remembered Novel Plot",
    placeholder="Describe the specific type of book narrative context you remember...",
    label_visibility="collapsed"
)
search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

if not (search_clicked and query and query.strip()):
    st.info("Enter a Plot Fragment You Remember, Retrieve Top 5 Most Similar Novels.")
    return
```

### 6.3.3 Result Card Rendering
```python
def render_result(row):
    title, author, chars, locs, genres, tropes, summary, score = row
    score_pct = float(score) * 100.0 if score else 0.0

    header = f"**{title or 'Untitled'}**"
    if author: header += f" — *{author}*"
    header += f"　`similarity {score_pct:.1f}%`"

    with st.container(border=True):
        st.markdown(header)
        cols = st.columns(3)
        cols[0].markdown(f"**Genres**\n" + _tags(genres, "Unknown"))
        cols[1].markdown(f"**Main Characters**\n" + _tags(chars, "—"))
        cols[2].markdown(f"**Tropes**\n" + _tags(tropes, "—"))
        if locs: st.markdown(f"**Locations**: {', '.join(locs)}")
        if summary: st.markdown(f"> {summary}")

def _tags(values, fallback):
    if not values: return fallback
    return " · ".join(f"`{v}`" for v in values)
```

### 6.3.4 Sidebar Real-time Metrics
```python
def db_stats():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(id) FROM novel_catalog")
            total = cur.fetchone()[0]
            cur.execute("SELECT pg_size_pretty(pg_relation_size('novel_catalog'))")
            size = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM novel_catalog WHERE summary_embedding IS NOT NULL")
            embedded = cur.fetchone()[0]
    return {"total": int(total), "embedded": int(embedded), "size": str(size)}

def render_sidebar():
    st.sidebar.title("Novel-AIP")
    st.sidebar.caption("Local-first Semantic Novel Discovery")
    try:
        stats = db_stats()
        st.sidebar.metric("Total Books Profiled", f"{stats['total']:,}")
        st.sidebar.metric("Embedded Vectors", f"{stats['embedded']:,}")
        st.sidebar.metric("Physical DB Space", stats["size"])
    except Exception as e:
        st.sidebar.error(f"Database Unavailable: {e}")
```

---

## 6.4 Model Caching & Resource Management

### 6.4.1 Embedding Model Cache
```python
@st.cache_resource(show_spinner="Loading local embedding model...")
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    os.environ.setdefault("TQDM_DISABLE", "1")  # Critical: Disable tqdm to Avoid Streamlit stderr Conflict
    kwargs = {}
    if hf_token := os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        kwargs["token"] = hf_token
    model = SentenceTransformer(model_name, **kwargs)
    return model
```
- **`st.cache_resource`**: Cross-Session Model Reuse, Avoids Reload
- **`TQDM_DISABLE=1`**: Module Top **Must Be Set Before Importing transformers/sentence_transformers**, Else tqdm Progress Bar Writes to Streamlit-Redirected stderr → `OSError: [Errno 22] Invalid Argument`

### 6.4.2 DB Connection Management
```python
def get_connection():
    return psycopg2.connect(**db_conn_info())
```
- Simplified: New Connection Per Query, Leverages PG Connection Pooling
- Production: `psycopg2.pool.ThreadedConnectionPool` + Context Manager

---

## 6.4 Error Handling & Observability

### 6.4.1 Unified Exception Capture
```python
try:
    embedder = load_embedder()
    query_vector = embed_query(embedder, query.strip())
    rows, latency_ms = search_novels(query_vector)
    st.success(f"Query Latency: **{latency_ms:,.2f} ms** · Returned {len(rows)} Results")
    for row in rows:
        render_result(row)
except Exception as exc:
    import traceback
    LOGGER.exception("Search pipeline failed")
    st.error(f"Search Failed: {exc}\n\n```\n{traceback.format_exc()}\n```")
```
- **Full Stack**: Frontend Shows Traceback, User Feedback/Self-Diagnosis
- **Tiered Logging**: `LOGGER.exception` Structured Logs, Frontend User-Friendly

### 6.4.2 Common Error Classification Display
| Error Type | Frontend Hint | Backend Log |
|------------|---------------|-------------|
| Model Load Fail | "Model Load Failed, Check HF_TOKEN/Network/Disk" | `LOGGER.exception` + Full Stack |
| DB Connect Fail | "Database Unavailable, Check PostgreSQL Service" | Conn Params, Error Code |
| Vector Dim Mismatch | "Vector Dim Error, Re-run database_worker.py" | Dim Comparison Log |
| Empty Results | "No Matches, Try Different Description" | Query Vector, SQL Plan |

---

## 6.5 Production Deployment

### Systemd Unit (`/etc/systemd/system/novel-aip.service`)
```ini
[Unit]
Description=Novel-AIP Streamlit Search Service
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=novel-aip
WorkingDirectory=/opt/novel-aip
Environment=PATH=/opt/novel-aip/.venv/bin
EnvironmentFile=/opt/novel-aip/.env
ExecStart=/opt/novel-aip/.venv/bin/streamlit run app.py --server.headless=true --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Enable & Manage
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now novel-aip
sudo journalctl -u novel-aip -f  # Follow Logs
```

### Nginx Reverse Proxy (Optional)
```nginx
server {
    listen 80;
    server_name novel-aip.local;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

---

## 6.6 Performance Tuning Suggestions

| Optimization | Current | Target | Implementation |
|--------------|---------|--------|----------------|
| Model Load First Paint | First Query Triggers Load | Startup Pre-warm | `@st.cache_resource` + Startup Script Pre-Call |
| Query P99 Latency | ~130 ms (Sample) | < 50 ms (100k Scale) | `hnsw.ef_search=200` + Conn Pool + Read Replica |
| Concurrency | Single-Thread Blocking | 10+ Concurrent | `ThreadedConnectionPool` + Streamlit Multi-Process |
| Static Assets | No CDN | CDN Acceleration | Nginx `proxy_cache` + Cloudflare |
| Auth/Authorization | None | Optional | `streamlit-authenticator` / OAuth2 Proxy |

---

## 6.7 Related Documents

- [Vector Database Design & Data Loading](05-Vector-Database-Design-and-Data-Loading.md) — Table/Indexes Search SQL Depends On
- [Operations, Monitoring & Performance Tuning](07-Operations-Monitoring-and-Performance-Tuning.md) — Latency Metrics, Conn Pool, Read Replicas
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — Search Errors/Empty Results/Latency Spikes
- [API Reference & Specifications](09-API-Reference-and-Specifications.md) — Internal Function Signatures