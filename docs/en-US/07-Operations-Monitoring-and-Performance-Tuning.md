# 07 Operations, Monitoring & Performance Tuning

## 7.1 Key Metrics Framework

### 7.1.1 Red-Line Alerts (Must Alert)

| Metric | Alert Threshold | Collection Interval | Source |
|--------|----------------|---------------------|--------|
| **Search P99 Latency** | > 500 ms | 30 s | `app.py` `perf_counter` / Prometheus `histogram_quantile` |
| **Search Error Rate** | > 1% / 5min | 60 s | Structured Logs `level=ERROR` |
| **DB Connection Pool Usage** | > 80% | 30 s | `pg_stat_activity` / Pool Metrics |
| **Disk Usage** | > 85% | 5 min | `df -h` / `pg_database_size` |
| **Memory Usage** | > 90% | 60 s | `psutil` / `node_exporter` |
| **GPU VRAM Usage** | > 95% | 60 s | `rocm-smi` / `nvidia-smi` |
| **LLM Service Availability** | < 99.9% | 60 s | `/v1/models` Health Check |

### 7.1.2 Business KPIs (Trend Observation)

| Metric | Purpose | Collection |
|--------|---------|------------|
| **Daily Search Count** | Capacity Planning | `app.py` Counter / Prometheus `counter` |
| **Vector Coverage** | `embedded / total` | Periodic SQL `COUNT(embedding IS NOT NULL) / COUNT(*)` |
| **Avg Search Latency** | Performance Baseline | `histogram` Buckets |
| **Ingestion Throughput** | `records/s` | `database_worker.py` Progress / Log Parsing |
| **LLM Inference Queue Length** | Scale Decision | `llama.cpp` Internal Queue / Custom Queue Monitor |

---

## 7.2 Monitoring Stack Deployment

### 7.2.1 Prometheus + Grafana (Docker Compose)
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:v2.53
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prom_data:/prometheus
    ports: ["9090:9090"]
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:10.4
    ports: ["3000:3000"]
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=changeme
    depends_on: [prometheus]

  node_exporter:
    image: prom/node-exporter:v1.7
    pid: host
    network_mode: host
    volumes: ["/:/host:ro,rslave"]
    command: ['--path.rootfs=/host']

volumes:
  prom_data:
  grafana_data:
```

### 7.2.2 Prometheus Scrape Config
```yaml
# prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'novel-aip-app'
    static_configs:
      - targets: ['host.docker.internal:8501']  # Streamlit Needs Custom /metrics
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['host.docker.internal:9187']  # postgres_exporter

  - job_name: 'node'
    static_configs:
      - targets: ['host.docker.internal:9100']

  - job_name: 'llama-cpp'
    static_configs:
      - targets: ['host.docker.internal:8080']  # Health Check Only
```

> **Note**: Streamlit Has No Native `/metrics`, Must Integrate `prometheus_client` to Expose `/metrics` or Use `starlette_exporter` Middleware.

---

## 7.3 Grafana Dashboard Key Panels

| Panel | Query Example | Visualization |
|-------|---------------|---------------|
| **Search P50/P95/P99** | `histogram_quantile(0.99, rate(app_search_latency_seconds_bucket[5m]))` | Heatmap / Time Series |
| **QPS** | `rate(app_search_total[1m])` | Time Series |
| **Error Rate** | `rate(app_errors_total[5m]) / rate(app_requests_total[5m])` | Stat + Threshold |
| **DB Connections** | `pg_stat_activity_count` | Gauge |
| **Table Size Trend** | `pg_relation_size('novel_catalog')` | Time Series |
| **Index Size** | `pg_indexes_size('novel_catalog')` | Time Series |
| **HNSW Index Size** | `pg_relation_size('idx_novel_hnsw')` | Time Series |
| **GPU VRAM** | `rocm_smi_memory_used_bytes / rocm_smi_memory_total_bytes` | Gauge |
| **LLM Inference Queue** | `llama_cpp_queue_length` (Custom) | Time Series |

---

## 7.4 Database Performance Tuning

### 7.4.1 Index Maintenance Strategy

| Operation | Frequency | Command | Impact |
|-----------|-----------|---------|--------|
| **GIN Concurrent Rebuild** | Weekly Off-Peak | `REINDEX INDEX CONCURRENTLY idx_novel_tropes_gin;` | No Lock, CPU/IO Only |
| **HNSW Rebuild** | Monthly/Data Drift | `REINDEX INDEX idx_novel_hnsw;` | **Exclusive Lock**, Maintenance Window |
| **Stats Update** | Auto/Daily | `ANALYZE novel_catalog;` | Trivial, Frequent OK |
| **VACUUM (FULL)** | Monthly/Bloat > 20% | `VACUUM (FULL) novel_catalog;` | **Exclusive Lock**, Maintenance Window |

### 7.4.2 HNSW Runtime Tuning
```sql
-- Session-Level Tuning (Current Session Only)
SET hnsw.ef_search = 200;  -- Default 40, Recall↑, Latency Slight↑

-- Verify Current
SHOW hnsw.ef_search;

-- Permanent (postgresql.conf)
hnsw.ef_search = 200
```
| `ef_search` | Recall (Approx) | Latency Increase | Use Case |
|-------------|----------------|------------------|----------|
| 40 (Default) | ~95% | Baseline | High Throughput, Tolerate Some Misses |
| 100 | ~98% | +20% | Balanced |
| 200 | ~99.5% | +50% | High Recall Priority |
| 400 | ~99.9% | +100% | Offline Batch/Eval |

### 7.4.3 Connection Pool Config
```python
# psycopg2.pool.ThreadedConnectionPool
pool = ThreadedConnectionPool(
    minconn=2,
    maxconn=20,
    **db_conn_info()
)

conn = pool.getconn()
try:
    with conn.cursor() as cur:
        cur.execute(...)
finally:
    pool.putconn(conn)
```
| Param | Recommended | Rationale |
|-------|-------------|-----------|
| `minconn` | 2 | Keep Hot Connections, Reduce Cold Start |
| `maxconn` | CPU Cores × 2 | Avoid PostgreSQL `max_connections` Exhaustion |
| `connect_timeout` | 10s | Avoid Long Block |
| `keepalives_idle` | 30 | Prevent Firewall/NAT Dropping Idle Connections |

---

## 7.5 Model Service Tuning

### 7.1 llama.cpp Launch Param Tuning
```bash
# Optimized for Qwen2.5-27B Q6_K @ RDNA 3.5 (gfx1151)
./llama-server \
  -m models/qwen2.5-27b-instruct-q6_k.gguf \
  -c 8192 \           # Context Window, Increase as Needed
  -ngl 99 \           # All Layers Offloaded to GPU
  -ub 512 \           # User Batch Size
  -b 512 \            # Batch Size
  -fa \               # Flash Attention (If Supported)
  --mlock \           # Lock Memory Prevent Swap
  --no-mmap \         # Disable mmap (More Stable on ROCm)
  --port 8080 \
  --host 127.0.0.1
```

| Param | Description | Tuning Direction |
|-------|-------------|-------------------|
| `-b` / `-ub` | Batch Size | Increase to 1024/2048 If VRAM Allows |
| `-c` | Context Window | Increase as Needed, VRAM Linear Growth |
| `--mlock` | Lock Memory | Prevent Swap Latency Jitter |
| `--no-mmap` | Disable mmap | ROCm Avoids Page Fault Latency Spikes |
| `-fa` | Flash Attention | If HW Supports, Significant Long-Context Speedup |

### 7.2 nomic-embed-text-v1.5 Inference Acceleration
```python
# CPU Multi-Thread (SentenceTransformer Uses torch Internally)
import torch
torch.set_num_threads(8)  # Physical Cores

# GPU Accel (Needs CUDA/ROCm torch)
model = SentenceTransformer(model_name, device="cuda")  # Or "hip"
# Batch Inference
vectors = model.encode(texts, batch_size=256, convert_to_numpy=True)
```
| Device | Relative Speed (batch=100) | VRAM Usage |
|--------|---------------------------|------------|
| CPU (8 Threads) | 1.0x (Baseline) | ~2 GB |
| ROCm (RDNA 3.5) | 8-12x | ~4 GB |
| CUDA (RTX 4090) | 15-20x | ~4 GB |

---

## 7.3 Capacity Planning Table

| Data Scale | Vector Memory (HNSW) | Disk (Table+Indexes) | RAM Recommended | GPU VRAM | Est. QPS |
|------------|---------------------|---------------------|-----------------|----------|----------|
| 10k | ~150 MB | ~500 MB | 8 GB | 8 GB | 50 |
| 100k | ~1.5 GB | ~5 GB | 16 GB | 12 GB | 30 |
| 1M | ~15 GB | ~50 GB | 64 GB | 24 GB | 20 |
| 10M | ~150 GB | ~500 GB | 256 GB | 48 GB+ (Multi-GPU) | 15 |

> **Rule of Thumb**: HNSW Memory ≈ `Vector Count × Dim × 4 Bytes × 1.5 (Graph Overhead)`; Disk ≈ Memory × 3-4 (Incl WAL, Indexes, TOAST).

---

## 7.4 Automated Ops Scripts

### 7.4.1 Daily Check (`daily_check.sh`)
```bash
#!/bin/bash
set -euo pipefail

LOG="/var/log/novel-aip/daily_$(date +%F).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) Daily Check ==="

# 1. Service Alive
for svc in postgresql llama-cpp streamlit; do
    systemctl is-active --quiet $svc && echo "✅ $svc active" || { echo "❌ $svc DOWN"; exit 1; }
done

# 2. Port Reachable
for port in 5432 8080 8501; do
    nc -z localhost $port && echo "✅ Port $port Open" || echo "❌ Port $port CLOSED"
done

# 3. Disk Space
df -h / | awk 'NR==2 {print "Disk: " $5 " Used (" $4 " Free)"}'

# 4. Memory/VRAM
free -h | awk 'NR==2 {print "RAM: " $3 "/" $2}'
rocm-smi --showmemuse vram 2>/dev/null | head -5

# 5. DB Connections
psql -U postgres -d fileindex -tAc "SELECT count(*) FROM pg_stat_activity;" | xargs -I{} echo "DB Connections: {}"

# 6. Table Size
psql -U postgres -d fileindex -c "SELECT pg_size_pretty(pg_relation_size('novel_catalog')) AS table_size, pg_size_pretty(pg_total_relation_size('novel_catalog')) AS total_size;"

# 7. Last 1h Error Rate
ERRORS=$(journalctl -u novel-aip --since "1 hour ago" -p err --no-pager | wc -l)
echo "Recent Errors (1h): $ERRORS"

echo "=== Check Complete ==="
```

### 7.4.2 Cron Jobs
```cron
# Daily 02:00 Check
0 2 * * * /opt/novel-aip/scripts/daily_check.sh >> /var/log/novel-aip/daily_check.log 2>&1

# Weekly Sun 03:00 GIN Concurrent Rebuild
0 3 * * 0 psql -U postgres -d fileindex -c "REINDEX INDEX CONCURRENTLY idx_novel_tropes_gin; REINDEX INDEX CONCURRENTLY idx_novel_genres_gin; REINDEX INDEX CONCURRENTLY idx_novel_main_characters_gin; REINDEX INDEX CONCURRENTLY idx_novel_key_locations_gin;"

# Monthly 1st 04:00 HNSW Rebuild + VACUUM FULL
0 4 1 * * psql -U postgres -d fileindex -c "REINDEX INDEX idx_novel_hnsw; VACUUM (FULL) novel_catalog; ANALYZE novel_catalog;"
```

---

## 7.5 Performance Regression Testing

### 7.5.1 Benchmark Script (`benchmark.py`)
```python
#!/usr/bin/env python3
import time, statistics, psycopg2, os
from sentence_transformers import SentenceTransformer

QUERIES = [
    "奇幻玄幻 主角重生 复仇",
    "都市异能 系统流 打脸",
    "历史穿越 权谋 皇帝",
    "科幻 星际 机甲 进化",
    "悬疑 推理 密室 杀人",
]

def benchmark():
    embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
    conn = psycopg2.connect(**db_conn_info())
    
    latencies = []
    for q in QUERIES * 20:  # 100 Queries
        vec = embedder.encode(["search_query: " + q])[0]
        vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
        
        start = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 1 - (summary_embedding <=> %s::vector) AS score
                FROM novel_catalog
                WHERE summary_embedding IS NOT NULL
                ORDER BY score DESC LIMIT 5
            """, (vec_str,))
            _ = cur.fetchall()
        latencies.append((time.perf_counter() - start) * 1000)
    
    print(f"Queries: {len(latencies)}")
    print(f"P50: {statistics.median(latencies):.2f} ms")
    print(f"P95: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
    print(f"P99: {statistics.quantiles(latencies, n=100)[98]:.2f} ms")
    print(f"Mean: {statistics.mean(latencies):.2f} ms")

if __name__ == "__main__":
    benchmark()
```

### Regression Thresholds
| Metric | Baseline | Regression Threshold | Action |
|--------|----------|----------------------|--------|
| P99 Latency | 80 ms | > 200 ms | Immediate Alert, Rollback/Scale |
| P50 Latency | 30 ms | > 100 ms | Alert, Investigate Index/Pool |
| Error Rate | 0% | > 0.1% | Immediate Alert |

---

## 7.5 Related Documents

- [Vector Database Design & Data Loading](05-Vector-Database-Design-and-Data-Loading.md) — Index Param Details
- [Hybrid Search Service & Web UI](06-Hybrid-Search-Service-and-Web-UI.md) — Service Metrics Exposure
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — Perf Jitter/Slow Query Debug
- [Extension Development Guide](10-Extension-Development-Guide.md) — Custom Metrics Extensions