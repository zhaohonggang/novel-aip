# 06 混合检索服务与 Web UI

## 6.1 服务架构概览

```mermaid
flowchart TB
    subgraph Client["🌐 客户端"]
        Browser["浏览器\nhttp://localhost:8501"]
    end

    subgraph Service["🔧 Search Service (app.py)"]
        UI[Streamlit UI\nst.text_input + st.button]
        Embed[nomic-embed-text-v1.5\nsearch_query: 前缀]
        Search[混合检索 SQL\npgvector <=> + GIN 过滤]
        Render[卡片渲染\nst.container(border=True)]
        Sidebar[侧栏实时指标\nCOUNT / pg_size_pretty / perf_counter]
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

## 6.2 核心检索逻辑

### 6.2.1 查询向量化
```python
QUERY_TASK_PREFIX = "search_query: "

def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    vector = embedder.encode([QUERY_TASK_PREFIX + query], convert_to_numpy=True)
    return list(map(float, vector[0]))
```
- **前缀关键**：`search_query:` 与入库时的 `search_document:` 配对，nomic 双塔架构要求
- **维度**：768 维 `float32` → Python `list[float]`

### 6.2.2 混合检索 SQL
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

### 关键点解析
| 组件 | 说明 |
|------|------|
| `<=>` | pgvector 余弦距离算子，返回 `[0, 2]`，越小越相似 |
| `1 - (emb <=> q)` | 转换为相似度分数 `[0, 1]`，越大越相似 |
| `%s::vector` | 显式转型，避免隐式转换报错 |
| `WHERE embedding IS NOT NULL` | 过滤未向量化脏数据 |
| `LIMIT 5` | Top-K 固定 5，可配置化 |

### 向量格式化
```python
def format_vector(vector: Sequence[float]) -> str:
    cleaned = [0.0 if (v != v or v == float('inf') or v == float('-inf')) else v for v in vector]
    return "[" + ",".join(f"{v:.8f}" for v in cleaned) + "]"
```
- **NaN/Inf 防护**：防止模型推理异常产生非法值导致 pgvector 报错
- **精度**：8 位小数 ≈ 1e-8，余弦相似度精度充分

---

## 6.3 Streamlit UI 实现

### 6.3.1 页面配置
```python
st.set_page_config(
    page_title="Novel-AIP",
    page_icon="📚",
    layout="wide"
)
st.title("📚 Novel-AIP — 语义小说检索平台")
st.caption("实时混合检索：本地 nomic-embed-text-v1.5 向量化 + PostgreSQL pgvector 余弦相似度")
```

### 6.3.2 搜索交互
```python
query = st.text_input(
    "搜索您记忆中的小说情节",
    placeholder="Describe the specific type of book narrative context you remember...",
    label_visibility="collapsed"
)
search_clicked = st.button("🔍 检索", type="primary", use_container_width=True)

if not (search_clicked and query and query.strip()):
    st.info("输入一段您记得的故事设定或情节，检索 Top 5 最相似的小说。")
    return
```

### 6.3.3 结果卡片渲染
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
        cols[0].markdown(f"**体裁**\n" + _tags(genres, "Unknown"))
        cols[1].markdown(f"**主要角色**\n" + _tags(chars, "—"))
        cols[2].markdown(f"**故事套路**\n" + _tags(tropes, "—"))
        if locs: st.markdown(f"**地点**: {', '.join(locs)}")
        if summary: st.markdown(f"> {summary}")

def _tags(values, fallback):
    if not values: return fallback
    return " · ".join(f"`{v}`" for v in values)
```

### 6.3.4 侧栏实时指标
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
    st.sidebar.caption("Local-first semantic novel discovery")
    try:
        stats = db_stats()
        st.sidebar.metric("Total Books Profiled", f"{stats['total']:,}")
        st.sidebar.metric("Embedded Vectors", f"{stats['embedded']:,}")
        st.sidebar.metric("Physical DB Space", stats["size"])
    except Exception as e:
        st.sidebar.error(f"Database unavailable: {e}")
```

---

## 6.4 模型缓存与资源管理

### 6.4.1 嵌入模型缓存
```python
@st.cache_resource(show_spinner="Loading local embedding model...")
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    os.environ.setdefault("TQDM_DISABLE", "1")  # 关键：禁用 tqdm 避免 Streamlit stderr 冲突
    kwargs = {}
    if hf_token := os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"):
        kwargs["token"] = hf_token
    model = SentenceTransformer(model_name, **kwargs)
    return model
```
- **`st.cache_resource`**：跨会话复用模型实例，避免重复加载
- **`TQDM_DISABLE=1`**：模块顶部设置，**必须在导入 transformers/sentence_transformers 之前**生效，否则 tqdm 进度条写入 Streamlit 重定向的 stderr 导致 `OSError: [Errno 22] Invalid argument`

### 6.4.2 数据库连接管理
```python
def get_connection():
    return psycopg2.connect(**db_conn_info())
```
- 简化版：每次查询新建连接，利用 PostgreSQL 连接池特性
- 生产建议：`psycopg2.pool.ThreadedConnectionPool` + 上下文管理器

---

## 6.5 错误处理与可观测性

### 6.5.1 统一异常捕获
```python
try:
    embedder = load_embedder()
    query_vector = embed_query(embedder, query.strip())
    rows, latency_ms = search_novels(query_vector)
    st.success(f"Query Latency: **{latency_ms:,.2f} ms** · 返回 {len(rows)} 条结果")
    for row in rows:
        render_result(row)
except Exception as exc:
    import traceback
    LOGGER.exception("Search pipeline failed")
    st.error(f"检索失败: {exc}\n\n```\n{traceback.format_exc()}\n```")
```
- **完整堆栈**：前端直接展示 traceback，便于用户反馈/自助排查
- **分级日志**：`LOGGER.exception` 记录结构化日志，前端展示用户友好信息

### 6.5.2 常见错误分类展示
| 错误类型 | 前端提示 | 后端日志 |
|----------|----------|----------|
| 模型加载失败 | "模型加载失败，请检查 HF_TOKEN/网络/磁盘空间" | `LOGGER.exception` + 完整堆栈 |
| 数据库连接失败 | "数据库不可用，请检查 PostgreSQL 服务" | 连接参数、错误码 |
| 向量维度不匹配 | "向量维度异常，请重新运行 database_worker.py" | 维度对比日志 |
| 空结果 | "未找到匹配结果，请尝试换一种描述" | 查询向量、SQL 执行计划 |

---

## 6.6 部署为系统服务

### Systemd 单元文件 (`/etc/systemd/system/novel-aip.service`)
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

### 启用与管理
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now novel-aip
sudo journalctl -u novel-aip -f  # 查看日志
```

### Nginx 反向代理 (可选)
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

## 6.6 性能调优建议

| 优化点 | 当前 | 目标 | 实现 |
|--------|------|------|------|
| 模型加载首屏 | 首次查询触发加载 | 启动时预热 | `@st.cache_resource` + 启动脚本预调用 |
| 查询延迟 P99 | ~130 ms (样本) | < 50 ms (10 万级) | `hnsw.ef_search=200` + 连接池 + 只读副本 |
| 并发支持 | 单线程阻塞 | 10+ 并发 | `ThreadedConnectionPool` + `streamlit` 多进程模式 |
| 静态资源 | 无 CDN | CDN 加速 | Nginx `proxy_cache` + Cloudflare |
| 认证授权 | 无 | 可选 | `streamlit-authenticator` / OAuth2 Proxy |

---

## 6.7 相关文档

- [向量数据库设计与数据入库](05-向量数据库设计与数据入库.md) — 检索 SQL 依赖的表/索引
- [运维监控与性能调优](07-运维监控与性能调优.md) — 延迟指标、连接池、只读副本
- [故障排查与常见问题](08-故障排查与常见问题.md) — 检索报错/空结果/延迟飙升
- [API 参考与接口规范](09-API-参考与接口规范.md) — 内部函数签名