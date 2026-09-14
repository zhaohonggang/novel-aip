# 09 API 参考与接口规范

## 9.1 内部 Python API

### 9.1.1 `ingestion_core` 模块

#### `CheckpointManager`
```python
class CheckpointManager:
    def __init__(self, path: Path | str = CHECKPOINT_PATH) -> None:
        """初始化检查点管理器。

        Args:
            path: 检查点文件路径，默认 ./checkpoint.json
        """

    def load(self) -> None:
        """从磁盘加载检查点状态。"""
        
    def save(self) -> None:
        """原子写入检查点到磁盘。"""
        
    def register(self, file_path: Path) -> None:
        """注册新文件为 PENDING 状态。

        Args:
            file_path: 绝对路径 Path 对象
        """
        
    def is_processed(self, file_path: Path) -> bool:
        """检查文件是否已处理完成。

        Returns:
            bool: True=已处理, False=未处理或失败
        """
        
    def mark_processed(self, file_path: Path) -> None:
        """标记文件为 PROCESSED，更新 size/mtime。"""
        
    def mark_failed(self, file_path: Path, error: str) -> None:
        """标记文件为 FAILED，记录错误堆栈。

        Args:
            error: traceback.format_exception() 字符串
        """
        
    @property
    def records(self) -> dict[str, dict[str, Any]]:
        """获取所有记录的只读副本。"""
```

#### `CheckpointStatus` 枚举
```python
class CheckpointStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
```

#### 核心函数
```python
def iter_text_files(root: Path) -> Iterator[Path]:
    """遍历根目录下所有 .txt/.md 文件。

    Yields:
        Path: 绝对路径
    """

def read_text_robust(file_path: Path) -> tuple[str, str]:
    """多编码回退读取文本。

    Returns:
        tuple: (content, encoding_used)

    Raises:
        UnicodeDecodeError: 所有编码均失败
        OSError: 文件系统错误
    """

def run_ingestion(
    source_root: Path = SOURCE_ROOT,
    checkpoint_path: Path = CHECKPOINT_PATH
) -> int:
    """执行完整采集流程。

    Returns:
        int: 失败文件数，0 表示全成功
    """
```

#### 常量
```python
SOURCE_ROOT = Path("./source_novels")
CHECKPOINT_PATH = Path("./checkpoint.json")
TEXT_EXTENSIONS = {".txt", ".md"}
ENCODING_FALLBACKS = ("utf-8", "gbk", "gb18030")
```

---

### 9.1.2 `profiler` 模块

#### `NovelSchema` (Pydantic Model)
```python
class NovelSchema(BaseModel):
    main_genres: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="最多 3 个主流体裁标签"
    )
    story_tropes: list[str] = Field(
        default_factory=list,
        description="情节套路/梗，如: 系统流、背叛、复仇"
    )
    one_sentence_summary: str = Field(
        default="",
        max_length=200,
        description="单句核心梗概，≤200 字"
    )
```

#### 核心函数
```python
def load_nlp(disable: Sequence[str] = ("parser", "tagger", "senter")) -> Any:
    """加载 spaCy 中文模型，裁剪管道。

    Args:
        disable: 禁用的管道组件名列表

    Returns:
        spacy.Language: 已加载的 nlp 对象
    """

def extract_top_entities(
    text: str,
    nlp: Any,
    head_chars: int = 50_000,
    top_n: int = 5
) -> tuple[list[str], list[str]]:
    """Stage 1: 前 N 字符 NER 抽取 Top-K 人物/地名。

    Returns:
        tuple: (top_persons, top_locations)
    """

def extract_chapter_headers(
    text: str,
    max_headers: int = 40
) -> list[str]:
    """正则抽取章节标题行。"""

def build_dense_snippet(
    text: str,
    intro_chars: int = 2_000,
    max_headers: int = 40
) -> str:
    """组装稠密片段：引言 + 章节概览。"""

def build_llm_chain(
    llm: ChatOpenAI,
    schema: type[BaseModel] = NovelSchema
) -> Runnable:
    """构建 LLM 结构化抽取链。

    Returns:
        Runnable: prompt | llm | JsonOutputParser
    """

def profile_text(
    text: str,
    nlp: Any,
    chain: Runnable
) -> NovelSchema:
    """完整画像流程：NER + 片段组装 + LLM 抽取。

    Returns:
        NovelSchema: 结构化画像
    """

def fallback_schema(
    main_characters: list[str],
    key_locations: list[str]
) -> NovelSchema:
    """LLM 不可用时的确定性降级 Schema。"""

def build_llm(
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "qwen2.5-27b-instruct",
    api_key: str = "not-needed-local-inference",
    temperature: float = 0.1
) -> ChatOpenAI:
    """构建本地化 ChatOpenAI 客户端。"""
```

#### 常量
```python
HEAD_SLICE_CHARS = 50_000
INTRO_CHARS = 2_000
MAX_CHAPTER_HEADERS = 40
CHAPTER_HEADER_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+[章节回卷集部篇段]")
DEFAULT_HOST = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen2.5-27b-instruct"
DEFAULT_API_KEY = "not-needed-local-inference"
DEFAULT_TEMPERATURE = 0.1
```

#### CLI 接口
```bash
python profiler.py --file PATH [--host HOST] [--model MODEL] [--offline] [--log-level LEVEL]
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--file` | Path | 必填 | 单个小说文件路径 |
| `--host` | str | `http://127.0.0.1:8080/v1` | LLM 服务地址 |
| `--model` | str | `qwen2.5-27b-instruct` | 模型名 |
| `--offline` | flag | False | 仅降级模式，不调用 LLM |
| `--log-level` | str | `INFO` | 日志级别 |

---

### 9.1.3 `database_worker` 模块

#### 核心函数
```python
def env_conn_info() -> dict[str, Any]:
    """从环境变量构建数据库连接参数。"""

def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """加载嵌入模型，支持 HF_TOKEN。"""

def embed_summaries(
    embedder: SentenceTransformer,
    summaries: Sequence[str],
    batch_size: int = BATCH_SIZE,
    task_prefix: str = TASK_PREFIX_RETRIEVAL
) -> list[list[float]]:
    """批量嵌入摘要文本。

    Args:
        task_prefix: "search_document: " (入库) 或 "search_query: " (查询)
    """

def format_vector(vector: Sequence[float]) -> str:
    """向量转 pgvector 文本字面量 '[0.1,0.2,...]'。"""

def insert_chunk(
    records: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    conn_info: dict,
    batch_size: int = BATCH_SIZE
) -> int:
    """单批次事务插入，返回插入行数。"""

def process_records(
    records: list[dict],
    embedder: SentenceTransformer,
    conn_info: dict,
    batch_size: int = BATCH_SIZE,
    retry_limit: int = RETRY_LIMIT,
    retry_delay: float = RETRY_DELAY_SECONDS
) -> bool:
    """完整入库流程：嵌入 -> 分批 -> 事务写入 -> 重试/熔断。

    Returns:
        bool: True=全成功, False=触发崩溃标志
    """

def load_records(source: Path) -> list[dict]:
    """加载 JSON 载荷，支持数组或 {records: [...]} 格式。"""
```

#### 常量
```python
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE = 100
RETRY_LIMIT = 3
RETRY_DELAY_SECONDS = 5.0

TASK_PREFIX_RETRIEVAL = "search_document: "
TASK_PREFIX_QUERY = "search_query: "

INSERT_COLUMNS = (
    "title", "author", "main_characters", "key_locations",
    "genres", "tropes", "one_liner_summary",
    "summary_embedding", "raw_json_payload"
)
```

#### CLI 接口
```bash
python database_worker.py --payload records.json [--model MODEL] [--batch-size N] [--log-level LEVEL]
```

---

### 9.1.4 `app` 模块 (Streamlit 服务)

#### 核心函数
```python
def db_conn_info() -> dict[str, Any]:
    """从环境变量构建数据库连接参数。"""

def format_vector(vector: Sequence[float]) -> str:
    """向量格式化，含 NaN/Inf 防护。"""

@st.cache_resource
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """加载并缓存嵌入模型 (TQDM_DISABLE=1)。"""

def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    """查询文本向量化 (search_query: 前缀)。"""

def get_connection() -> psycopg2.connection:
    """获取数据库连接。"""

def db_stats() -> dict[str, str]:
    """查询侧栏统计: total/embedded/size。"""

def search_novels(query_vector: Sequence[float], limit: int = 5) -> tuple[list, float]:
    """执行混合检索，返回 (rows, latency_ms)。"""

def render_result(row: tuple) -> None:
    """渲染单条结果卡片。"""

def render_sidebar() -> None:
    """渲染侧栏实时指标。"""

def main() -> None:
    """Streamlit 入口。"""
```

#### 常量
```python
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
QUERY_TASK_PREFIX = "search_query: "
TOP_K = 5
SEARCH_PLACEHOLDER = "Describe the specific type of book narrative context you remember..."
```

---

## 9.2 数据模型定义

### 9.2.1 `NovelSchema` (JSON)
```json
{
  "title": "string",
  "author": "string|null",
  "main_characters": ["string"],
  "key_locations": ["string"],
  "genres": ["string"] (max 3),
  "tropes": ["string"],
  "one_liner_summary": "string (≤200 chars)",
  "raw_json_payload": { "source_file": "string" }
}
```

### 9.2.2 `checkpoint.json` 记录结构
```json
{
  "C:\\abs\\path\\file.txt": {
    "file_size_bytes": 96568,
    "last_modified_timestamp": 1787153805.0961044,
    "status": "PENDING|PROCESSED|FAILED",
    "error_log": "string|null"
  }
}
```

### 9.2.3 数据库表 `novel_catalog`
```sql
-- 见 schema.sql 完整定义
id SERIAL PK
title VARCHAR(512) NOT NULL
author VARCHAR(256)
main_characters TEXT[]
key_locations TEXT[]
genres TEXT[]
tropes TEXT[]
one_liner_summary TEXT
summary_embedding VECTOR(768)
raw_json_payload JSONB
processed_at TIMESTAMP DEFAULT NOW()
```

---

## 9.3 SQL 查询接口

### 9.3.1 混合检索 (核心)
```sql
-- 参数: $1 = query_vector_text (如 '[0.1,0.2,...]'), $2 = limit
SELECT
    title,
    author,
    main_characters,
    key_locations,
    genres,
    tropes,
    one_liner_summary,
    1 - (summary_embedding <=> $1::vector) AS similarity_score
FROM novel_catalog
WHERE summary_embedding IS NOT NULL
ORDER BY similarity_score DESC
LIMIT $2;
```

### 9.3.2 侧栏统计查询
```sql
-- 总量
SELECT COUNT(id) FROM novel_catalog;

-- 向量覆盖数
SELECT COUNT(*) FROM novel_catalog WHERE summary_embedding IS NOT NULL;

-- 物理磁盘占用
SELECT pg_size_pretty(pg_relation_size('novel_catalog'));
```

### 9.3.3 索引维护
```sql
-- GIN 并发重建 (无锁)
REINDEX INDEX CONCURRENTLY idx_novel_tropes_gin;

-- HNSW 重建 (独占锁)
REINDEX INDEX idx_novel_hnsw;

-- 运行时调整 HNSW 搜索精度
SET hnsw.ef_search = 200;
```

---

## 9.4 环境变量规范

| 变量名 | 必填 | 默认值 | 说明 |
|--------|------|--------|------|
| `DATABASE_NAME` | ✅ | `fileindex` | 数据库名 |
| `DATABASE_USER` | ✅ | `postgres` | 用户名 |
| `DATABASE_PASSWORD` | ✅ | `postgres` | 密码 |
| `DATABASE_HOST` | ✅ | `localhost` | 主机 |
| `DATABASE_PORT` | ✅ | `5432` | 端口 |
| `SECRET_KEY` | ✅ | — | JWT 签名密钥 |
| `APP_ENV` | ⭕ | `development` | 环境标识 |
| `LOG_LEVEL` | ⭕ | `INFO` | 日志级别 |
| `HF_TOKEN` | ⭕ | — | Hugging Face 令牌 |
| `TQDM_DISABLE` | 内部 | `1` | 禁用 tqdm (app.py 模块顶部硬编码) |

---

## 9.5 错误码规范

| 码 | HTTP 状态 | 含义 | 典型场景 |
|------|-----------|------|----------|
| `INGESTION_FAILED` | 500 | 采集阶段失败 | 编码错误、权限、磁盘满 |
| `PROFILING_FAILED` | 500 | 画像生成失败 | LLM 超时、显存溢出、Prompt 解析失败 |
| `EMBEDDING_FAILED` | 500 | 向量化失败 | 模型加载失败、维度不匹配、显存 OOM |
| `DATABASE_CONNECTION_FAILED` | 503 | 数据库不可用 | 服务未启动、连接池耗尽、网络分区 |
| `DATABASE_QUERY_FAILED` | 500 | SQL 执行失败 | 语法错误、类型不匹配、约束冲突 |
| `INDEX_BUILD_FAILED` | 500 | 索引构建失败 | 内存不足、磁盘满、参数非法 |
| `MODEL_LOAD_FAILED` | 503 | 模型加载失败 | 权重缺失、版本不兼容、显存 OOM |
| `VALIDATION_ERROR` | 400 | 输入校验失败 | Pydantic 校验不通过、必填字段缺失 |
| `RATE_LIMIT_EXCEEDED` | 429 | 限流触发 | LLM 并发过高 / API 限流 |

### 错误响应格式
```json
{
  "error": {
    "code": "DATABASE_CONNECTION_FAILED",
    "message": "could not connect to server: Connection refused",
    "details": {
      "host": "localhost",
      "port": 5432
    },
    "trace_id": "abc123-def456"
  }
}
```

---

## 9.6 版本控制与兼容性

| 组件 | 版本策略 | 兼容性承诺 |
|------|----------|------------|
| Python API | 语义化版本 (MAJOR.MINOR.PATCH) | MINOR 兼容，MAJOR 破坏 |
| 数据库 Schema | 迁移脚本 (Alembic/手工) | 向后兼容，新增列可空 |
| 向量维度 | 固定 768 | 模型升级需同步迁移 |
| LLM Prompt | 版本化常量 | 非兼容变更需 MAJOR 版本 |
| REST API (未来) | OpenAPI 3.0 | 遵循语义化版本 |

---

## 9.6 相关文档

- [双阶段实体抽取与结构化分析](04-双阶段实体抽取与结构化分析.md) — Schema 字段含义
- [向量数据库设计与数据入库](05-向量数据库设计与数据入库.md) — SQL 完整定义
- [故障排查与常见问题](08-故障排查与常见问题.md) — 错误码对应排查
- [扩展开发与二次开发指南](10-扩展开发与二次开发指南.md) — 自定义错误码扩展