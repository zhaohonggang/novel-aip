# 09 API Reference & Specifications

## 9.1 Internal Python API

### 9.1.1 `ingestion_core` Module

#### `CheckpointManager`
```python
class CheckpointManager:
    def __init__(self, path: Path | str = CHECKPOINT_PATH) -> None:
        """Initialize Checkpoint Manager.

        Args:
            path: Checkpoint File Path, Default ./checkpoint.json
        """

    def load(self) -> None:
        """Load Checkpoint State from Disk."""
        
    def save(self) -> None:
        """Atomically Write Checkpoint to Disk."""
        
    def register(self, file_path: Path) -> None:
        """Register New File as PENDING.

        Args:
            file_path: Absolute Path Path Object
        """
        
    def is_processed(self, file_path: Path) -> bool:
        """Check If File Already Processed.

        Returns:
            bool: True=Processed, False=Pending/Failed
        """
        
    def mark_processed(self, file_path: Path) -> None:
        """Mark File PROCESSED, Update Size/Mtime."""
        
    def mark_failed(self, file_path: Path, error: str) -> None:
        """Mark File FAILED, Record Error Traceback.

        Args:
            error: traceback.format_exception() String
        """
        
    @property
    def records(self) -> dict[str, dict[str, Any]]:
        """Get Read-Only Copy of All Records."""
```

#### `CheckpointStatus` Enum
```python
class CheckpointStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"
```

#### Core Functions
```python
def iter_text_files(root: Path) -> Iterator[Path]:
    """Iterate All .txt/.md Files Under Root.

    Yields:
        Path: Absolute Path
    """

def read_text_robust(file_path: Path) -> tuple[str, str]:
    """Multi-Encoding Fallback Text Read.

    Returns:
        tuple: (content, encoding_used)

    Raises:
        UnicodeDecodeError: All Encodings Failed
        OSError: Filesystem Error
    """

def run_ingestion(
    source_root: Path = SOURCE_ROOT,
    checkpoint_path: Path = CHECKPOINT_PATH
) -> int:
    """Execute Full Ingestion Pipeline.

    Returns:
        int: Failed File Count, 0 = All Success
    """
```

#### Constants
```python
SOURCE_ROOT = Path("./source_novels")
CHECKPOINT_PATH = Path("./checkpoint.json")
TEXT_EXTENSIONS = {".txt", ".md"}
ENCODING_FALLBACKS = ("utf-8", "gbk", "gb18030")
```

---

### 9.1.2 `profiler` Module

#### `NovelSchema` (Pydantic Model)
```python
class NovelSchema(BaseModel):
    main_genres: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Max 3 Main Genre Tags"
    )
    story_tropes: list[str] = Field(
        default_factory=list,
        description="Plot Tropes/Mechanisms, e.g.: System, Betrayal, Revenge"
    )
    one_sentence_summary: str = Field(
        default="",
        max_length=200,
        description="Single-Sentence Core Premise, ≤200 Chars"
    )
```

#### Core Functions
```python
def load_nlp(disable: Sequence[str] = ("parser", "tagger", "senter")) -> Any:
    """Load spaCy Chinese Model, Prune Pipeline.

    Args:
        disable: Disabled Pipeline Components

    Returns:
        spacy.Language: Loaded nlp Object
    """

def extract_top_entities(
    text: str,
    nlp: Any,
    head_chars: int = 50_000,
    top_n: int = 5
) -> tuple[list[str], list[str]]:
    """Stage 1: First N Chars NER Extract Top-K Person/Location.

    Returns:
        tuple: (top_persons, top_locations)
    """

def extract_chapter_headers(
    text: str,
    max_headers: int = 40
) -> list[str]:
    """Regex Extract Chapter Header Lines."""

def build_dense_snippet(
    text: str,
    intro_chars: int = 2_000,
    max_headers: int = 40
) -> str:
    """Assemble Dense Snippet: Intro + Chapter Overview."""

def build_llm_chain(
    llm: ChatOpenAI,
    schema: type[BaseModel] = NovelSchema
) -> Runnable:
    """Build LLM Structured Extraction Chain.

    Returns:
        Runnable: prompt | llm | JsonOutputParser
    """

def profile_text(
    text: str,
    nlp: Any,
    chain: Runnable
) -> NovelSchema:
    """Full Profiling Pipeline: NER + Snippet Assembly + LLM Extraction.

    Returns:
        NovelSchema: Structured Profile
    """

def fallback_schema(
    main_characters: list[str],
    key_locations: list[str]
) -> NovelSchema:
    """Deterministic Fallback Schema When LLM Unavailable."""

def build_llm(
    base_url: str = "http://127.0.0.1:8080/v1",
    model: str = "qwen2.5-27b-instruct",
    api_key: str = "not-needed-local-inference",
    temperature: float = 0.1
) -> ChatOpenAI:
    """Build Localized ChatOpenAI Client."""
```

#### Constants
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

#### CLI Interface
```bash
python profiler.py --file PATH [--host HOST] [--model MODEL] [--offline] [--log-level LEVEL]
```

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `--file` | Path | Required | Single Novel File Path |
| `--host` | str | `http://127.0.0.1:8080/v1` | LLM Service URL |
| `--model` | str | `qwen2.5-27b-instruct` | Model Name |
| `--offline` | flag | False | Fallback Only, No LLM Call |
| `--log-level` | str | `INFO` | Log Level |

---

### 9.1.3 `database_worker` Module

#### Core Functions
```python
def env_conn_info() -> dict[str, Any]:
    """Build DB Conn Params from Env Vars."""

def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load Embedding Model, Supports HF_TOKEN."""

def embed_summaries(
    embedder: SentenceTransformer,
    summaries: Sequence[str],
    batch_size: int = BATCH_SIZE,
    task_prefix: str = TASK_PREFIX_RETRIEVAL
) -> list[list[float]]:
    """Batch Embed Summaries.

    Args:
        task_prefix: "search_document: " (Ingestion) Or "search_query: " (Query)
    """

def format_vector(vector: Sequence[float]) -> str:
    """Vector → pgvector Text Literal '[0.1,0.2,...]'."""

def insert_chunk(
    records: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    conn_info: dict,
    batch_size: int = BATCH_SIZE
) -> int:
    """Single Batch Transaction Insert, Returns Inserted Rows."""

def process_records(
    records: list[dict],
    embedder: SentenceTransformer,
    conn_info: dict,
    batch_size: int = BATCH_SIZE,
    retry_limit: int = RETRY_LIMIT,
    retry_delay: float = RETRY_DELAY_SECONDS
) -> bool:
    """Full Load Pipeline: Embed → Batch → Transaction Write → Retry/Circuit Breaker.

    Returns:
        bool: True=All Success, False=Crash Flag Triggered
    """

def load_records(source: Path) -> list[dict]:
    """Load JSON Payload, Supports Array or {records: [...]} Format."""
```

#### Constants
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

#### CLI
```bash
python database_worker.py --payload records.json [--model MODEL] [--batch-size N] [--log-level LEVEL]
```

---

### 9.1.4 `app` Module (Streamlit Service)

#### Core Functions
```python
def db_conn_info() -> dict[str, Any]:
    """Build DB Conn Params from Env Vars."""

def format_vector(vector: Sequence[float]) -> str:
    """Vector Format, NaN/Inf Guard."""

@st.cache_resource
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load & Cache Embedding Model (TQDM_DISABLE=1)."""

def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    """Query Vectorization (search_query: Prefix)."""

def get_connection() -> psycopg2.connection:
    """Get DB Connection."""

def db_stats() -> dict[str, str]:
    """Sidebar Stats: total/embedded/size."""

def search_novels(query_vector: Sequence[float], limit: int = 5) -> tuple[list, float]:
    """Execute Hybrid Search, Returns (rows, latency_ms)."""

def render_result(row: tuple) -> None:
    """Render Single Result Card."""

def render_sidebar() -> None:
    """Render Sidebar Real-time Metrics."""

def main() -> None:
    """Streamlit Entry Point."""
```

#### Constants
```python
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
QUERY_TASK_PREFIX = "search_query: "
TOP_K = 5
SEARCH_PLACEHOLDER = "Describe the specific type of book narrative context you remember..."
```

---

## 9.2 Data Model Definitions

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

### 9.2.2 `checkpoint.json` Record
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

### 9.2.3 DB Table `novel_catalog`
```sql
-- See schema.sql Full Definition
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

## 9.3 SQL Query Interface

### 9.3.1 Hybrid Search (Core)
```sql
-- Params: $1 = query_vector_text ('[0.1,0.2,...]'), $2 = limit
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

### 9.3.2 Sidebar Stats
```sql
-- Total Count
SELECT COUNT(id) FROM novel_catalog;

-- Vector Coverage
SELECT COUNT(*) FROM novel_catalog WHERE summary_embedding IS NOT NULL;

-- Physical Disk Size
SELECT pg_size_pretty(pg_relation_size('novel_catalog'));
```

### 9.3.3 Index Maintenance
```sql
-- GIN Concurrent Rebuild (Non-Blocking)
REINDEX INDEX CONCURRENTLY idx_novel_tropes_gin;

-- HNSW Rebuild (Exclusive Lock)
REINDEX INDEX idx_novel_hnsw;

-- Runtime HNSW Search Precision
SET hnsw.ef_search = 200;
```

---

## 9.4 Environment Variables Spec

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_NAME` | ✅ | `fileindex` | DB Name |
| `DATABASE_USER` | ✅ | `postgres` | Username |
| `DATABASE_PASSWORD` | ✅ | `postgres` | Password |
| `DATABASE_HOST` | ✅ | `localhost` | Host |
| `DATABASE_PORT` | ✅ | `5432` | Port |
| `SECRET_KEY` | ✅ | — | JWT Signing Key |
| `APP_ENV` | ⭕ | `development` | Env Label |
| `LOG_LEVEL` | ⭕ | `INFO` | Log Level |
| `HF_TOKEN` | ⭕ | — | Hugging Face Token |
| `TQDM_DISABLE` | Internal | `1` | Disable tqdm (app.py Module Top Hardcoded) |

---

## 9.5 Error Code Specification

| Code | HTTP Status | Meaning | Typical Scenario |
|------|-------------|---------|------------------|
| `INGESTION_FAILED` | 500 | Ingestion Failed | Encoding Error, Permissions, Disk Full |
| `PROFILING_FAILED` | 500 | Profiling Failed | LLM Timeout, VRAM OOM, Prompt Parse Fail |
| `EMBEDDING_FAILED` | 500 | Embedding Failed | Model Load Fail, Dim Mismatch, VRAM OOM |
| `DATABASE_CONNECTION_FAILED` | 503 | DB Unavailable | Service Down, Pool Exhausted, Network Partition |
| `DATABASE_QUERY_FAILED` | 500 | SQL Execution Fail | Syntax Error, Type Mismatch, Constraint Violation |
| `INDEX_BUILD_FAILED` | 500 | Index Build Fail | OOM, Disk Full, Invalid Params |
| `MODEL_LOAD_FAILED` | 503 | Model Load Fail | Weights Missing, Version Mismatch, VRAM OOM |
| `VALIDATION_ERROR` | 400 | Input Validation Fail | Pydantic Validation Fail, Required Field Missing |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate Limit Hit | LLM Concurrency High / API Rate Limit |

### Error Response Format
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

## 9.6 Versioning & Compatibility

| Component | Versioning | Compatibility Promise |
|-----------|------------|----------------------|
| Python API | SemVer (MAJOR.MINOR.PATCH) | MINOR Compatible, MAJOR Breaking |
| DB Schema | Migration Scripts (Alembic/Manual) | Backward Compatible, New Cols Nullable |
| Vector Dim | Fixed 768 | Model Upgrade Requires Sync Migration |
| LLM Prompt | Versioned Constants | Breaking Change Requires MAJOR |
| REST API (Future) | OpenAPI 3.0 | Follows SemVer |

---

## 9.6 Related Documents

- [Dual-Stage NER & LLM Profiling](04-Dual-Stage-NER-and-LLM-Profiling.md) — Schema Field Meanings
- [Vector Database Design & Data Loading](05-Vector-Database-Design-and-Data-Loading.md) — Full SQL Definitions
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — Error Code → Debug Mapping
- [Extension Development Guide](10-Extension-Development-Guide.md) — Custom Error Code Extensions