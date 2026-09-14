# 10 Extension Development Guide

## 10.1 Extension Point Architecture

```mermaid
flowchart LR
    subgraph Core["Core Pipeline (Immutable)"]
        A[Ingestion] --> B[Profiler]
        B --> C[Embedding]
        C --> D[Indexing]
        D --> E[Search]
    end

    subgraph Ext["Extension Points (Pluggable)"]
        EP1[Encoding Detector]
        EP2[NER Pipeline]
        EP3[Prompt Template]
        EP4[Embedding Model]
        EP5[Retrieval Operator]
        EP6[UI Component]
        EP7[Auth/Authorization]
        EP8[Metrics Export]
    end

    EP1 -.-> A
    EP2 -.-> B
    EP3 -.-> B
    EP4 -.-> C
    EP5 -.-> E
    EP6 -.-> E
    EP7 -.-> E
    EP8 -.-> A & B & C & D & E
```

---

## 10.2 Extension Point Specifications

### 10.2.1 Encoding Detector (`EncodingDetector`)

```python
# novels/extensions/encoding_detector.py
from typing import Protocol
from pathlib import Path

class EncodingDetector(Protocol):
    def detect(self, file_path: Path) -> tuple[str, str]:
        """Returns (content, encoding) or Raises Exception."""

class DefaultEncodingDetector:
    """Default: utf-8 → gbk → gb18030"""
    
    FALLBACKS = ("utf-8", "gbk", "gb18030")
    
    def detect(self, file_path: Path) -> tuple[str, str]:
        for enc in self.FALLBACKS:
            try:
                return file_path.read_text(encoding=enc), enc
            except (UnicodeDecodeError, OSError):
                continue
        raise UnicodeDecodeError("All encodings failed")
```

**Registration**:
```python
# config/extensions.py
ENCODING_DETECTOR = "novels.extensions.encoding_detector.DefaultEncodingDetector"
```

### 10.2.2 NER Pipeline (`NERPipeline`)

```python
# novels/extensions/ner_pipeline.py
from typing import Protocol
from typing import list, tuple
from spacy.language import Language

class NERPipeline(Protocol):
    def load(self) -> Language: ...
    def extract(self, nlp: Language, text: str, head_chars: int) -> tuple[list[str], list[str]]: ...

class SpacyChineseNER:
    """Default spaCy zh_core_web_sm."""
    
    DISABLE = ["parser", "tagger", "senter"]
    KEEP_LABELS = ("PERSON", "GPE")
    HEAD_CHARS = 50_000
    TOP_N = 5
    
    def load(self) -> Language:
        import spacy
        return spacy.load("zh_core_web_sm", disable=self.DISABLE)
    
    def extract(self, nlp: Language, text: str, head_chars: int = None) -> tuple[list[str], list[str]]:
        from collections import Counter
        head = head_chars or self.HEAD_CHARS
        pc, gc = Counter(), Counter()
        doc = nlp(text[:head])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                pc[ent.text.strip()] += 1
            elif ent.label_ == "GPE":
                gc[ent.text.strip()] += 1
        return [k for k,_ in pc.most_common(self.TOP_N)], [k for k,_ in gc.most_common(self.TOP_N)]
```

**Extensions**: HanLP NER, LTP NER, BERT-based NER, Rule-based Dictionary NER.

### 10.2.3 Prompt Template (`PromptTemplate`)

```python
# novels/extensions/prompt_template.py
from typing import Protocol
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class PromptTemplate(Protocol):
    def build(self, schema: type[BaseModel]) -> ChatPromptTemplate: ...

class DefaultNovelPrompt:
    """Default Novel Profiling Prompt."""
    
    SYSTEM_TEMPLATE = """
    You are an Experienced Chinese Web Novel Editor...
    {format_instructions}
    """
    
    HUMAN_TEMPLATE = "Novel Snippet & Chapter Overview:\n\n{input}\n\nOutput JSON."
    
    def build(self, schema: type[BaseModel]) -> ChatPromptTemplate:
        from langchain_core.output_parsers import JsonOutputParser
        parser = JsonOutputParser(pydantic_object=schema)
        escaped = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
        system = self.SYSTEM_TEMPLATE.format(format_instructions=escaped)
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", self.HUMAN_TEMPLATE),
        ])
```

**Extensions**: Multi-Language Prompt, Domain-Specific (Xianxia/SciFi/Romance), Few-Shot, Chain-of-Thought.

### 10.2.4 Embedding Model (`EmbeddingModel`)

```python
# novels/extensions/embedding_model.py
from typing import Protocol, Sequence
from sentence_transformers import SentenceTransformer

class EmbeddingModel(Protocol):
    def encode(self, texts: list[str], **kwargs) -> list[list[float]]: ...
    def get_dimension(self) -> int: ...

class SentenceTransformerWrapper:
    """Default SentenceTransformer Wrapper."""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", **kwargs):
        self.model = SentenceTransformer(model_name, **kwargs)
        self.task_prefix_retrieval = "search_document: "
        self.task_prefix_query = "search_query: "
    
    def encode(self, texts: list[str], task: str = "retrieval", **kwargs) -> list[list[float]]:
        prefix = self.task_prefix_retrieval if task == "retrieval" else self.task_prefix_query
        texts = [prefix + t if t else t for t in texts]
        vecs = self.model.encode(texts, convert_to_numpy=True, **kwargs)
        return [list(map(float, row)) for row in vecs]
    
    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

**Extensions**: BGE-M3, E5-Large, OpenAI text-embedding-3-large, Custom Contrastive Model.

### 10.2.5 Retrieval Operator (`RetrievalOperator`)

```python
# novels/extensions/retrieval_operator.py
from typing import Protocol, Sequence
import psycopg2

class RetrievalOperator(Protocol):
    def search(
        self,
        conn: psycopg2.extensions.connection,
        query_vector: list[float],
        limit: int = 5,
        filters: dict | None = None
    ) -> list[tuple]: ...

class PgVectorCosineOperator:
    """Default pgvector Cosine Similarity Top-K."""
    
    def search(self, conn, query_vector, limit=5, filters=None):
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"
        sql = """
            SELECT title, author, main_characters, key_locations,
                   genres, tropes, one_liner_summary,
                   1 - (summary_embedding <=> %s::vector) AS similarity_score
            FROM novel_catalog
            WHERE summary_embedding IS NOT NULL
        """
        params = [f"[{','.join(f'{v:.8f}' for v in query_vector)}]"]
        if filters:
            # Dynamic WHERE Clause Builder
            pass
        sql += " ORDER BY similarity_score DESC LIMIT %s"
        with conn.cursor() as cur:
            cur.execute(sql, (*params, limit))
            return cur.fetchall()
```

**Extensions**: Hybrid Search (BM25 + Vector), Re-Ranking, MMR Diversity, KG-Augmented.

### 10.2.6 UI Component (`UIComponent`)

```python
# novels/extensions/ui_component.py
from typing import Protocol
import streamlit as st

class UIComponent(Protocol):
    def render_sidebar(self) -> None: ...
    def render_result(self, row: tuple) -> None: ...
    def render_error(self, exc: Exception) -> None: ...

class DefaultUI:
    def render_sidebar(self):
        # See app.py render_sidebar
        pass
    
    def render_result(self, row):
        # See app.py render_result
        pass
    
    def render_error(self, exc):
        import traceback
        st.error(f"Search Failed: {exc}\n\n```\n{traceback.format_exc()}\n```")
```

**Extensions**: Card/List/Table Toggle, Keyword Highlight, Export CSV/JSON, Bookmark/Rate/Share.

### 10.2.7 Auth/Authorization (`AuthProvider`)

```python
# novels/extensions/auth.py
from typing import Protocol, Optional

class AuthProvider(Protocol):
    def get_current_user(self) -> Optional[dict]: ...
    def require_auth(self) -> None: ...
    def logout(self) -> None: ...

class NoAuthProvider:
    """No Auth (Default)."""
    def get_current_user(self): return {"name": "local", "role": "admin"}
    def require_auth(self): pass
    def logout(self): pass

class OAuth2Provider:
    """OAuth2 / OIDC Integration."""
    def __init__(self, client_id, client_secret, issuer_url):
        pass
    # ...
```

### 10.2.8 Metrics Export (`MetricsExporter`)

```python
# novels/extensions/metrics.py
from typing import Protocol

class MetricsExporter(Protocol):
    def inc_counter(self, name: str, labels: dict = None) -> None: ...
    def observe_histogram(self, name: str, value: float, labels: dict = None) -> None: ...
    def set_gauge(self, name: str, value: float, labels: dict = None) -> None: ...

class PrometheusExporter:
    def __init__(self, port=9090):
        from prometheus_client import start_http_server, Counter, Histogram, Gauge
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        start_http_server(port)
    
    def inc_counter(self, name, labels=None):
        # ...
        pass
    # ...
```

---

## 10.3 Config-Driven Extension Loading

### 10.3.1 Config File (`config/extensions.yaml`)
```yaml
extensions:
  encoding_detector: "novels.extensions.encoding_detector.DefaultEncodingDetector"
  ner_pipeline: "novels.extensions.ner_pipeline.SpacyChineseNER"
  prompt_template: "novels.extensions.prompt_template.DefaultNovelPrompt"
  embedding_model: "novels.extensions.embedding_model.SentenceTransformerWrapper"
  retrieval_operator: "novels.extensions.retrieval_operator.PgVectorCosineOperator"
  ui_component: "novels.extensions.ui_component.DefaultUI"
  auth_provider: "novels.extensions.auth.NoAuthProvider"
  metrics_exporter: "novels.extensions.metrics.PrometheusExporter"

# Per-Extension Params
encoding_detector:
  fallbacks: ["utf-8", "gbk", "gb18030", "big5"]

ner_pipeline:
  disable: ["parser", "tagger", "senter"]
  keep_labels: ["PERSON", "GPE"]
  head_chars: 50000
  top_n: 5

embedding_model:
  model_name: "nomic-ai/nomic-embed-text-v1.5"
  task_prefix_retrieval: "search_document: "
  task_prefix_query: "search_query: "
  batch_size: 100
  device: "cuda"  # cpu/cuda/hip

retrieval_operator:
  default_limit: 5
  ef_search: 100
```

### 10.3.2 Extension Loader
```python
# novels/extensions/loader.py
import importlib
from typing import Type, Any

def load_extension(dotted_path: str, **kwargs) -> Any:
    """Dynamically Import & Instantiate Extension Class.

    Args:
        dotted_path: "package.module.ClassName"
        **kwargs: Constructor Args

    Returns:
        Instantiated Extension Object
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)

def load_all_extensions(config: dict) -> dict[str, Any]:
    """Batch Load All Extensions."""
    extensions = {}
    for key, dotted_path in config.get("extensions", {}).items():
        params = config.get(key, {})
        extensions[key] = load_extension(dotted_path, **params)
    return extensions
```

---

## 10.4 Testing Strategy

### 10.4.1 Test Pyramid
```
        ┌─────────────┐
        │  E2E (Few)  │  Critical Path: ingestion→profile→embed→search
        ├─────────────┤
        │ Integration │  Cross-Module: ingestion→profiler, worker→DB
        ├─────────────┤
        │ Unit (Many) │  Single Func/Class: format_vector, extract_entities, Prompt Render
        └─────────────┘
```

### 10.4.2 Unit Test Spec
```python
# tests/test_format_vector.py
import pytest
from app import format_vector
import math

def test_format_vector_basic():
    vec = [0.1, -0.2, 0.0]
    assert format_vector(vec) == "[0.10000000,-0.20000000,0.00000000]"

def test_format_vector_nan_inf():
    vec = [float('nan'), float('inf'), float('-inf'), 0.5]
    # NaN/Inf → 0.0
    result = format_vector(vec)
    assert "[0.00000000,0.00000000,0.00000000,0.50000000]" == result

def test_format_vector_empty():
    assert format_vector([]) == "[]"
```

### 10.3 Test Categories & Commands
```bash
# Unit (Fast, Parallel)
pytest tests/unit -xvs -n auto --tb=short

# Integration (Needs DB/LLM)
pytest tests/integration -xvs --tb=short -k "not slow"

# E2E (Full Pipeline, Marked Slow)
pytest tests/e2e -xvs --tb=long -k "slow"

# Coverage
pytest --cov=novels --cov-report=html --cov-fail-under=80
```

### 10.4 Test Fixtures
```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)

@pytest.fixture(scope="session")
def sample_text():
    return "第一章 觉醒\n主角严平在雷雨夜回到故乡..."

@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("zh_core_web_sm", disable=["parser", "tagger", "senter"])

@pytest.fixture(scope="module")
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")
```

---

## 10.4 CI/CD Pipeline

### 10.4.1 GitHub Actions (`.github/workflows/ci.yml`)
```yaml
name: CI

on: [push, pull_request]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg18
        env:
          POSTGRES_DB: fileindex
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U postgres"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - name: Install Deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov ruff
      - name: Lint
        run: ruff check .
      - name: Type Check
        run: mypy --strict .
      - name: Unit Tests
        run: pytest tests/unit -xvs --cov=novels --cov-fail-under=80
      - name: Integration Tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/fileindex
        run: pytest tests/integration -xvs --tb=short
      - name: Upload Coverage
        uses: codecov/codecov-action@v4

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin
      - run: mkdocs build --strict
      - uses: actions/upload-artifact@v4
        with: { name: site, path: site/ }
```

### 10.4.2 Docs Deploy (`.github/workflows/docs.yml`)
```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-mermaid2-plugin
      - run: mkdocs build --strict
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - uses: actions/deploy-pages@v4
```

---

## 10.4 Code Standards

| Tool | Config | Rules |
|------|--------|-------|
| **Ruff** | `ruff.toml` | `line-length=100`, `target-version=py312`, `select=["E","F","I","UP","B","C4","SIM","T20"]` |
| **MyPy** | `mypy.ini` | `strict=true`, `python_version=3.12`, `warn_return_any=true` |
| **Pre-commit** | `.pre-commit-config.yaml` | `ruff`, `mypy`, `end-of-file-fixer`, `trailing-whitespace` |

```bash
# Install
pre-commit install

# Manual Run
pre-commit run --all-files
```

---

## 10.5 Release Process

```mermaid
gitGraph
    commit id:"main"
    branch develop
    checkout develop
    commit id:"feat: new NER"
    commit id:"fix: embedding dim"
    branch release/v1.2.0
    checkout release/v1.2.0
    commit id:"chore: version bump"
    tag "v1.2.0"
    checkout main
    merge release/v1.2.0
    tag "v1.2.0"
    checkout develop
```

### Release Checklist
- [ ] All CI Pass
- [ ] Version Bump (`pyproject.toml` / `__version__`)
- [ ] `CHANGELOG.md` Updated
- [ ] Docs Build Pass (`mkdocs build --strict`)
- [ ] Docker Image Build/Push (Optional)
- [ ] GitHub Release Create + Assets
- [ ] Notify Downstream/Users

---

## 10.6 Contribution Guide

### 10.6.1 Branch Strategy
| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production Ready | PR + Review + CI Required |
| `develop` | Integration | Direct Push (Core Team) |
| `feature/*` | Feature Dev | Naming `feat/xxx` `fix/xxx` `refactor/xxx` |
| `release/v*` | Release Prep | Version Bump/Docs Only |

### 10.6.2 Commit Convention (Conventional Commits)
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

| Type | Meaning | Example |
|------|---------|---------|
| `feat` | New Feature | `feat(profiler): add HanLP NER pipeline` |
| `fix` | Bug Fix | `fix(worker): handle NaN in vector formatting` |
| `refactor` | Refactor (No Behavior Change) | `refactor(ingestion): extract encoding detector` |
| `perf` | Perf Optimization | `perf(worker): increase batch_size to 512` |
| `docs` | Doc Update | `docs: add troubleshooting FAQ` |
| `test` | Test Related | `test(unit): add format_vector edge cases` |
| `chore` | Build/Tool/Deps | `chore(deps): upgrade psycopg2 to 2.9.13` |

### 10.6.3 PR Template
```markdown
## Change Type
- [ ] feat / [ ] fix / [ ] refactor / [ ] perf / [ ] docs / [ ] test / [ ] chore

## Description
Brief What & Why.

## Related Issue
Closes #123

## Testing
- [ ] Unit Tests Pass
- [ ] Integration Tests Pass (Local/CI)
- [ ] Manual Verify: Steps/Screenshots

## Documentation
- [ ] Code Comments Updated
- [ ] Docs Synced (Corresponding .md)
- [ ] CHANGELOG.md Entry

## Deployment Impact
- [ ] None
- [ ] Needs DB Migration
- [ ] Needs New Env Var
- [ ] Needs Service Restart
```

---

## 10.6 License & Compliance

| Component | License | Compatible | Notes |
|-----------|---------|------------|-------|
| Project Code | MIT | ✅ | Permissive |
| spaCy `zh_core_web_sm` | MIT | ✅ | Includes Model Weights |
| llama.cpp | MIT | ✅ | Retain Copyright |
| Qwen2.5-27B | Tongyi Qianwen License | ⚠️ | Commercial Use Verify |
| nomic-embed-text-v1.5 | Apache 2.0 | ✅ | Retain NOTICE |
| PostgreSQL | PostgreSQL License | ✅ | BSD-Style |
| pgvector | MIT | ✅ | |
| Streamlit | Apache 2.0 | ✅ | |
| Transitive Deps | Per Package | ⚠️ | `pip-licenses` Periodic Audit |

```bash
# License Audit
pip install pip-licenses
pip-licenses --format=json --output-file=licenses.json
```

---

## 10.7 Related Documents

- [API Reference & Specifications](09-API-Reference-and-Specifications.md) — Extension Point Interfaces
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — Extension Failure Debug
- [CI/CD Config](../.github/workflows/) — Automation Pipelines