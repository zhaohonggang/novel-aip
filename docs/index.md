# Novel-AIP Technical Documentation

**Novel Automated Indexing & Profiling Platform** — Local-first, sovereign semantic novel discovery platform

---

## 🎯 Project Positioning

| Dimension | Description |
|-----------|-------------|
| **Core Capability** | 220k+ long-form novels → Structured profiles → 768-dim dense vectors → Hybrid retrieval (SQL + HNSW ANN) |
| **Deployment** | Fully local, zero external dependency, data never leaves premises |
| **Hardware Baseline** | AMD Strix Halo APU (gfx1151) / Ryzen AI 9 HX 370, Unified Memory |
| **Local LLM** | llama.cpp ROCm/HIP backend, Qwen2.5-27B-Instruct Q6_K, OpenAI-compatible API @ `http://127.0.0.1:8080/v1` |
| **Vector Model** | nomic-ai/nomic-embed-text-v1.5 (768-d, Matryoshka) |
| **Retrieval Latency** | P99 < 150 ms (60k+ document scale) |

---

## 📚 Documentation Navigation

| Chapter | Chinese | English | Core Content |
|---------|---------|---------|--------------|
| 01 | [项目概述与架构设计](../zh-CN/01-项目概述与架构设计.md) | [Project Overview & Architecture](01-Project-Overview-and-Architecture.md) | Business context, full architecture, tech choices, NFRs |
| 02 | [环境部署与配置指南](../zh-CN/02-环境部署与配置指南.md) | [Environment Setup & Configuration](02-Environment-Setup-and-Configuration.md) | Hardware/drivers, Python venv, PostgreSQL+pgvector, llama.cpp, env vars |
| 03 | [数据采集与预处理](../zh-CN/03-数据采集与预处理.md) | [Data Ingestion & Preprocessing](03-Data-Ingestion-and-Preprocessing.md) | `ingestion_core.py`, checkpoint state machine, multi-encoding fallback, incremental scan, resumable |
| 04 | [双阶段实体抽取与结构化分析](../zh-CN/04-双阶段实体抽取与结构化分析.md) | [Dual-Stage NER & LLM Profiling](04-Dual-Stage-NER-and-LLM-Profiling.md) | Stage 1 spaCy NER, Stage 2 LLM structuring, Prompt engineering, fallback |
| 05 | [向量数据库设计与数据入库](../zh-CN/05-向量数据库设计与数据入库.md) | [Vector Database Design & Data Loading](05-Vector-Database-Design-and-Data-Loading.md) | `schema.sql`, `novel_catalog`, GIN/HNSW, `database_worker.py` batch pipeline |
| 06 | [混合检索服务与 Web UI](../zh-CN/06-混合检索服务与-Web-UI.md) | [Hybrid Search Service & Web UI](06-Hybrid-Search-Service-and-Web-UI.md) | `app.py`, cosine similarity Top-K, Streamlit components, real-time sidebar |
| 07 | [运维监控与性能调优](../zh-CN/07-运维监控与性能调优.md) | [Operations, Monitoring & Performance Tuning](07-Operations-Monitoring-and-Performance-Tuning.md) | Key metrics, Grafana alerts, index maintenance, connection pooling, model warmup |
| 08 | [故障排查与常见问题](../zh-CN/08-故障排查与常见问题.md) | [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) | Categorized index, symptom→root cause→diagnosis→fix→prevention |
| 09 | [API 参考与接口规范](../zh-CN/09-API-参考与接口规范.md) | [API Reference & Specifications](09-API-Reference-and-Specifications.md) | Internal Python API, data models, error codes, request/response examples |
| 10 | [扩展开发与二次开发指南](../zh-CN/10-扩展开发与二次开发指南.md) | [Extension Development Guide](10-Extension-Development-Guide.md) | Plugin extension points, test strategy, CI/CD, contribution guide |

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/zhaohonggang/novel-aip.git
cd novel-aip

# 2. One-click environment init (requires Python 3.12+)
.\setup_env.ps1 -PythonExe "C:\Path\To\Python312\python.exe"

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Data ingestion (resumable)
python ingestion_core.py

# 5. Structured profiling (batch, calls local Qwen-27B)
python -m profiler_batch 12  # generates records.json

# 6. Vectorize & load into DB
python database_worker.py --payload records.json

# 7. Launch search service
streamlit run app.py  # http://localhost:8501
```

---

## 🏗️ System Architecture Overview

```mermaid
flowchart TD
    subgraph Source["📂 Source Data Layer"]
        SRC[(source_novels/ 619 .txt files<br/>~5.4 MB sample / Target 220k / 10 GB)]
    end

    subgraph Ingestion["🔄 Ingestion Layer (Phase 1)"]
        IC[ingestion_core.py<br/>CheckpointManager<br/>Multi-encoding fallback utf-8→gbk→gb18030<br/>tqdm progress bar]
        CP[(checkpoint.json<br/>File-level state machine<br/>PENDING/PROCESSED/FAILED)]
    end

    subgraph Profiling["🧠 Profiling Layer (Phase 2)"]
        SP[profiler.py<br/>Stage 1: spaCy zh_core_web_sm<br/>PERSON/GPE Top-5 frequency]
        LP[Stage 2: LLM Structuring<br/>Qwen-27B @ llama.cpp<br/>2k Intro + Chapter Overview<br/>NovelSchema JSON]
        RP[(records.json<br/>Structured Profile Batch Output)]
    end

    subgraph Storage["💾 Storage & Index Layer (Phase 3)"]
        PG[(PostgreSQL 18 + pgvector<br/>novel_catalog table<br/>VECTOR(768) + JSONB)]
        GI[GIN Index ×4<br/>genres/tropes/main_characters/key_locations]
        HI[HNSW Index<br/>summary_embedding<br/>m=16, ef_construction=64]
        DW[database_worker.py<br/>nomic-embed-text-v1.5<br/>Batch execute_values<br/>Retry/Circuit Breaker]
    end

    subgraph Search["🔍 Search Service Layer (Phase 4)"]
        APP[app.py Streamlit<br/>search_query: prefix vectorization<br/>Cosine similarity 1-(emb<=>q)<br/>Top-5 Card Rendering]
        SIDE[Sidebar Real-time Metrics<br/>Total/Embedded/Disk/Latency]
    end

    SRC --> IC
    IC --> CP
    CP --> SP
    SP --> LP
    LP --> RP
    RP --> DW
    DW --> PG
    PG --> GI
    PG --> HI
    PG --> APP
    APP --> SIDE

    classDef phase1 fill:#e3f2fd,stroke:#1976d2;
    classDef phase2 fill:#f3e5f5,stroke:#7b1fa2;
    classDef phase3 fill:#e8f5e9,stroke:#388e3c;
    classDef phase4 fill:#fff3e0,stroke:#f57c00;
    class IC,CP phase1
    class SP,LP,RP phase2
    class PG,GI,HI,DW phase3
    class APP,SIDE phase4
```

---

## 🔗 Key Links

- **GitHub Repository**: https://github.com/zhaohonggang/novel-aip
- **Online Documentation**: https://zhaohonggang.github.io/novel-aip/
- **Issue Tracker**: https://github.com/zhaohonggang/novel-aip/issues
- **Local Search Service**: http://localhost:8501 (after launch)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](../LICENSE) file for details.