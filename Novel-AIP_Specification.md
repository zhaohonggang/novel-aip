# Novel-AIP: Deep Architecture & Implementation Specification
**Project Name:** Novel-AIP (Novel Automated Indexing & Profiling platform)  
**Target Architecture:** Forward Deployed Engineering (FDE) Production-Grade Blueprint  
**Hardware Profile:** AMD Strix Halo APU (`gfx1151`) / Ryzen AI 9 HX 370 with Unified VRAM/RAM  
**Local AI Base URL:** `http://127.0.0.1:8080/v1` (`llama.cpp` ROCm/HIP Backend running Qwen2.5-27B-Instruct Q6_K)  

---

## 🛠️ Section 1: Executive Summary & System Ontology
`Novel-AIP` is a localized, sovereign enterprise-grade data engineering framework designed to ingest, normalize, feature-profile, and hybrid-index high-volume, low-density non-structured narrative datasets (**220,000 text files / 10GB raw text**) without leaking data to external cloud APIs or exploding token compute budgets. 

Instead of routing massive full-text dumps directly to highly latent Large Language Models, `Novel-AIP` treats ingestion as an asymmetric multi-stage funnel. It relies on deterministic rule-based algorithms and local CPU-bound Named Entity Recognition (NER) for high-speed preprocessing, utilizing the reasoning power of a local 27B LLM strictly for strategic high-density structural extraction.

### Core Architecture Flow
```
[ 220,000 Raw Novel Files (10GB) ]
               │
               ▼  (Stage 1: OS Disk Scanner & Checkpoint Logger)
     [ Extraction Pipeline ] ──► [ checkpoint.json ] (State Persistence)
               │
               ├────────────────────────────────────────┐
               ▼ (Fast CPU Processing)                  ▼ (High-Density Assembly)
        [ spaCy Chinese NER ]                    [ Intro + Table of Contents ]
               │                                        │
               ▼                                        ▼
    [ High-Freq Character/Place ]             [ Bundled Batch Payload (10 Books) ]
               │                                        │
               │                                        ▼ (Local GPU Inference)
               │                                  [ Qwen-27B (llama.cpp ROCm) ]
               │                                        │
               │                                        ▼
               │                                [ Structured JSON Schema ]
               │                                (Genres, Tropes, One-Liner Summary)
               │                                        │
               │                                        ▼ (Sentence-Transformers)
               │                                [ Matryoshka Vector Embedding ]
               │                                        │
               └───────────────────┬────────────────────┘
                                   ▼
                      [ PostgreSQL + pgvector ]
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
       (B-Tree / GIN Index)                     (HNSW Graph Index)
   Structured Attributes & Tags              Dense Summary Vector
                ▲                                     ▲
                └──────────────────┬──────────────────┘
                                   │  (Hybrid Query: SQL + Cosine Distance)
                                   ▼
                       [ Streamlit User UI ]
```

---

## 📅 Section 2: Phase-by-Phase Technical Blueprint

### Phase 1: Infrastructure, Environment Automation & Crash-Resilient Ingestion Core
*   **FDE Target Objective:** Provision a locked, reproducible local Python execution ecosystem and map the raw OS file-system boundary. The ingestion script must handle non-standard text encodings (`UTF-8`, `GBK`, `GB18030`) and ensure **zero-data-loss checkpointing**. If the system loses power or encounters an unexpected OS file-lock, restarting the script must cause it to resume immediately from the exact last unindexed file buffer within milliseconds.
*   **Key Files to Generate:** `setup_env.ps1`, `ingestion_core.py`

### Phase 2: Dual-Stage Entity Extraction & Pydantic Profiling Pipeline
*   **FDE Target Objective:** Eradicate context pollution and optimize compute unit expenditure. This module executes **Stage 1 (Asymmetric Trimming)** by running a fast, localized CPU-bound Chinese tokenizer and Named Entity Recognition (NER) engine to analyze character arrays and spatial frequencies without any LLM compute cost. It then executes **Stage 2 (High-Density Context Alignment)** by extracting book front-matter and chapter lists, compiling them into a rigid prompt context window, and utilizing the local Qwen-27B server to synthesize the underlying metadata structure directly into an explicit JSON configuration file.
*   **Key Files to Generate:** `profiler.py`

### Phase 3: Relational Schema Design & Persistent Hybrid Vector Indexing
*   **FDE Target Objective:** Map the structured data array and dense embeddings into a centralized transactional datastore capable of handling hybrid keyword and proximity operations. The schema utilizes native `JSONB` for diagnostic serialization, alongside custom Postgres data array allocations. Rather than standard vector embeddings, the indexing tier incorporates **Matryoshka Representation Learning (MRL)** via local embedding models, storing dense array points inside an isolated `VECTOR` space optimized with an HNSW graph topography.
*   **Key Files to Generate:** `schema.sql`, `database_worker.py`

### Phase 4: Production-Grade Prototyping & Low-Latency UI Verification
*   **FDE Target Objective:** Surface the enterprise capabilities of the hybrid pipeline through a real-time analytics and discovery interface. The interface handles arbitrary human query statements, instantly projects them through the local embedding space, and executes a unified SQL retrieval command that combines strict metadata bitwise containment and floating-point cosine similarities. The UI must include diagnostic profiling tracking processing latencies, database size metrics, and memory utilization thresholds.
*   **Key Files to Generate:** `app.py`

---

## 🤖 Section 3: AI Code-Generation Execution Triggers

*Instructions for OpenCode AI: Read the following phase blocks independently. Do not provide high-level abstractions, partial wrappers, or summarized explanations. Generate production-grade, syntactically flawless Python 3.11+, PostgreSQL 16, and PowerShell scripts incorporating complete error catching, explicit type hints, and full structural implementations for each file requested.*

---

### 🚀 Trigger 1: Phase 1 Code Generation (Execute to Build Ingestion & Environment)
```markdown
Execute Phase 1 System Generation for "Novel-AIP".

Generate the following two files exactly as described:

1. `setup_env.ps1` (Windows PowerShell Script):
- Automatically verifies if a local Python 3.11+ executable is visible in the PATH variable.
- Initializes an isolated virtual environment named `.venv` in the local working directory.
- Programmatically updates pip and installs the explicit requirements array: `langchain-openai`, `pydantic`, `psycopg2-binary`, `spacy`, `tqdm`, `streamlit`, `sentence-transformers`.
- Invokes a native python sub-shell call to download the optimized language module: `python -m spacy download zh_core_web_sm`.
- Prints diagnostic setup logs verifying installation sanity.

2. `ingestion_core.py` (Robust Disk File Scanner & State Machine):
- Implement a thread-safe `CheckpointManager` class that manages an operations state array serialized into a local file path `./checkpoint.json`.
- The `checkpoint.json` must map every tracked file using its absolute string file path as the unique key. The payload attributes must include: `file_size_bytes` (int), `last_modified_timestamp` (float), `status` (string enum: ["PENDING", "PROCESSED", "FAILED"]), and `error_log` (string/null).
- Implement a core file-system crawler function using `os.walk` targeting a relative path folder `./source_novels`.
- The crawler loop must parse any text content file matching extensions `.txt` or `.md`.
- Implement robust multi-encoding fallback verification for the file reader loop: try opening via `utf-8`, fallback to `gbk`, fallback to `gb18030`, and gracefully capture errors by writing the trace output back to the checkpoint payload under a "FAILED" label if all reading passes drop frames or throw decode errors.
- Connect an industrial progress display engine using `tqdm` that dynamically outputs the calculated operational rates over the 220,000 text sequence.
```

---

### 🚀 Trigger 2: Phase 2 Code Generation (Execute to Build Natural Language Engine)
```markdown
Execute Phase 2 System Generation for "Novel-AIP".

Generate the following file exactly as described:

1. `profiler.py` (Dual-Stage NER and Structured LLM Profiling Middleware):
- Initialize a local `spacy.load("zh_core_web_sm")` runtime pointer. Disable unused components `["parser", "ner"]` where appropriate, or keep specific entity extractors alive to track `PERSON` (characters) and `GPE` (locations) types.
- Write a Stage 1 Python function that slices the first 50,000 characters of a text stream, runs tokenization, extracts all entities matching `PERSON` and `GPE`, uses a `collections.Counter` map to calculate frequencies, and returns the top 5 highest-ranking characters and locations as native string arrays.
- Write a structural processing function that clips a dense text block composed of the initial 2,000 characters (the introduction) combined with a regex-matched sequence capturing chapter headers (e.g., targeting strings matching "第[一二三四五六七八九十百千万0-9]+章").
- Use `pydantic` to define a clear validation schema class inheriting from `BaseModel` named `NovelSchema`. The schema structure must match:
  * `main_genres`: list[str] (Limited strictly to 3 dominant literary genres)
  * `story_tropes`: list[str] (List of tropes or plot mechanisms found, e.g., "System", "Betrayal", "Revenge")
  * `one_sentence_summary`: str (An explicit single-sentence summary of the overarching premise, max 200 characters)
- Instantiate an explicit `ChatOpenAI` wrapper from `langchain_openai` pointed directly at `http://127.0.0.1:8080/v1` with a fixed arbitrary string API key, explicit model variable string set to `qwen2.5-27b-instruct`, and temperature clamped hard to `0.1` to force structural deterministic convergence.
- Construct an explicit chain payload utilizing `JsonOutputParser(pydantic_object=NovelSchema)` and an authoritative System/User ChatPromptTemplate structure that forces the LLM to process the high-density snippet and output valid JSON aligning with the model schema definitions.
```

---

### 🚀 Trigger 3: Phase 3 Code Generation (Execute to Build Storage Core)
```markdown
Execute Phase 3 System Generation for "Novel-AIP".

Generate the following two files exactly as described:

1. `schema.sql` (Database Initialization Matrix):
- Include declarative commands to verify and initialize the target extension architecture: `CREATE EXTENSION IF NOT EXISTS vector;`.
- Construct a physical database table named `novel_catalog` featuring these explicit typing parameters:
  * `id`: SERIAL PRIMARY KEY
  * `title`: VARCHAR(512) NOT NULL
  * `author`: VARCHAR(256)
  * `main_characters`: TEXT[]
  * `key_locations`: TEXT[]
  * `genres`: TEXT[]
  * `tropes`: TEXT[]
  * `one_liner_summary`: TEXT
  * `summary_embedding`: VECTOR(768)
  * `raw_json_payload`: JSONB
  * `processed_at`: TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- Provision an authoritative GIN (Generalized Inverted Index) pointer targeting the `tropes` and `main_characters` arrays to allow swift string containment testing queries.
- Provision a highly scalable HNSW graph proximity allocation using Cosine Distance metric formatting:
  `CREATE INDEX idx_novel_hnsw ON novel_catalog USING hnsw (summary_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`

2. `database_worker.py` (Vector Embedding Engine & Batch Database Committer):
- Initialize a local `SentenceTransformer` inference pipeline using the explicit model asset string: `nomic-ai/nomic-embed-text-v1.5`. Use localized pooling configurations if necessary to clip or hold dimensions stable at 768.
- Construct a robust Python pipeline worker using `psycopg2` that opens a transactional connection array to a targeted PostgreSQL 16 service.
- The worker wrapper must process incoming lists of dictionary records synthesized from Phase 1 and Phase 2.
- For each record, pass the `one_sentence_summary` string payload into the local embedding model to produce a float array structure.
- Construct a fast, secure psycopg2 bulk transaction loop using `extras.execute_values()` to execute batch inserts (where `batch_size = 100`) directly into the `novel_catalog` table. 
- Implement an explicit try-except re-connection wrapper that catches database dropped states, sleeps for 5 seconds, and cleanly retries execution blocks 3 times before setting an operations crash exit flag.
```

---

### 🚀 Trigger 4: Phase 4 Code Generation (Execute to Build User Presentation Web App)
```markdown
Execute Phase 4 System Generation for "Novel-AIP".

Generate the following file exactly as described:

1. `app.py` (Streamlit Real-Time Hybrid Semantic Discovery App):
- Initialize a clean, premium Streamlit dashboard view.
- In the main app section, construct a standard `st.chat_input` or a wide semantic search query textbox displaying placeholder text: "Describe the specific type of book narrative context you remember...".
- Upon receiving a user search interaction string, pass the input variable into the same `sentence-transformers` `nomic` model instance loaded into memory to convert the human search string into a float array vector.
- Construct and execute a combined unified SQL retrieval operation against the local PostgreSQL backend:
  * The query must use the cosine operator `<=>` to calculate the direct semantic similarity between the user input query vector and the table storage column `summary_embedding`.
  * Calculate an algebraic score map: `similarity_score = (1 - (summary_embedding <=> %s))`.
  * Order the entire table array strictly by `similarity_score DESC` and return the Top 5 most proximate matching items.
- Render the retrieved list elements cleanly on the UI web grid using clear markdown layouts containing fields for: Title, Author, Primary Genre tags, High-Frequency Characters array, and the One-Sentence Core Summary.
- Implement an optimized sidebar status layout (`st.sidebar`) that executes asynchronous counts on the target table to output real-time server parameters:
  * "Total Books Profiled" (rendered via an atomic `COUNT(id)` execution).
  * "Physical DB Space consumption" (derived via standard PostgreSQL infrastructure inspection queries: `pg_size_pretty(pg_relation_size('novel_catalog'))`).
  * "Search Latency Counter" (using a high-precision `time.perf_counter()` wrap that isolates the database roundtrip and prints the result as "Query Latency: XX.XX ms").
```

---

## 💡 Operational Workflow for the OpenCode AI Engine
1. **Directory Setup:** Manually create a root folder named `Novel-AIP` and a subfolder inside it named `source_novels`. Put a small sample of text documents inside `source_novels` for validation.
2. **Environment Initialization:** Run **Trigger 1** inside the AI window. Save the outputs into the root folder and run `./setup_env.ps1` via a PowerShell terminal to lock down dependencies.
3. **Pipeline Construction:** Sequentially pass **Trigger 2** and **Trigger 3** to build the processing workers and initialize your local PostgreSQL schema.
4. **App Launch:** Run **Trigger 4** to generate the presentation dashboard, then launch your sovereign search platform using the command: `streamlit run app.py`.