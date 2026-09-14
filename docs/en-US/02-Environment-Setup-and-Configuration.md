# 02 Environment Setup & Configuration

## 2.1 Hardware & OS Requirements

### Minimum Specs
| Component | Spec | Notes |
|-----------|------|-------|
| **CPU** | AMD Ryzen AI 9 HX 370 (Strix Halo, 12C/24T) | Unified memory, ROCm compute cores |
| **Memory** | 64 GB LPDDR5X-7500 (Unified) | Shared VRAM/RAM, recommend ≥48 GB |
| **Storage** | 1 TB NVMe SSD (PCIe 4.0) | Raw data + vector indexes + model weights |
| **GPU** | Integrated Radeon 890M (RDNA 3.5, 16 CU) | ROCm/HIP compute, no discrete GPU needed |
| **Network** | Gigabit Ethernet | Model downloads, HF mirrors |

### Recommended (Production)
| Component | Spec |
|-----------|------|
| Memory | 128 GB LPDDR5X |
| Storage | 2 TB NVMe (RAID 1) |
| Backup | Off-site object storage (S3-compatible) |

### Operating System
- **Windows 11 23H2+** (Primary dev env, PowerShell 5.1+)
- **Ubuntu 22.04/24.04 LTS** (Production recommended, systemd native)
- **WSL2 + Ubuntu** (Windows compromise for Linux stack)

---

## 2.2 Drivers & Runtime Installation

### AMD ROCm/HIP Driver (Windows)
```powershell
# 1. Install AMD Adrenalin Driver (includes ROCm runtime)
# Download: https://www.amd.com/en/support/download/drivers.html
# Select: Radeon 890M / Ryzen AI 300 Series

# 2. Verify ROCm
rocminfo | grep -E "Name|gfx"
# Should show: gfx1151 (RDNA 3.5)

# 3. Env Vars (persistent)
[Environment]::SetEnvironmentVariable("HSA_OVERRIDE_GFX_VERSION", "11.5.0", "Machine")
[Environment]::SetEnvironmentVariable("ROCM_PATH", "C:\Program Files\AMD\ROCm", "Machine")
```

### Python 3.12 Installation
```powershell
# Official: https://www.python.org/downloads/windows/
# Check: "Add Python to PATH"
# Verify
python --version  # Python 3.12.x
```

### PostgreSQL 18 + pgvector Compile Install
```powershell
# 1. Install PostgreSQL 18 (EDB Installer)
# https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
# Remember: Port 5432, Superuser postgres, Password postgres

# 2. Compile pgvector (needs Visual Studio 2022 + C++ Workload)
git clone --branch v0.8.6 --depth 1 https://github.com/pgvector/pgvector.git %TEMP%\pgvector
cd %TEMP%\pgvector
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PGROOT=C:\Program Files\PostgreSQL\18"
nmake /F Makefile.win
nmake /F Makefile.win install  # Needs Admin

# 3. Restart Service & Create Extension
net stop postgresql-x64-18
net start postgresql-x64-18
psql -U postgres -d fileindex -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

## 2.3 One-Click Environment Init Script

### `setup_env.ps1` Usage
```powershell
# Basic (auto-detect python in PATH)
.\setup_env.ps1

# Explicit Python 3.12 (Recommended)
.\setup_env.ps1 -PythonExe "C:\Users\xxx\AppData\Local\Programs\Python\Python312\python.exe"

# Parameters
param(
    [string]$PythonExe = 'python'  # Target python executable path
)
```

### Script Execution Flow
```mermaid
flowchart TD
    A[Check Python ≥3.11] --> B[Create .venv Virtual Env]
    B --> C[Upgrade pip]
    C --> D[pip install 7 Core Deps]
    D --> E[spacy download zh_core_web_sm]
    E --> F[Import Verify 7 Packages + spaCy Model]
    F --> G[Print Diagnostic Logs]
```

### Dependency Manifest
| Package | Version | Purpose |
|---------|---------|---------|
| `langchain-openai` | ≥1.6 | OpenAI-compatible client |
| `pydantic` | ≥2.13 | Data validation & Schema |
| `psycopg2-binary` | ≥2.9 | PostgreSQL Driver |
| `spacy` | ≥3.8 | Chinese NER |
| `tqdm` | ≥4.70 | Progress Bar |
| `streamlit` | ≥1.63 | Web UI |
| `sentence-transformers` | ≥6.0 | Embedding Model Inference |

---

## 2.4 Environment Variables (`.env`)

```ini
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/fileindex
DATABASE_NAME=fileindex
DATABASE_USER=postgres
DATABASE_PASSWORD=postgres
DATABASE_HOST=localhost
DATABASE_PORT=5432

# Application Settings
APP_ENV=development
SECRET_KEY=your-32-byte-hex-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Processing
MAX_FILE_SIZE_MB=100
SUPPORTED_FILE_TYPES=txt,md,pdf,docx,xlsx,pptx,epub,jpg,jpeg,png,gif,mp4,avi,mov,mp3,wav

# Search Settings
SEARCH_RESULTS_PER_PAGE=200
ENABLE_FULLTEXT_SEARCH=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Hugging Face (Optional, Accelerates Model Download)
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Key Variables
| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_*` | ✅ | PostgreSQL Connection |
| `SECRET_KEY` | ✅ | JWT Signing Key, **Must Change in Prod** |
| `HF_TOKEN` | ⭕ | Hugging Face Token, Accelerates nomic Model Download |
| `TQDM_DISABLE` | Internal | `app.py` Module Top Hardcoded `1`, Disables tqdm to Avoid Streamlit stderr Conflict |

---

## 2.5 Local LLM Service (llama.cpp)

### Prebuilt Binary
```bash
# Option A: Official Release (with ROCm Backend)
# https://github.com/ggerganov/llama.cpp/releases
# Download: llama-bXXXX-rocm-windows.zip

# Option B: Self-Compile (Needs CMake + ROCm SDK)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_ROCM=ON -DAMDGPU_TARGETS=gfx1151
cmake --build build --config Release -j$(nproc)
```

### Model Weights
```bash
# Qwen2.5-27B-Instruct Q6_K (~16 GB)
# Hugging Face: https://huggingface.co/Qwen/Qwen2.5-27B-Instruct-GGUF
# File: qwen2.5-27b-instruct-q6_k.gguf

# Placement
mkdir -p models
mv qwen2.5-27b-instruct-q6_k.gguf models/
```

### Launch Service
```powershell
# Windows PowerShell
$env:GGML_ROCM_FORCE_GEMM = "1"
.\llama-server.exe `
  -m models/qwen2.5-27b-instruct-q6_k.gguf `
  -c 8192 `          # Context Window
  -ngl 99 `          # All Layers Offloaded to GPU
  -ub 512 `          # User Batch Size
  -b 512 `           # Batch Size
  --port 8080 `
  --host 127.0.0.1 `
  --api-key "not-needed-local-inference"  # Arbitrary String
```

### Health Check
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -Method Get
# Should Return Model List JSON
```

---

## 2.6 Verification Checklist

| Step | Command | Expected |
|------|---------|----------|
| Python Version | `python --version` | `Python 3.12.x` |
| Virtual Env | `.\.venv\Scripts\Activate.ps1; python -c "import sys; print(sys.prefix)"` | Shows `.venv` Path |
| Core Packages Import | `python -c "import langchain_openai, pydantic, psycopg2, spacy, tqdm, streamlit, sentence_transformers; print('OK')"` | `OK` |
| spaCy Model | `python -c "import spacy; nlp=spacy.load('zh_core_web_sm'); print(nlp.meta['name'])"` | `zh_core_web_sm` |
| PostgreSQL Connect | `psql -U postgres -d fileindex -c "SELECT version();"` | PostgreSQL 18.x |
| pgvector Extension | `psql -U postgres -d fileindex -c "SELECT * FROM pg_extension WHERE extname='vector';"` | vector 0.8.6 |
| llama.cpp Service | `curl -s http://127.0.0.1:8080/v1/models \| jq` | Model List JSON |
| Disk Space | `Get-PSDrive C \| Select-Object Free` | > 50 GB Free |

---

## 2.7 Common Install Issues

| Symptom | Root Cause | Solution |
|---------|------------|----------|
| `setup_env.ps1` "Python 3.11+ required" | PATH python version too low | Explicit `-PythonExe "C:\...\Python312\python.exe"` |
| `spacy download` Hangs/Timeouts | Network HF Blocked/No Proxy | Set `HF_ENDPOINT=https://hf-mirror.com` or Pre-download `.whl` Offline Install |
| `pgvector` Compile Fail `Cannot open include file: 'postgres.h'` | `PGROOT` Wrong / Dev Package Missing | Verify PG Install Path, Usually `C:\Program Files\PostgreSQL\18` |
| `llama-server.exe` `GGML_ROCM` Error | Driver Version Mismatch / GPU Unsupported | Update AMD Adrenalin to 24.10.1+, Verify `rocminfo` Shows gfx1151 |
| `sentence-transformers` nomic 401/403 | Model Requires Accept Terms/Token | Set `HF_TOKEN` Env Var or `huggingface-cli login` |

---

## 2.8 Related Documents

- [Project Overview & Architecture](01-Project-Overview-and-Architecture.md)
- [Data Ingestion & Preprocessing](03-Data-Ingestion-and-Preprocessing.md) — Phase 1 Input Source
- [GitHub Actions CI/CD Config](../.github/workflows/docs-deploy.yml)