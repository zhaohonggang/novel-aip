# 04 Dual-Stage NER & LLM Profiling

## 4.1 Design Motivation: Asymmetric Compute Allocation

| Stage | Compute Profile | Hardware | Process Ratio | Output |
|-------|----------------|----------|---------------|--------|
| **Stage 1: CPU NER** | Rule/Statistical, High Throughput, Low Latency | CPU (All Cores) | 100% Docs | Top-5 Person/Location Freq |
| **Stage 2: LLM Structuring** | Inference-Heavy, High VRAM, High Latency | GPU (ROCm) | < 1% Docs (High-Density) | `NovelSchema` JSON |

**Key Insight**: Feeding Full Text Directly to LLM = High Cost, Context Window Overflow, High Hallucination. Only Invoke LLM on "High-Density Snippets" → 99%+ Inference Cost Reduction.

---

## 4.2 Stage 1: Fast CPU-NER (spaCy)

### Pipeline Pruning Strategy
```python
nlp = spacy.load("zh_core_web_sm", disable=["parser", "tagger", "senter"])
# Keep: tok2vec + ner + attribute_ruler
# Drop: parser(Dep Parse), tagger(POS), senter(Sent Boundary)
```
| Component | Kept? | Reason |
|-----------|-------|--------|
| `tok2vec` | ✅ | Embedding Layer, NER Depends |
| `ner` | ✅ | Core Entity Recognition |
| `attribute_ruler` | ✅ | Entity Normalization Rules |
| `parser` | ❌ | Dep Parse, NER Independent |
| `tagger` | ❌ | POS Tagging, NER Independent |
| `senter` | ❌ | Sentence Boundary, NER Independent |

### Entity Extraction & Frequency Counting
```python
def extract_top_entities(text, nlp, head_chars=50_000, top_n=5):
    person_ctr = Counter()
    gpe_ctr = Counter()
    doc = nlp(text[:head_chars])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_ctr[ent.text.strip()] += 1
        elif ent.label_ == "GPE":
            gpe_ctr[ent.text.strip()] += 1
    return person_ctr.most_common(top_n), gpe_ctr.most_common(top_n)
```
- **Slice Length**: 50,000 Chars ≈ 25k Chinese Chars ≈ First 3-5 Chapters
- **Entity Types**: `PERSON` (Characters), `GPE` (Geo-Political: Cities/Countries/Sects/Planets)
- **Output**: Top-5 Frequent Characters, Top-5 Frequent Locations

### spaCy Chinese Model Capability Bounds
| Entity Type | Coverage Examples | Limitations |
|-------------|-------------------|-------------|
| `PERSON` | "Yan Ping", "Lin Meimei", "Wu Zhongjun" | Only Explicit Names, Pronouns/Aliases Need LLM |
| `GPE` | "Guangdong", "Changsha", "Mojingren"(Mis-ID) | Sects/Planets/Fictional Places Often Missed, Need LLM |
| `ORG` | Not Enabled | Can Enable `ORG` for Gangs/Sects/Companies |

---

## 4.3 Stage 2: High-Density Context Assembly

### 4.3.1 Intro Slice
```python
INTRO_CHARS = 2_000  # ~1,000 Chinese Chars, Covers Opening Hook
intro = text[:INTRO_CHARS]
```
- 2,000 Chars ≈ Novel Opening "Hook" Paragraphs, Contains Core Premise, Protagonist Intro, Worldview Hook

### 4.3.2 Chapter Overview Extraction
```python
CHAPTER_HEADER_RE = re.compile(
    r"第[一二三四五六七八九十百千万〇零0-9]+[章节回卷集部篇段]"
)

def build_dense_snippet(text, intro_chars=2_000, max_headers=40):
    intro = text[:intro_chars].strip()
    chapter_lines = [
        line.strip() for line in text[:200_000].splitlines()
        if CHAPTER_HEADER_RE.search(line)
    ]
    # Dedup + Truncate
    headers = []
    seen = set()
    for line in chapter_lines:
        if line not in seen:
            seen.add(line)
            headers.append(line[:80])
            if len(headers) >= max_headers:
                break
    parts = [intro]
    if headers:
        parts.append("Chapter Overview: " + " / ".join(headers))
    return "\n\n".join(parts)
```
- **Regex Coverage**: Chinese Numerals (一/二/三), Arabic Numerals, Chapter/Section/Volume/Collection/Part/Episode/Segment
- **Scan Range**: First 200k Chars ≈ First 100k Chars ≈ Complete TOC Area
- **Dedup**: Avoid Duplicate Chapter Titles (e.g., "Chapter 1 xxx" Appears Multiple Times)
- **Output Format**: `Intro\n\nChapter Overview: Ch1 xxx / Ch2 yyy / ...`

### 4.3.3 Dense Snippet Example
```
[Intro First 2000 Chars]
...Protagonist Yan Ping Returns to Hometown in Stormy Night, Ancestral Mansion Emits Strange Sounds...

Chapter Overview: Chapter 1 Stormy Night / Chapter 2 Ancestral Mansion Sounds / Chapter 3 Daoist Ritual / Chapter 5 Supernatural Manifests / Chapter 10 Truth Revealed / ...
```

---

## 4.4 LLM Structured Extraction

### 4.4.1 Schema Definition (`NovelSchema`)
```python
class NovelSchema(BaseModel):
    main_genres: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Max 3 Main Genre Tags"
    )
    story_tropes: list[str] = Field(
        default_factory=list,
        description="Plot Tropes/Mechanisms, e.g.: System, Betrayal, Revenge, Transmigration, Rebirth"
    )
    one_sentence_summary: str = Field(
        default="",
        max_length=200,
        description="Single-Sentence Core Premise, ≤200 Chars"
    )
```

### 4.4.2 ChatOpenAI Local Config
```python
llm = ChatOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen2.5-27b-instruct",
    api_key="not-needed-local-inference",  # Arbitrary String
    temperature=0.1,        # Low Temp → Deterministic Convergence
    max_retries=2,
    request_timeout=300,    # 27B Inference Can Take 2-3 Minutes
)
```
| Param | Value | Reason |
|-------|-------|--------|
| `temperature` | 0.1 | Force Deterministic JSON, Suppress Hallucination |
| `max_retries` | 2 | Network Jitter Self-Heal |
| `request_timeout` | 300s | 27B Single Inference Can Take 2-3 Min |

### 4.4.3 Prompt Engineering
```python
system_content = f"""
You are an Experienced Chinese Web Novel Editor.
Task: Extract Structured Metadata from Novel Opening Snippet & Chapter Overview.
Constraints:
- main_genres ≤ 3 Genre Tags (e.g.: Xianxia/Urban/Suspense)
- story_tropes List Plot Tropes (e.g.: System/Betrayal/Revenge)
- one_sentence_summary ≤ 200 Chars, Single Sentence Core Premise
Must ONLY Output JSON Object Matching Schema, NO Other Text.

{escaped_format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_content),
    ("human", "Novel Snippet & Chapter Overview:\n\n{input}\n\nOutput JSON.")
])
chain = prompt | llm | JsonOutputParser(pydantic_object=NovelSchema)
```
**Critical Trick**: `format_instructions` Contains `{` `}` Must Double-Brace Escape `{{` `}}`, Else `ChatPromptTemplate` Misinterprets as Template Var → `INVALID_PROMPT_INPUT`.

---

## 4.5 Fallback Strategy (LLM Unavailable)

```python
def fallback_schema(main_characters, key_locations):
    genre_hints = {
        "Xianxia": ("Xiantian", "Cultivation", "Dantian", "Spiritual Root", "Realm", "Artifact", "Immortal"),
        "Urban": ("Company", "Group", "Urban", "Office", "CEO", "Mall"),
        "Historical": ("Emperor", "Court", "General", "World", "Dynasty", "Army"),
        "Suspense": ("Police", "Case", "Detective", "Locked Room", "Killer", "Clue"),
        "SciFi": ("Universe", "Alien", "Spaceship", "Tech", "Quantum", "Mecha"),
        "Romance": ("Tears", "Wedding", "Marriage", "Divorce", "Misunderstanding", "Heartbeat"),
    }
    text_sample = "".join(main_characters + key_locations)
    matched = [g for g, hints in genre_hints.items() if any(h in text_sample for h in hints)]
    return NovelSchema(
        main_genres=matched[:3],
        story_tropes=[],
        one_sentence_summary=f"Story Revolving Around {('、'.join(main_characters) or 'Protagonist')} of Unknown Genre."
    )
```
- **Keyword Heuristic**: Match Domain Words in Entity/Location Names to Genres
- **Safe Fallback**: Guarantees Schema Structure, Avoids Downstream Parse Crash
- **Trigger**: LLM Exception, Timeout, Empty Summary, JSON Parse Fail

---

## 4.6 Batch Driver (`batch_profile.py`)

### Core Logic
```python
for p in processed_files[:MAX_RECORDS]:
    text = read_robust(p)
    chars, locs = extract_top_entities(text, nlp)  # Stage 1
    schema = profile_text(text, nlp, chain)        # Stage 2
    records.append({
        "title": parse_title(p),
        "author": parse_author(p),
        "main_characters": chars,
        "key_locations": locs,
        "genres": schema.main_genres,
        "tropes": schema.story_tropes,
        "one_liner_summary": schema.one_liner_summary,
        "raw_json_payload": {"source_file": str(p.relative_to(ROOT))}
    })
```

### Metadata Parsing
```python
def parse_meta(path: Path):
    folder = path.parent.name
    # Dir Example: "【纹面】（完结） 作者：漂泊旅人[原创]"
    title = re.search(r"【([^】]+)】", folder)?.group(1)
    author = re.search(r"作者[:：]([^\]\[]+?)(?:\[|$)", folder)?.group(1)
    return title, author
```

---

## 4.7 Run & Output

```bash
# Single File Test (Online)
python profiler.py --file "source_novels/xxx.txt"

# Single File Test (Offline Fallback)
python profiler.py --file "source_novels/xxx.txt" --offline

# Batch Generate records.json (Default 6)
python batch_profile.py 12
```

### Output `records.json` Structure
```json
[
  {
    "title": "Wenmian",
    "author": "Piaobo Luren",
    "main_characters": ["Yan Ping", "Lin Meimei", "Huang Yue"],
    "key_locations": ["Guangdong", "Changsha", "Ancestral Mansion"],
    "genres": ["Suspense", "Supernatural", "Urban"],
    "tropes": ["Return to Hometown", "Haunted Ancestral Home", "Daoist Ritual", "Suspense Mystery"],
    "one_liner_summary": "Protagonist Yan Ping Returns to Hometown, Asks Childhood Friend About Haunted Mansion, Unravels Supernatural Mystery.",
    "raw_json_payload": {"source_file": "source_novels/.../xxx.txt"}
  }
]
```

---

## 4.8 Performance & Cost Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Stage 1 NER Time | ~0.3 s/Doc | spaCy CPU Single-Core |
| Stage 2 LLM Time | 90-180 s/Doc | Qwen-27B Q6_K @ ROCm |
| VRAM Usage | ~16 GB | Q6_K Quant + 8k Context |
| Batch Throughput | ~2-4 Docs/Min | Limited by LLM Serial Inference |
| Fallback Mode Time | < 0.5 s/Doc | Pure CPU Heuristic |

### Optimization Roadmap
| Direction | Expected Gain | Complexity |
|-----------|---------------|------------|
| Batch Prompt (10 Docs/Call) | 5-8x Throughput | Medium (Prompt Refactor for Array) |
| vLLM / TGI Replace llama.cpp | 3-5x Throughput | High (Needs CUDA/ROCm Recompile) |
| Quantize Q4_K_M (Smaller) | VRAM -30%, Speed +20% | Low (Swap Model File) |
| Cache Repeated Snippets | Depends on Duplication | Low (LRU Cache) |

---

## 4.8 Related Documents

- [Data Ingestion & Preprocessing](03-Data-Ingestion-and-Preprocessing.md) — Input Data Source
- [Vector Database Design & Data Loading](05-Vector-Database-Design-and-Data-Loading.md) — Output records.json Load
- [Troubleshooting & FAQ](08-Troubleshooting-and-FAQ.md) — LLM Timeout/Fallback/VRAM OOM