from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

LOGGER = logging.getLogger("novel_aip.profiler")

DEFAULT_HOST = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen2.5-27b-instruct"
DEFAULT_API_KEY = "not-needed-local-inference"
DEFAULT_TEMPERATURE = 0.1

HEAD_SLICE_CHARS = 50_000
INTRO_CHARS = 2_000
MAX_CHAPTER_HEADERS = 40

CHAPTER_HEADER_RE = re.compile(r"第[一二三四五六七八九十百千万〇零0-9]+[章节回卷集部篇段]")

_KEEP_LABELS = ("PERSON", "GPE")


class NovelSchema(BaseModel):
    main_genres: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="3 dominant literary genres at most",
    )
    story_tropes: list[str] = Field(
        default_factory=list,
        description="Tropes or plot mechanisms, e.g. System, Betrayal, Revenge",
    )
    one_sentence_summary: str = Field(
        default="",
        max_length=200,
        description="Single-sentence premise summary, max 200 characters",
    )


def load_nlp(*, disable: Sequence[str] = ("parser", "tagger", "senter")) -> Any:
    import spacy

    return spacy.load("zh_core_web_sm", disable=list(disable))


def extract_top_entities(
    text: str,
    nlp: Any,
    *,
    head_chars: int = HEAD_SLICE_CHARS,
    top_n: int = 5,
) -> tuple[list[str], list[str]]:
    person_counter: Counter[str] = Counter()
    gpe_counter: Counter[str] = Counter()
    doc = nlp(text[:head_chars])
    for ent in doc.ents:
        label = ent.label_
        if label not in _KEEP_LABELS:
            continue
        name = ent.text.strip()
        if not name:
            continue
        if label == "PERSON":
            person_counter[name] += 1
        elif label == "GPE":
            gpe_counter[name] += 1
    top_person = [name for name, _count in person_counter.most_common(top_n)]
    top_gpe = [name for name, _count in gpe_counter.most_common(top_n)]
    return top_person, top_gpe


def extract_chapter_headers(
    text: str,
    *,
    max_headers: int = MAX_CHAPTER_HEADERS,
) -> list[str]:
    headers = []
    seen: set[str] = set()
    for match in CHAPTER_HEADER_RE.finditer(text):
        header = match.group(0)
        if header in seen:
            continue
        seen.add(header)
        headers.append(header)
        if len(headers) >= max_headers:
            break
    return headers


def build_dense_snippet(
    text: str,
    *,
    intro_chars: int = INTRO_CHARS,
    max_headers: int = MAX_CHAPTER_HEADERS,
) -> str:
    intro = text[:intro_chars].strip()
    chapter_lines = [line.strip() for line in text[:200_000].splitlines() if CHAPTER_HEADER_RE.search(line)]
    headers = []
    seen: set[str] = set()
    for line in chapter_lines:
        if line in seen:
            continue
        seen.add(line)
        headers.append(line[:80])
        if len(headers) >= max_headers:
            break
    parts = [intro]
    if headers:
        parts.append("章节概览: " + " / ".join(headers))
    return "\n\n".join(parts)


def build_llm_chain(
    llm: ChatOpenAI,
    schema: type[BaseModel] = NovelSchema,
):
    output_parser = JsonOutputParser(pydantic_object=schema)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个经验丰富的中文网络小说编辑。你的任务是严格根据用户提供的"
                "小说开头片段与章节概览，抽取结构化元数据。\n"
                "你必须只输出符合以下JSON Schema的JSON对象，不要输出任何其他文字。\n"
                "约束：main_genres 最多3个文学体裁标签；story_tropes 给出情节套路（如"
                "\"系统流\",\"背叛\",\"复仇\"）；one_sentence_summary 用一句话概括核心"
                "设定与总体故事前提，不得超过200个汉字。\n\n{format_instructions}",
            ),
            (
                "human",
                "以下是小说的前序文本片段与章节概览：\n\n{input}\n\n"
                "请输出符合JSON Schema的结构化结果。",
            ),
        ]
    )
    return prompt | llm | output_parser


def profile_text(
    text: str,
    nlp: Any,
    chain: Any,
) -> NovelSchema:
    main_characters, key_locations = extract_top_entities(text, nlp)
    snippet = build_dense_snippet(text)
    try:
        raw = chain.invoke({"input": snippet})
        parsed = NovelSchema.model_validate(raw)
    except Exception as exc:
        LOGGER.warning("LLM profiling failed, using deterministic fallback: %s", exc)
        return fallback_schema(main_characters, key_locations)
    if not parsed.one_sentence_summary.strip():
        return fallback_schema(main_characters, key_locations)
    return parsed


def fallback_schema(
    main_characters: list[str],
    key_locations: list[str],
) -> NovelSchema:
    genre_hints = {
        "玄幻": ("先天", "修炼", "丹田", "灵根", "境界", "法宝", "仙"),
        "都市": ("公司", "集团", "都市", "上班", "总裁", "商场"),
        "历史": ("皇帝", "朝廷", "将军", "天下", "王朝", "兵马"),
        "悬疑": ("警方", "案件", "侦探", "密室", "凶手", "线索"),
        "科幻": ("宇宙", "外星", "飞船", "科技", "量子", "机甲"),
        "言情": ("眼泪", "婚礼", "结婚", "离婚", "误会", "心动"),
    }
    text_sample = "".join(main_characters + key_locations)
    matched = [genre for genre, hints in genre_hints.items() if any(h in text_sample for h in hints)]
    return NovelSchema(
        main_genres=matched[:3],
        story_tropes=[],
        one_sentence_summary=f"以{('、'.join(main_characters) or '主角')}为中心展开的未知类型故事。",
    )


def build_llm(
    *,
    base_url: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
        request_timeout=300,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel-AIP dual-stage NER + LLM profiler")
    parser.add_argument("--file", type=Path, required=True, help="Path to a .txt/.md novel file")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name on the server")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip LLM call and emit deterministic fallback schema",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file}")

    text = _read_any(args.file)
    nlp = load_nlp()

    if args.offline:
        main_characters, key_locations = extract_top_entities(text, nlp)
        schema = fallback_schema(main_characters, key_locations)
    else:
        llm = build_llm(base_url=args.host, model=args.model)
        chain = build_llm_chain(llm)
        schema = profile_text(text, nlp, chain)
        schema.main_genres = schema.main_genres[:3]

    print(schema.model_dump_json(indent=2))


def _read_any(file_path: Path) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8", "gbk", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to decode {file_path}: {last_error}") from last_error


if __name__ == "__main__":
    main()