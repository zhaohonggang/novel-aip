from __future__ import annotations

import logging
import os
import time
from typing import Any, Sequence

import psycopg2
import streamlit as st
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("novel_aip.app")

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
QUERY_TASK_PREFIX = "search_query: "
TOP_K = 5

SEARCH_PLACEHOLDER = "Describe the specific type of book narrative context you remember..."


def db_conn_info() -> dict[str, Any]:
    return {
        "dbname": os.getenv("DATABASE_NAME", "fileindex"),
        "user": os.getenv("DATABASE_USER", "postgres"),
        "password": os.getenv("DATABASE_PASSWORD", "postgres"),
        "host": os.getenv("DATABASE_HOST", "localhost"),
        "port": int(os.getenv("DATABASE_PORT", "5432")),
        "connect_timeout": 10,
    }


def format_vector(vector: Sequence[float]) -> str:
    cleaned = [0.0 if (v != v or v == float('inf') or v == float('-inf')) else v for v in vector]
    return "[" + ",".join(f"{v:.8f}" for v in cleaned) + "]"


@st.cache_resource(show_spinner="Loading local embedding model...")
def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    kwargs: dict[str, Any] = {}
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    model = SentenceTransformer(model_name, **kwargs)
    if model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
        LOGGER.warning(
            "Model dimension %d differs from expected %d",
            model.get_sentence_embedding_dimension(),
            EMBEDDING_DIM,
        )
    return model


def embed_query(embedder: SentenceTransformer, query: str) -> list[float]:
    vector = embedder.encode([QUERY_TASK_PREFIX + query], convert_to_numpy=True)
    return list(map(float, vector[0]))


def get_connection() -> Any:
    return psycopg2.connect(**db_conn_info())


def db_stats() -> dict[str, str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(id) FROM novel_catalog")
            total = cur.fetchone()[0]
            cur.execute("SELECT pg_size_pretty(pg_relation_size('novel_catalog'))")
            size = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM novel_catalog WHERE summary_embedding IS NOT NULL")
            embedded = cur.fetchone()[0]
    return {"total": int(total), "embedded": int(embedded), "size": str(size)}


def search_novels(query_vector: Sequence[float], limit: int = TOP_K) -> list[tuple[Any, ...]]:
    sql = """
        SELECT
            title,
            author,
            main_characters,
            key_locations,
            genres,
            tropes,
            one_liner_summary,
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


def render_result(row: tuple[Any, ...]) -> None:
    (
        title,
        author,
        main_characters,
        key_locations,
        genres,
        tropes,
        summary,
        score,
    ) = row

    score_pct = float(score) * 100.0 if score is not None else 0.0

    header = f"**{title or 'Untitled'}**"
    if author:
        header += f" — *{author}*"
    header += f"　`similarity {score_pct:.1f}%`"

    with st.container(border=True):
        st.markdown(header)

        cols = st.columns(3)
        cols[0].markdown(f"**体裁**\n" + _tags(genres, "Unknown"))
        cols[1].markdown(f"**主要角色**\n" + _tags(main_characters, "—"))
        cols[2].markdown(f"**故事套路**\n" + _tags(tropes, "—"))

        if key_locations:
            st.markdown(f"**地点**: {', '.join(key_locations)}")
        if summary:
            st.markdown(f"> {summary}")


def _tags(values: Sequence[Any] | None, fallback: str) -> str:
    if not values:
        return fallback
    return " · ".join(f"`{v}`" for v in values)


def render_sidebar() -> None:
    st.sidebar.title("Novel-AIP")
    st.sidebar.caption("Local-first semantic novel discovery")

    try:
        stats = db_stats()
        st.sidebar.metric("Total Books Profiled", f"{stats['total']:,}")
        st.sidebar.metric("Embedded Vectors", f"{stats['embedded']:,}")
        st.sidebar.metric("Physical DB Space", stats["size"])
    except Exception as exc:
        st.sidebar.error(f"Database unavailable: {exc}")


def main() -> None:
    st.set_page_config(page_title="Novel-AIP", page_icon="📚", layout="wide")
    st.title("📚 Novel-AIP — 语义小说检索平台")
    st.caption(
        "实时混合检索：本地 nomic-embed-text-v1.5 向量化 + PostgreSQL pgvector 余弦相似度"
    )

    render_sidebar()

    query = st.text_input(
        "搜索您记忆中的小说情节",
        placeholder=SEARCH_PLACEHOLDER,
        label_visibility="collapsed",
    )
    search_clicked = st.button("🔍 检索", type="primary", use_container_width=True)

    if not (search_clicked and query and query.strip()):
        st.info("输入一段您记得的故事设定或情节，检索 Top 5 最相似的小说。")
        return

    try:
        embedder = load_embedder()
        with st.spinner("向量化查询..."):
            query_vector = embed_query(embedder, query.strip())

        with st.spinner("执行混合检索..."):
            rows, latency_ms = search_novels(query_vector)

        st.success(f"Query Latency: **{latency_ms:,.2f} ms** · 返回 {len(rows)} 条结果")
        if not rows:
            st.warning("未找到匹配结果，请尝试换一种描述。")
            return

        for row in rows:
            render_result(row)

    except Exception as exc:
        import traceback
        LOGGER.exception("Search pipeline failed")
        st.error(f"检索失败: {exc}\n\n```\n{traceback.format_exc()}\n```")


if __name__ == "__main__":
    main()