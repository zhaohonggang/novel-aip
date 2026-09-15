from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import extras, sql
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("novel_aip.database_worker")

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
BATCH_SIZE = 100
RETRY_LIMIT = 3
RETRY_DELAY_SECONDS = 5

TASK_PREFIX_RETRIEVAL = "search_document: "
TASK_PREFIX_QUERY = "search_query: "

INSERT_COLUMNS = (
    "title",
    "author",
    "main_characters",
    "key_locations",
    "genres",
    "tropes",
    "one_liner_summary",
    "summary_embedding",
    "raw_json_payload",
)


def env_conn_info() -> dict[str, Any]:
    return {
        "dbname": os.getenv("DATABASE_NAME", "fileindex"),
        "user": os.getenv("DATABASE_USER", "postgres"),
        "password": os.getenv("DATABASE_PASSWORD", "postgres"),
        "host": os.getenv("DATABASE_HOST", "localhost"),
        "port": int(os.getenv("DATABASE_PORT", "5432")),
        "connect_timeout": 10,
    }


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


def embed_summaries(
    embedder: SentenceTransformer,
    summaries: Sequence[str],
    *,
    batch_size: int = BATCH_SIZE,
    task_prefix: str = TASK_PREFIX_RETRIEVAL,
) -> list[list[float]]:
    texts = [task_prefix + s if s else s for s in summaries]
    vectors = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return [list(map(float, row)) for row in vectors]


def format_vector(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


def _connect(conn_info: dict[str, Any]) -> Any:
    return psycopg2.connect(**conn_info)


def insert_chunk(
    records: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    conn_info: dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    conn = _connect(conn_info)
    try:
        with conn.cursor() as cur:
            insert_query = sql.SQL(
                "INSERT INTO novel_catalog ({cols}) VALUES %s"
            ).format(
                cols=sql.SQL(", ").join(sql.Identifier(col) for col in INSERT_COLUMNS)
            )
            row_template = sql.SQL(
                "(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)"
            ).as_string(conn)
            rows = []
            for record, vector in zip(records, embeddings, strict=True):
                rows.append(
                    (
                        record["title"],
                        record.get("author"),
                        record.get("main_characters"),
                        record.get("key_locations"),
                        record.get("genres"),
                        record.get("tropes"),
                        record.get("one_liner_summary"),
                        format_vector(vector),
                        json.dumps(record.get("raw_json_payload") or {}, ensure_ascii=False),
                    )
                )
            extras.execute_values(
                cur,
                insert_query,
                rows,
                template=row_template,
                page_size=batch_size,
            )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def process_records(
    records: list[dict[str, Any]],
    embedder: SentenceTransformer,
    conn_info: dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    retry_limit: int = RETRY_LIMIT,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> bool:
    summaries = [str(r.get("one_liner_summary") or "") for r in records]
    LOGGER.info("Embedding %d summary payload(s)", len(summaries))
    embeddings = embed_summaries(embedder, summaries, batch_size=batch_size)

    total_inserted = 0
    crash_flag = False
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        chunk_vectors = embeddings[start : start + batch_size]
        attempt = 0
        while True:
            try:
                inserted = insert_chunk(chunk, chunk_vectors, conn_info, batch_size=batch_size)
                total_inserted += inserted
                LOGGER.info("Inserted chunk of %d record(s) (cumulative %d)", inserted, total_inserted)
                break
            except psycopg2.OperationalError as exc:
                attempt += 1
                LOGGER.warning("Database connection failure (%s); attempt %d/%d", exc, attempt, retry_limit)
                if attempt >= retry_limit:
                    LOGGER.error("Crash exit: %d consecutive database failures on chunk %d", attempt, start)
                    crash_flag = True
                    return False
                time.sleep(retry_delay)
            except Exception as exc:
                LOGGER.error("Non-recoverable insert failure on chunk %d: %s", start, exc)
                crash_flag = True
                return False

    LOGGER.info("Batch insert complete: %d record(s) total", total_inserted)
    return not crash_flag


def load_records(source: Path) -> list[dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("records")
        if records is None:
            raise ValueError("JSON object must contain a 'records' array")
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("Payload must be a JSON array of record objects")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Novel-AIP vector embedding engine & batch database committer"
    )
    parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="JSON file containing an array of records or {records: [...]}",
    )
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="SentenceTransformer model id")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    records = load_records(args.payload)
    LOGGER.info("Loaded %d record(s) from %s", len(records), args.payload)

    embedder = load_embedder(args.model)
    ok = process_records(
        records,
        embedder,
        env_conn_info(),
        batch_size=args.batch_size,
    )
    if not ok:
        raise SystemExit("Database worker crashed after exhausting connection retries.")


if __name__ == "__main__":
    main()