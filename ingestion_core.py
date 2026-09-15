from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import traceback
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from tqdm import tqdm

LOGGER = logging.getLogger("novel_aip.ingestion")

SOURCE_ROOT = Path("./source_novels")
CHECKPOINT_PATH = Path("./checkpoint.json")
TEXT_EXTENSIONS = {".txt", ".md"}
ENCODING_FALLBACKS = ("utf-8", "gbk", "gb18030")


class CheckpointStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


VALID_STATUSES = {status.value for status in CheckpointStatus}


def _loggable_error(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class CheckpointManager:
    def __init__(self, path: Path | str = CHECKPOINT_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        with self._lock:
            if not self._path.exists():
                self._records = {}
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.error("Failed to load checkpoint %s: %s", self._path, exc)
                self._records = {}
                return
            self._records = {}
            loaded = 0
            for key, payload in raw.items():
                if not isinstance(key, str) or not isinstance(payload, dict):
                    continue
                if not self._is_valid_payload(payload):
                    continue
                self._records[key] = payload
                loaded += 1
            LOGGER.info("Loaded %d valid checkpoint record(s) from %s", loaded, self._path)

    @staticmethod
    def _is_valid_payload(payload: dict[str, Any]) -> bool:
        if payload.get("status") not in VALID_STATUSES:
            return False
        if not isinstance(payload.get("file_size_bytes"), int):
            return False
        if not isinstance(payload.get("last_modified_timestamp"), (int, float)):
            return False
        error_log = payload.get("error_log")
        if error_log is not None and isinstance(error_log, str):
            return True
        return error_log is None

    def save(self) -> None:
        with self._lock:
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(self._records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._path)

    def register(self, file_path: Path) -> None:
        resolved = str(file_path.resolve())
        with self._lock:
            if resolved in self._records:
                return
            stat = file_path.stat()
            self._records[resolved] = {
                "file_size_bytes": stat.st_size,
                "last_modified_timestamp": stat.st_mtime,
                "status": CheckpointStatus.PENDING.value,
                "error_log": None,
            }

    def is_processed(self, file_path: Path) -> bool:
        resolved = str(file_path.resolve())
        with self._lock:
            record = self._records.get(resolved)
            return record is not None and record["status"] == CheckpointStatus.PROCESSED.value

    def mark_processed(self, file_path: Path) -> None:
        resolved = str(file_path.resolve())
        with self._lock:
            stat = file_path.stat()
            self._records[resolved] = {
                "file_size_bytes": stat.st_size,
                "last_modified_timestamp": stat.st_mtime,
                "status": CheckpointStatus.PROCESSED.value,
                "error_log": None,
            }

    def mark_failed(self, file_path: Path, error: str) -> None:
        resolved = str(file_path.resolve())
        with self._lock:
            stat = file_path.stat()
            self._records[resolved] = {
                "file_size_bytes": stat.st_size,
                "last_modified_timestamp": stat.st_mtime,
                "status": CheckpointStatus.FAILED.value,
                "error_log": error,
            }

    @property
    def records(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._records)


def iter_text_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        LOGGER.warning("Source root does not exist: %s", root)
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if Path(filename).suffix.lower() in TEXT_EXTENSIONS:
                yield Path(dirpath) / filename


def read_text_robust(file_path: Path) -> tuple[str, str]:
    last_err: UnicodeDecodeError | OSError | None = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return file_path.read_text(encoding=encoding), encoding
        except (UnicodeDecodeError, OSError) as exc:
            last_err = exc
    if last_err is None:
        raise RuntimeError("No encodings attempted")
    raise last_err


def register_new_files(manager: CheckpointManager, files: list[Path]) -> None:
    for file_path in files:
        manager.register(file_path)


def process_file(manager: CheckpointManager, file_path: Path) -> None:
    if manager.is_processed(file_path):
        return
    try:
        text, _encoding = read_text_robust(file_path)
        if not text.strip():
            raise ValueError("Empty text content")
        manager.mark_processed(file_path)
    except Exception as exc:
        error = _loggable_error(exc)
        manager.mark_failed(file_path, error)
        LOGGER.error("Failed to ingest %s: %s", file_path, exc)


def run_ingestion(source_root: Path = SOURCE_ROOT, checkpoint_path: Path = CHECKPOINT_PATH) -> int:
    manager = CheckpointManager(checkpoint_path)
    manager.load()

    LOGGER.info("Scanning text files under %s", source_root)
    files = sorted(iter_text_files(source_root))
    LOGGER.info("Discovered %d candidate file(s)", len(files))

    register_new_files(manager, files)
    manager.save()

    pending = [f for f in files if not manager.is_processed(f)]
    LOGGER.info("%d file(s) pending processing", len(pending))

    processed = 0
    failed = 0
    with tqdm(total=len(files), desc="Ingesting", unit="file", ncols=100) as progress:
        for file_path in files:
            if manager.is_processed(file_path):
                progress.update(1)
                continue
            try:
                process_file(manager, file_path)
                record = manager.records.get(str(file_path.resolve()))
                if record is not None and record["status"] == CheckpointStatus.FAILED.value:
                    failed += 1
                else:
                    processed += 1
            finally:
                progress.update(1)
                progress.set_postfix({"processed": processed, "failed": failed})

    manager.save()

    summary = {
        "total_discovered": len(files),
        "processed": processed,
        "failed": failed,
        "skipped_already_processed": len(files) - len(pending),
    }
    LOGGER.info("Ingestion complete: %s", json.dumps(summary, ensure_ascii=False))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel-AIP crash-resilient disk ingestion core")
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_ROOT,
        help="Root folder to crawl for .txt/.md files (default: ./source_novels)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to checkpoint.json (default: ./checkpoint.json)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    failed = run_ingestion(source_root=args.source, checkpoint_path=args.checkpoint)
    if failed:
        raise SystemExit(f"Ingestion finished with {failed} failed file(s). Review checkpoint.json error_log entries.")


if __name__ == "__main__":
    main()