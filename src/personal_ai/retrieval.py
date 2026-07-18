from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from personal_ai.config import AppConfig


SUPPORTED_KNOWLEDGE_SUFFIXES = {".json", ".jsonl", ".md", ".txt"}
VECTOR_SCHEMA_VERSION = "2"
RRF_K = 60
CURATED_KNOWLEDGE_MAX_SIMILARITY_GAP = 0.08
TELEGRAM_SESSION_GAP_SECONDS = 12 * 60 * 60
_RESIDENT_EMBEDDING_MODELS: dict[tuple[str, str], Any] = {}


def _json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            yield text
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_strings(item)


def _message_text(message: dict[str, Any]) -> str:
    text = message.get("text", "")
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in text
        ).strip()
    return ""


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split readable text into deterministic overlapping paragraph chunks."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [
            paragraph[start : start + max_chars]
            for start in range(0, len(paragraph), max(1, max_chars - overlap_chars))
        ]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _knowledge_documents(directory: Path) -> Iterator[tuple[str, str]]:
    if not directory.is_dir():
        return
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_KNOWLEDGE_SUFFIXES:
            continue
        if path.suffix.casefold() in {".json", ".jsonl"}:
            values: list[Any]
            if path.suffix.casefold() == ".jsonl":
                values = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                values = [json.loads(path.read_text(encoding="utf-8"))]
            text = "\n".join(part for value in values for part in _json_strings(value))
        else:
            text = path.read_text(encoding="utf-8")
        if text.strip():
            yield f"knowledge/{path.relative_to(directory).as_posix()}", text


def _telegram_documents(cleaned_path: Path) -> Iterator[tuple[str, str]]:
    if not cleaned_path.is_file():
        return
    cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
    for chat in cleaned.get("chats", []):
        if chat.get("type") != "personal_chat":
            continue
        chat_name = str(chat.get("name") or chat.get("id"))
        for session in chat.get("sessions", []):
            lines = [f"Чат: {chat_name}"]
            for message in session.get("messages", []):
                text = _message_text(message)
                if text:
                    speaker = str(message.get("from") or message.get("from_id") or "unknown")
                    lines.append(f"{speaker}: {text}")
            if len(lines) > 1:
                source = f"telegram/{chat.get('id')}/{session.get('session_id')}"
                yield source, "\n".join(lines)


def _raw_telegram_documents(export_path: Path) -> Iterator[tuple[str, str]]:
    """Yield session documents from every chat type in a Telegram Desktop export."""
    if not export_path.is_file():
        return
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    for chat in exported.get("chats", {}).get("list", []):
        chat_id = str(chat.get("id") or "unknown")
        chat_type = str(chat.get("type") or "unknown")
        chat_name = str(chat.get("name") or chat_id)
        session_index = 0
        yielded_document = False
        current_lines = [f"Тип чата: {chat_type}", f"Чат: {chat_name}"]
        previous_timestamp: int | None = None
        for message in chat.get("messages", []):
            timestamp_value = message.get("date_unixtime")
            timestamp = int(timestamp_value) if timestamp_value is not None else None
            if (
                timestamp is not None
                and previous_timestamp is not None
                and timestamp - previous_timestamp > TELEGRAM_SESSION_GAP_SECONDS
            ):
                if len(current_lines) > 2:
                    source = (
                        f"telegram/{chat_type}/{chat_id}/session_{session_index:04d}"
                    )
                    yield source, "\n".join(current_lines)
                    yielded_document = True
                    session_index += 1
                current_lines = [f"Тип чата: {chat_type}", f"Чат: {chat_name}"]
            text = _message_text(message)
            if text:
                speaker = str(message.get("from") or message.get("from_id") or "unknown")
                current_lines.append(f"{speaker}: {text}")
            if timestamp is not None:
                previous_timestamp = timestamp
        if len(current_lines) > 2:
            source = f"telegram/{chat_type}/{chat_id}/session_{session_index:04d}"
            yield source, "\n".join(current_lines)
            yielded_document = True
        if not yielded_document:
            source = f"telegram/{chat_type}/{chat_id}/session_0000"
            yield source, "\n".join(current_lines)


def _chunks(
    documents: Iterable[tuple[str, str]], max_chars: int, overlap_chars: int
) -> Iterator[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for source, text in documents:
        for index, chunk in enumerate(chunk_text(text, max_chars, overlap_chars)):
            item = (f"{source}#{index}", chunk)
            if item not in seen:
                seen.add(item)
                yield item


@lru_cache(maxsize=4)
def _embedding_model(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Vector RAG dependencies are missing. Run: pip install -e ."
        ) from exc
    return SentenceTransformer(model_name, device=device)


def _encode_documents(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    model = _embedding_model(model_name, device)
    embeddings = model.encode(
        [f"passage: {text}" for text in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) >= batch_size * 2,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_query(query: str, model_name: str, device: str) -> np.ndarray:
    model = _embedding_model(model_name, device)
    embedding = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embedding[0], dtype=np.float32)


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key, value FROM rag_metadata").fetchall())
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "RAG index uses the old keyword-only schema. Run: personal-ai build-rag."
        ) from exc


def preload_retrieval_embedding_model(database: Path) -> dict[str, str]:
    """Load and retain the embedding model declared by an existing RAG index."""
    if not database.is_file():
        raise FileNotFoundError(f"RAG index is missing: {database}. Run personal-ai build-rag.")
    with closing(sqlite3.connect(database)) as connection:
        metadata = _metadata(connection)
    model_name = metadata["embedding_model"]
    device = metadata.get("embedding_device", "cpu")
    model = _embedding_model(model_name, device)
    _RESIDENT_EMBEDDING_MODELS[(model_name, device)] = model
    return {
        "embedding_model": model_name,
        "embedding_device": device,
    }


def build_retrieval_index(config: AppConfig) -> dict[str, int]:
    """Build private dense-vector and keyword indexes from local personal knowledge."""
    documents = list(_knowledge_documents(config.retrieval.knowledge_dir))
    if config.retrieval.include_cleaned_telegram:
        raw_export = getattr(config.data, "source", None)
        if isinstance(raw_export, Path) and raw_export.is_file():
            documents.extend(_raw_telegram_documents(raw_export))
        else:
            documents.extend(_telegram_documents(config.data.cleaned))
    chunks = list(
        _chunks(
            documents,
            config.retrieval.chunk_chars,
            config.retrieval.chunk_overlap_chars,
        )
    )
    if not chunks:
        raise RuntimeError(
            "No RAG sources found. Add files under private_data/knowledge or run "
            "clean_telegram_export.py first."
        )

    embeddings = _encode_documents(
        [content for _, content in chunks],
        config.retrieval.embedding_model,
        config.retrieval.embedding_device,
        config.retrieval.embedding_batch_size,
    )
    if len(embeddings) != len(chunks) or embeddings.ndim != 2:
        raise RuntimeError("Embedding model returned an invalid vector matrix")

    database = config.retrieval.database
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_suffix(f"{database.suffix}.building")
    if temporary.exists():
        temporary.unlink()
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.execute(
                "CREATE VIRTUAL TABLE memory_chunks USING fts5("
                "source UNINDEXED, content, tokenize='unicode61')"
            )
            connection.execute(
                "CREATE TABLE chunk_vectors("
                "chunk_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE rag_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            indexed_chunks = [
                (index, source, content)
                for index, (source, content) in enumerate(chunks, start=1)
            ]
            connection.executemany(
                "INSERT INTO memory_chunks(rowid, source, content) VALUES (?, ?, ?)",
                indexed_chunks,
            )
            connection.executemany(
                "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
                (
                    (index, embeddings[index - 1].astype("<f4", copy=False).tobytes())
                    for index in range(1, len(chunks) + 1)
                ),
            )
            connection.executemany(
                "INSERT INTO rag_metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", VECTOR_SCHEMA_VERSION),
                    ("embedding_model", config.retrieval.embedding_model),
                    ("embedding_device", config.retrieval.embedding_device),
                    ("embedding_dimension", str(embeddings.shape[1])),
                    ("vector_weight", str(config.retrieval.vector_weight)),
                ),
            )
        os.replace(temporary, database)
    finally:
        if temporary.exists():
            temporary.unlink()
    _load_vector_index.cache_clear()
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dimensions": int(embeddings.shape[1]),
    }


@lru_cache(maxsize=8)
def _load_vector_index(
    database_string: str,
    modified_ns: int,
) -> tuple[list[int], list[str], list[str], np.ndarray, dict[str, str]]:
    del modified_ns
    database = Path(database_string)
    with closing(sqlite3.connect(database)) as connection:
        metadata = _metadata(connection)
        dimension = int(metadata["embedding_dimension"])
        rows = connection.execute(
            "SELECT v.chunk_id, m.source, m.content, v.embedding "
            "FROM chunk_vectors AS v "
            "JOIN memory_chunks AS m ON m.rowid = v.chunk_id "
            "ORDER BY v.chunk_id"
        ).fetchall()
    if not rows:
        raise RuntimeError("Vector RAG index contains no chunks")
    vectors = np.vstack(
        [np.frombuffer(blob, dtype="<f4", count=dimension) for _, _, _, blob in rows]
    )
    return (
        [int(row[0]) for row in rows],
        [str(row[1]) for row in rows],
        [str(row[2]) for row in rows],
        vectors,
        metadata,
    )


def _keyword_ranking(
    database: Path,
    query: str,
    limit: int,
) -> list[tuple[int, float]]:
    terms = re.findall(r"[\w-]+", query.casefold(), flags=re.UNICODE)
    if not terms:
        return []
    fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:16])
    with closing(sqlite3.connect(database)) as connection:
        return [
            (int(rowid), float(score))
            for rowid, score in connection.execute(
                "SELECT rowid, bm25(memory_chunks) AS score "
                "FROM memory_chunks WHERE memory_chunks MATCH ? "
                "ORDER BY score LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        ]


def search_retrieval(database: Path, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return semantic passages using dense vectors plus exact-keyword rank fusion."""
    if not database.is_file():
        raise FileNotFoundError(f"RAG index is missing: {database}. Run personal-ai build-rag.")
    query = query.strip()
    if not query:
        return []
    bounded_limit = max(1, min(int(limit), 10))
    chunk_ids, sources, contents, vectors, metadata = _load_vector_index(
        str(database.resolve()),
        database.stat().st_mtime_ns,
    )
    if metadata.get("schema_version") != VECTOR_SCHEMA_VERSION:
        raise RuntimeError("RAG vector schema is outdated. Run: personal-ai build-rag.")
    query_vector = _encode_query(
        query,
        metadata["embedding_model"],
        metadata.get("embedding_device", "cpu"),
    )
    if query_vector.shape[0] != vectors.shape[1]:
        raise RuntimeError("Query and document embedding dimensions do not match")

    similarities = vectors @ query_vector
    candidate_limit = min(len(chunk_ids), max(50, bounded_limit * 10))
    vector_order = np.argsort(-similarities)[:candidate_limit]
    vector_rank = {
        chunk_ids[int(index)]: rank
        for rank, index in enumerate(vector_order, start=1)
    }
    keyword_rows = _keyword_ranking(database, query, candidate_limit)
    keyword_rank = {
        chunk_id: rank for rank, (chunk_id, _) in enumerate(keyword_rows, start=1)
    }
    keyword_score = dict(keyword_rows)
    vector_weight = float(metadata.get("vector_weight", "0.75"))
    keyword_weight = 1.0 - vector_weight
    fused: dict[int, float] = {}
    for chunk_id in vector_rank.keys() | keyword_rank.keys():
        score = 0.0
        if chunk_id in vector_rank:
            score += vector_weight / (RRF_K + vector_rank[chunk_id])
        if chunk_id in keyword_rank:
            score += keyword_weight / (RRF_K + keyword_rank[chunk_id])
        fused[chunk_id] = score

    positions = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    fused_ids = sorted(fused, key=fused.get, reverse=True)
    ranked_ids = fused_ids[:bounded_limit]
    if bounded_limit >= 2:
        knowledge_positions = [
            index
            for index, source in enumerate(sources)
            if source.startswith("knowledge/")
        ]
        if knowledge_positions:
            best_similarity = float(similarities.max())
            knowledge_positions.sort(
                key=lambda index: float(similarities[index]),
                reverse=True,
            )
            knowledge_slots = min(2, bounded_limit // 2)
            reserved_knowledge_ids = [
                chunk_ids[index]
                for index in knowledge_positions
                if best_similarity - float(similarities[index])
                <= CURATED_KNOWLEDGE_MAX_SIMILARITY_GAP
            ][:knowledge_slots]
            for rank, chunk_id in enumerate(reserved_knowledge_ids, start=1):
                fused.setdefault(
                    chunk_id,
                    vector_weight / (RRF_K + rank),
                )
            ranked_ids = reserved_knowledge_ids + [
                chunk_id
                for chunk_id in fused_ids
                if chunk_id not in reserved_knowledge_ids
            ][: bounded_limit - len(reserved_knowledge_ids)]
    results: list[dict[str, Any]] = []
    for chunk_id in ranked_ids:
        position = positions[chunk_id]
        results.append(
            {
                "source": sources[position],
                "content": contents[position],
                "score": round(fused[chunk_id], 8),
                "vector_score": round(float(similarities[position]), 6),
                "keyword_score": (
                    round(keyword_score[chunk_id], 6)
                    if chunk_id in keyword_score
                    else None
                ),
                "retrieval": "hybrid_vector_keyword",
                "source_priority": (
                    "curated_knowledge"
                    if sources[position].startswith("knowledge/")
                    else "conversation_memory"
                ),
            }
        )
    return results
