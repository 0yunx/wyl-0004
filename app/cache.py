"""Deduplication cache using SQLite and SHA-256 content hashing.

Stores pre-computed chunk embeddings keyed by the SHA-256 hash of the
full document text so that identical documents skip the embedding API
call entirely, saving tokens and API quota.

Schema
------
``dedupe_cache`` table columns
    hash          PRIMARY KEY  hex-encoded SHA-256 of the document text
    chunks_json   TEXT         JSON array of chunk strings
    vectors_json  TEXT         JSON 2-D array of embedding vectors (list of lists)
    vector_dim    INTEGER      dimensionality of each embedding vector
    chunk_count   INTEGER      number of chunks (equal to len(chunks_json))
    created_at    TEXT         ISO-8601 timestamp of first insertion
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np

from .config import settings


class DeduplicationCache:
    """SQLite-backed cache for document chunk embeddings.

    The cache is keyed by the SHA-256 hex digest of the *parsed plain
    text* of a document (i.e. after PDF/TXT extraction, but before
    chunking).  Two byte-identical files will always produce the same
    key; two files with identical textual content but different binary
    representations (e.g. the same text saved with UTF-8 vs UTF-8-BOM)
    will *not* share a cache entry because the raw bytes differ — the
    parsed text is used instead.
    """

    TABLE = "dedupe_cache"

    def __init__(self, db_path: str = None) -> None:
        self.db_path = db_path or settings.CACHE_DB_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Schema ───────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE} (
                hash          TEXT PRIMARY KEY,
                chunks_json   TEXT NOT NULL,
                vectors_json  TEXT NOT NULL,
                vector_dim    INTEGER NOT NULL,
                chunk_count   INTEGER NOT NULL,
                created_at    TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # ── Hashing ──────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(text: str) -> str:
        """Return the hex SHA-256 digest of *text* (UTF-8 encoded)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ── Lookup ───────────────────────────────────────────────────────

    def lookup(self, text: str) -> Optional[Tuple[List[str], List[np.ndarray]]]:
        """Look up cached chunks + vectors for *text*.

        Args:
            text: Full parsed document text.

        Returns:
            ``(chunks_text, chunk_vectors)`` on cache hit, or ``None`` on miss.
            ``chunk_vectors`` is a list of 1-D float32 NumPy arrays.
        """
        digest = self.compute_hash(text)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT chunks_json, vectors_json FROM {self.TABLE} WHERE hash = ?",
            (digest,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        chunks_text = json.loads(row["chunks_json"])
        vectors_raw = json.loads(row["vectors_json"])
        chunk_vectors = [np.array(v, dtype=np.float32) for v in vectors_raw]
        return chunks_text, chunk_vectors

    # ── Insertion ────────────────────────────────────────────────────

    def store(
        self,
        text: str,
        chunks_text: List[str],
        chunk_vectors: List[np.ndarray],
    ) -> None:
        """Persist *chunks_text* and *chunk_vectors* keyed by *text*.

        If the hash already exists this is a no-op (the first cached
        result wins).
        """
        digest = self.compute_hash(text)
        vectors_raw = [v.tolist() for v in chunk_vectors]
        dim = len(chunk_vectors[0]) if chunk_vectors else 0
        now = datetime.now(timezone.utc).isoformat()

        cur = self._conn.cursor()
        cur.execute(
            f"""
            INSERT OR IGNORE INTO {self.TABLE}
                (hash, chunks_json, vectors_json, vector_dim, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                json.dumps(chunks_text, ensure_ascii=False),
                json.dumps(vectors_raw),
                dim,
                len(chunks_text),
                now,
            ),
        )
        self._conn.commit()

    # ── Diagnostics ──────────────────────────────────────────────────

    def count(self) -> int:
        """Return the number of entries in the cache."""
        cur = self._conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass
