"""Binary-backed vector store with HNSW approximate-nearest-neighbor index.

Storage layout on disk::

    data/
    ├── metadata.json      Document & chunk metadata (no vectors)
    ├── vectors.bin        Raw float32 array, shape (N, dim), append-only
    └── hnsw_index.json    Serialized HNSW graph

Design decisions
----------------
* **Binary vectors** — 4 bytes/float vs ~20 bytes/float in JSON.
  A 1536-d OpenAI embedding stored as JSON ≈ 30 KB; as raw binary ≈ 6 KB.
* **Append-only writes** — new vectors are appended to ``vectors.bin``;
  the file is only rewritten during compaction (triggered when garbage
  exceeds 20 % of total rows).
* **Memory-mapped reads** — ``np.memmap`` lets the OS page vectors in
  and out on demand; a 1 GB store doesn't need 1 GB of resident RAM.
* **HNSW index** — O(log N) search instead of O(N) brute-force.
  Vectors are L2-normalized before insertion so that Euclidean distance
  is monotonically related to cosine distance.
* **Atomic metadata writes** — metadata is written to a ``.tmp`` file
  then ``os.replace``-d, preventing corruption on crash mid-write.
"""

import os
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import settings
from .hnsw import HNSWIndex


class VectorStore:
    """Persistent vector store backed by binary vector files and an HNSW index.

    Attributes:
        data_dir: Directory containing all storage files.
        dim: Dimensionality of stored (normalized) vectors.
    """

    def __init__(self, data_dir: str = None) -> None:
        self.data_dir = data_dir or os.path.dirname(
            settings.VECTOR_STORE_PATH
        ) or "./data"
        os.makedirs(self.data_dir, exist_ok=True)

        self._meta_path: str = os.path.join(self.data_dir, "metadata.json")
        self._vectors_path: str = os.path.join(self.data_dir, "vectors.bin")
        self._index_path: str = os.path.join(self.data_dir, "hnsw_index.json")

        self.documents: Dict[str, Dict] = {}
        self.chunks: List[Dict] = []
        self._row_to_idx: Dict[int, int] = {}

        self._vectors_mmap: Optional[np.memmap] = None
        self._dim: int = 0
        self._n_vectors: int = 0
        self._index: Optional[HNSWIndex] = None
        self._deleted_rows: set = set()

        self._load()

    @property
    def dim(self) -> int:
        """Dimensionality of stored vectors."""
        return self._dim

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        """Load metadata, memory-map vectors, and restore HNSW index."""
        if not os.path.exists(self._meta_path):
            return

        with open(self._meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = data.get("documents", {})
        self.chunks = data.get("chunks", [])
        self._dim = data.get("dim", 0)
        self._n_vectors = data.get("n_vectors", 0)
        self._deleted_rows = set(data.get("deleted_rows", []))
        self._rebuild_row_index()

        if self._n_vectors > 0 and self._dim > 0 and os.path.exists(self._vectors_path):
            self._vectors_mmap = np.memmap(
                self._vectors_path,
                dtype=np.float32,
                mode="r",
                shape=(self._n_vectors, self._dim),
            )

        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                self._index = HNSWIndex.from_dict(json.load(f))

    def _save_metadata(self) -> None:
        """Write metadata JSON atomically (write-to-tmp + rename)."""
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "dim": self._dim,
            "n_vectors": self._n_vectors,
            "deleted_rows": list(self._deleted_rows),
        }
        tmp = self._meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._meta_path)

    def _save_index(self) -> None:
        """Write HNSW index JSON atomically."""
        if self._index is None:
            return
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index.to_dict(), f)
        os.replace(tmp, self._index_path)

    def _rebuild_row_index(self) -> None:
        """Rebuild the row_idx → chunks-list-position lookup dict."""
        self._row_to_idx = {}
        for i, chunk in enumerate(self.chunks):
            row = chunk.get("row_idx")
            if row is not None:
                self._row_to_idx[row] = i

    # ── Vector I/O ───────────────────────────────────────────────────

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize each row vector in-place, guarding against zero rows."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms
        return vectors

    def _append_vectors(self, vectors: np.ndarray) -> int:
        """Append L2-normalized vectors to the binary file and refresh the memmap.

        The file is opened in append-binary mode so existing data is never
        rewritten.  After the append the memmap is re-created to cover the
        new length.

        Args:
            vectors: Matrix of shape (n, dim) to append.

        Returns:
            Row index of the first newly appended vector.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        start_idx = self._n_vectors
        rows, dim = vectors.shape

        if self._dim == 0:
            self._dim = dim
        assert dim == self._dim, (
            f"Vector dimension mismatch: store has dim={self._dim}, got {dim}"
        )

        self._normalize(vectors)

        if self._vectors_mmap is not None:
            del self._vectors_mmap
            self._vectors_mmap = None

        with open(self._vectors_path, "ab") as f:
            f.write(vectors.tobytes())

        self._n_vectors += rows

        self._vectors_mmap = np.memmap(
            self._vectors_path,
            dtype=np.float32,
            mode="r",
            shape=(self._n_vectors, self._dim),
        )
        return start_idx

    def _compact(self) -> None:
        """Rewrite the binary vector file, excluding soft-deleted rows.

        Also remaps chunk ``row_idx`` values and rebuilds the HNSW index
        because row indices change after compaction.

        Triggered automatically by :meth:`delete_document` when garbage
        exceeds 20 % of total rows.
        """
        if not self._deleted_rows:
            return

        kept = [i for i in range(self._n_vectors) if i not in self._deleted_rows]
        if not kept:
            if self._vectors_mmap is not None:
                del self._vectors_mmap
                self._vectors_mmap = None
            if os.path.exists(self._vectors_path):
                os.remove(self._vectors_path)
            self._n_vectors = 0
            self._dim = 0
            self._deleted_rows.clear()
            return

        old_to_new = {old: new for new, old in enumerate(kept)}
        new_vectors = np.ascontiguousarray(
            self._vectors_mmap[kept], dtype=np.float32,
        )

        del self._vectors_mmap
        self._vectors_mmap = None

        with open(self._vectors_path, "wb") as f:
            f.write(new_vectors.tobytes())

        self._n_vectors = len(kept)
        self._vectors_mmap = np.memmap(
            self._vectors_path,
            dtype=np.float32,
            mode="r",
            shape=(self._n_vectors, self._dim),
        )

        for chunk in self.chunks:
            old = chunk.get("row_idx")
            if old is not None and old in old_to_new:
                chunk["row_idx"] = old_to_new[old]

        self._deleted_rows.clear()
        self._rebuild_row_index()
        self._rebuild_hnsw_index()

    # ── HNSW index management ────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Lazy-init the HNSW index when the first vector arrives."""
        if self._index is None and self._dim > 0:
            self._index = HNSWIndex(
                dim=self._dim,
                M=settings.HNSW_M,
                ef_construction=settings.HNSW_EF_CONSTRUCTION,
                ef=settings.HNSW_EF,
            )

    def _rebuild_hnsw_index(self) -> None:
        """Full rebuild of the HNSW graph from current vectors.

        O(N log N).  Called after compaction where row indices shift.
        """
        self._ensure_index()
        if self._index is None or self._n_vectors == 0:
            return

        self._index = HNSWIndex(
            dim=self._dim,
            M=settings.HNSW_M,
            ef_construction=settings.HNSW_EF_CONSTRUCTION,
            ef=settings.HNSW_EF,
        )
        for i in range(self._n_vectors):
            self._index.insert(i, self._vectors_mmap[i], self._vectors_mmap)

    # ── Public API ───────────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        file_type: str,
        chunks_text: List[str],
        chunk_vectors: List[np.ndarray],
    ) -> str:
        """Ingest a new document: normalize, store, and index its chunks.

        Args:
            filename: Original upload file name.
            file_type: ``'txt'`` or ``'pdf'``.
            chunks_text: Text of each chunk.
            chunk_vectors: Embedding vector for each chunk (raw, pre-normalization).

        Returns:
            The UUID assigned to the new document.
        """
        doc_id = str(uuid.uuid4())

        self.documents[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "chunks_count": len(chunks_text),
        }

        start_idx = self._append_vectors(np.stack(chunk_vectors))

        self._ensure_index()
        for i, text in enumerate(chunks_text):
            row_idx = start_idx + i
            chunk_id = f"{doc_id}_{i}"

            self.chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "filename": filename,
                "content": text,
                "row_idx": row_idx,
                "index": i,
            })
            self._row_to_idx[row_idx] = len(self.chunks) - 1

            if self._index is not None:
                self._index.insert(
                    row_idx, self._vectors_mmap[row_idx], self._vectors_mmap,
                )

        self._save_metadata()
        self._save_index()
        return doc_id

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None,
    ) -> List[Tuple[Dict, float]]:
        """Find the *top_k* most similar chunks to a query vector.

        Uses the HNSW index for O(log N) approximate search.  Falls back
        to brute-force if the index is unavailable (empty store or
        corrupted index).

        Args:
            query_vector: Query embedding, shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of (chunk_metadata_dict, cosine_similarity) pairs sorted
            by similarity descending.
        """
        if self._n_vectors == 0:
            return []

        top_k = top_k or settings.TOP_K

        if self._index is not None and len(self._index.node_levels) > 0:
            return self._search_hnsw(query_vector, top_k)
        return self._search_brute(query_vector, top_k)

    def _search_hnsw(
        self, query_vector: np.ndarray, top_k: int,
    ) -> List[Tuple[Dict, float]]:
        """HNSW-based ANN search with cosine-similarity re-scoring.

        HNSW operates on squared Euclidean distance internally.  Because
        all stored vectors are L2-normalized, we convert:

            cos(q, v) = 1 − ||q − v||² / 2   (when ||q|| = ||v|| = 1)

        We over-fetch (top_k × 3) from HNSW, filter soft-deleted rows,
        re-score, and truncate.
        """
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        query_normalized = (query_vector / q_norm).astype(np.float32)

        n_candidates = max(top_k * 3, top_k + len(self._deleted_rows) + 10)
        raw = self._index.search(query_normalized, n_candidates, self._vectors_mmap)

        results: List[Tuple[Dict, float]] = []
        for sq_dist, row_idx in raw:
            if row_idx in self._deleted_rows:
                continue
            cos_sim = 1.0 - sq_dist / 2.0
            pos = self._row_to_idx.get(row_idx)
            if pos is not None:
                results.append((self.chunks[pos].copy(), cos_sim))
            if len(results) >= top_k:
                break

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _search_brute(
        self, query_vector: np.ndarray, top_k: int,
    ) -> List[Tuple[Dict, float]]:
        """Fallback brute-force cosine similarity search (no index)."""
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            return []
        q = (query_vector / q_norm).astype(np.float32).reshape(1, -1)

        sims = (self._vectors_mmap @ q.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k + len(self._deleted_rows)]

        results: List[Tuple[Dict, float]] = []
        for idx in top_indices:
            idx = int(idx)
            if idx in self._deleted_rows:
                continue
            pos = self._row_to_idx.get(idx)
            if pos is not None:
                results.append((self.chunks[pos].copy(), float(sims[idx])))
            if len(results) >= top_k:
                break

        return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Return metadata for a document, or ``None`` if not found."""
        return self.documents.get(doc_id)

    def list_documents(self) -> List[Dict]:
        """Return a list of all stored document metadata dicts."""
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks.

        Vectors are soft-deleted (added to ``_deleted_rows``) and the
        HNSW entries are patched out.  If garbage exceeds 20 % of total
        rows a full compaction is triggered to reclaim disk space.

        Returns:
            ``True`` if the document was found and deleted.
        """
        if doc_id not in self.documents:
            return False

        rows_to_delete = {
            c["row_idx"] for c in self.chunks if c["document_id"] == doc_id
        }
        self._deleted_rows.update(rows_to_delete)

        if self._index is not None:
            for row_idx in rows_to_delete:
                self._index.remove(row_idx)

        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c["document_id"] != doc_id]
        self._rebuild_row_index()

        if self._n_vectors > 0 and len(self._deleted_rows) / self._n_vectors > 0.2:
            self._compact()
        else:
            self._save_metadata()
            self._save_index()

        return True

    def clear(self) -> None:
        """Wipe all stored data (metadata, vectors, index files)."""
        self.documents = {}
        self.chunks = []
        self._row_to_idx = {}
        self._deleted_rows.clear()
        self._n_vectors = 0
        self._dim = 0
        if self._vectors_mmap is not None:
            del self._vectors_mmap
            self._vectors_mmap = None
        self._index = None
        for path in (self._meta_path, self._vectors_path, self._index_path):
            if os.path.exists(path):
                os.remove(path)
