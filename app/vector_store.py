import os
import json
import uuid
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .config import settings


class VectorStore:
    def __init__(self, store_path: str = None):
        self.store_path = store_path or settings.VECTOR_STORE_PATH
        self._ensure_storage_dir()
        self.documents: Dict[str, Dict] = {}
        self.chunks: List[Dict] = []
        self.vectors: Optional[np.ndarray] = None
        self._load()

    def _ensure_storage_dir(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

    def _load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data.get("documents", {})
            self.chunks = data.get("chunks", [])
            if self.chunks:
                self.vectors = np.array([c["vector"] for c in self.chunks], dtype=np.float32)
            else:
                self.vectors = None

    def _save(self):
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_document(
        self,
        filename: str,
        file_type: str,
        chunks_text: List[str],
        chunk_vectors: List[np.ndarray],
    ) -> str:
        doc_id = str(uuid.uuid4())

        self.documents[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "chunks_count": len(chunks_text),
        }

        for i, (text, vector) in enumerate(zip(chunks_text, chunk_vectors)):
            chunk_id = f"{doc_id}_{i}"
            self.chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "filename": filename,
                "content": text,
                "vector": vector.tolist(),
                "index": i,
            })

        self._rebuild_vector_index()
        self._save()
        return doc_id

    def _rebuild_vector_index(self):
        if self.chunks:
            self.vectors = np.array([c["vector"] for c in self.chunks], dtype=np.float32)
        else:
            self.vectors = None

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None,
    ) -> List[Tuple[Dict, float]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        top_k = top_k or settings.TOP_K
        query_vector = query_vector.reshape(1, -1)

        similarities = cosine_similarity(query_vector, self.vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            del chunk["vector"]
            score = float(similarities[idx])
            results.append((chunk, score))

        return results

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.documents.get(doc_id)

    def list_documents(self) -> List[Dict]:
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c["document_id"] != doc_id]
        self._rebuild_vector_index()
        self._save()
        return True

    def clear(self):
        self.documents = {}
        self.chunks = []
        self.vectors = None
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
