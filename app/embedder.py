import httpx
import numpy as np
from typing import List
from .config import settings


class Embedder:
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.base_url = settings.OPENAI_BASE_URL.rstrip("/")
        self.model = settings.EMBEDDING_MODEL

    async def embed(self, text: str) -> np.ndarray:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if self.api_key == "dummy-key":
            return [self._mock_embed(text) for text in texts]

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        embeddings = []
        for item in sorted(result["data"], key=lambda x: x["index"]):
            embeddings.append(np.array(item["embedding"], dtype=np.float32))

        return embeddings

    def _mock_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vocab = {}
        idx = 0
        for word in words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1

        vector = np.zeros(128, dtype=np.float32)
        for i, word in enumerate(words):
            if word in vocab:
                vector[vocab[word] % 128] += 1.0
                vector[(i * 7) % 128] += 0.5

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
