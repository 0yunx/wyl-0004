"""Text embedding client with OpenAI-compatible API support.

When ``OPENAI_API_KEY`` is set to its default (``dummy-key``), the
:class:`Embedder` falls back to a deterministic **mock embedding** that
produces 128-dimensional bag-of-words vectors.  This allows full
end-to-end testing without an LLM API, but the vectors are *not*
semantically meaningful.

With a real API key, :meth:`embed_batch` calls the
``/embeddings`` endpoint and returns the model's native dimensionality
(e.g. 1536 for ``text-embedding-3-small``).
"""

import httpx
import numpy as np
from typing import List
from .config import settings


class Embedder:
    """Produce vector embeddings for text via an OpenAI-compatible API.

    Attributes:
        api_key: OpenAI API key (``dummy-key`` enables mock mode).
        base_url: Base URL of the embeddings endpoint.
        model: Model identifier sent to the API.
    """

    def __init__(self) -> None:
        self.api_key: str = settings.OPENAI_API_KEY
        self.base_url: str = settings.OPENAI_BASE_URL.rstrip("/")
        self.model: str = settings.EMBEDDING_MODEL

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Convenience wrapper around :meth:`embed_batch`.

        Args:
            text: Input text.

        Returns:
            1-D float32 array of shape ``(dim,)``.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts in a single API call.

        In mock mode (``api_key == 'dummy-key'``) each text is encoded
        with a deterministic bag-of-words hash into 128 dimensions.

        Args:
            texts: List of input strings.

        Returns:
            List of 1-D float32 arrays, one per input text.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx response.
        """
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

    @staticmethod
    def _mock_embed(text: str) -> np.ndarray:
        """Produce a deterministic 128-d mock embedding for *text*.

        Uses a bag-of-words hash with positional mixing so that
        different orderings of the same words produce different vectors.

        Args:
            text: Input string.

        Returns:
            L2-normalized float32 vector of shape ``(128,)``.
        """
        words = text.lower().split()
        vocab: dict = {}
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
