"""Application configuration loaded from environment variables.

All tunables are read from ``.env`` via :mod:`python-dotenv`.  Every setting
has a sensible default so the app works out-of-the-box in "mock" mode
(no real LLM API key required for basic testing).

Environment variables
---------------------
OPENAI_API_KEY         API key for the OpenAI-compatible embedding/chat endpoint.
                       Set to ``dummy-key`` (the default) to enable mock mode.
OPENAI_BASE_URL        Base URL of the OpenAI-compatible API.
EMBEDDING_MODEL        Model name for text embeddings.
CHAT_MODEL             Model name for chat completions.
CHUNK_SIZE             Max characters per text chunk (default 500).
CHUNK_OVERLAP          Overlap characters between consecutive chunks (default 100).
TOP_K                  Default number of context chunks retrieved per query (default 3).
HNSW_M                 HNSW graph max-edges-per-node (default 16).
HNSW_EF_CONSTRUCTION   HNSW beam width during index build (default 200).
HNSW_EF                HNSW beam width during queries (default 50).
VECTOR_STORE_PATH      Path used to derive the data directory for vector storage.
ALLOWED_EXTENSIONS     Comma-separated list of accepted file extensions.
MAX_FILE_SIZE          Maximum upload size in bytes (default 10 MB).
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Read-once configuration container.

    Values are resolved at import time; changing env vars requires a
    process restart (or calling ``load_dotenv(override=True)``).
    """

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "dummy-key")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    HNSW_M: int = int(os.getenv("HNSW_M", "16"))
    HNSW_EF_CONSTRUCTION: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    HNSW_EF: int = int(os.getenv("HNSW_EF", "50"))
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store.json")
    CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", "./data/dedupe_cache.db")
    ALLOWED_EXTENSIONS: set = {".txt", ".pdf"}
    MAX_FILE_SIZE: int = 10 * 1024 * 1024


settings = Settings()
