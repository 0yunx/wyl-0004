import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K = int(os.getenv("TOP_K", "3"))
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store.json")
    ALLOWED_EXTENSIONS = {".txt", ".pdf"}
    MAX_FILE_SIZE = 10 * 1024 * 1024


settings = Settings()
