from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

_DEFAULTS = {
    "openai": {"chat": "gpt-4o-mini", "embedding": "text-embedding-3-small"},
    "gemini": {"chat": "gemini-2.5-flash", "embedding": "models/gemini-embedding-001"},
}


@dataclass
class Settings:
    provider: str
    chat_model: str
    embedding_model: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_path: str
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_retrieval_attempts: int
    max_generation_attempts: int
    db_path: str

    @property
    def collection_name(self) -> str:
        # Embedding dimensions differ per provider, so keep a collection per provider.
        return f"{self.collection}_{self.provider}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in _DEFAULTS:
        raise ValueError(f"LLM_PROVIDER must be one of {list(_DEFAULTS)}, got '{provider}'")
    d = _DEFAULTS[provider]
    return Settings(
        provider=provider,
        chat_model=os.getenv("CHAT_MODEL", d["chat"]),
        embedding_model=os.getenv("EMBEDDING_MODEL", d["embedding"]),
        qdrant_url=os.getenv("QDRANT_URL") or None,
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_path=os.getenv("QDRANT_PATH", "./.qdrant"),
        collection=os.getenv("QDRANT_COLLECTION", "documind"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "5")),
        max_retrieval_attempts=int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2")),
        max_generation_attempts=int(os.getenv("MAX_GENERATION_ATTEMPTS", "2")),
        db_path=os.getenv("METRICS_DB", "./documind_metrics.db"),
    )
