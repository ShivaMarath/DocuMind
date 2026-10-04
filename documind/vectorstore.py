"""Qdrant vector store setup (embedded local mode or a Qdrant server / Qdrant Cloud)."""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import get_settings

_PAYLOAD_SOURCE_KEY = "metadata.source_name"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return QdrantClient(path=s.qdrant_path)


def get_vectorstore(embeddings=None):
    from langchain_qdrant import QdrantVectorStore

    from .llm import get_embeddings

    s = get_settings()
    embeddings = embeddings or get_embeddings()
    client = get_client()

    if not client.collection_exists(s.collection_name):
        dim = len(embeddings.embed_query("dimension probe"))
        client.create_collection(
            collection_name=s.collection_name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client, collection_name=s.collection_name, embedding=embeddings
    )


def delete_source(source_name: str, keep_ids: list[str] | None = None) -> None:
    """Remove a file's chunks from the index.

    With ``keep_ids`` only chunks NOT in that list are removed, so re-indexing an edited file first
    writes the new chunks and then drops the stale ones (a failed embedding call never wipes a document).
    """
    s = get_settings()
    client = get_client()
    if not client.collection_exists(s.collection_name):
        return
    client.delete(
        collection_name=s.collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[qm.FieldCondition(key=_PAYLOAD_SOURCE_KEY, match=qm.MatchValue(value=source_name))],
                must_not=[qm.HasIdCondition(has_id=keep_ids)] if keep_ids else None,
            )
        ),
    )
