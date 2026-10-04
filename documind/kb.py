"""Knowledge-base management: upload, list, inspect and remove documents."""
from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.documents import Document
from qdrant_client.http import models as qm

from .config import get_settings
from .ingestion import SUPPORTED_EXTENSIONS, ingest_paths
from .vectorstore import _PAYLOAD_SOURCE_KEY, delete_source, get_client


def upload_dir() -> Path:
    return Path(os.getenv("UPLOAD_DIR", "uploads"))


def safe_name(name: str) -> str:
    """Strip path components and odd characters from an uploaded filename."""
    n = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip()
    return n or "unnamed"


def add_files(files: list[tuple[str, bytes]], vectorstore=None) -> dict:
    """Save uploaded (filename, bytes) pairs and index them. Same filename = replace the old version."""
    upload_dir().mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []
    for name, data in files:
        n = safe_name(name)
        if Path(n).suffix.lower() not in SUPPORTED_EXTENSIONS:
            rejected.append(n)
            continue
        dest = upload_dir() / n
        dest.write_bytes(data)
        saved.append(dest)
    summary = ingest_paths(saved, vectorstore=vectorstore) if saved else {"files": 0, "chunks": 0, "details": []}
    summary["rejected"] = rejected
    return summary


def _scroll(with_payload=True, sources: list[str] | None = None):
    s = get_settings()
    client = get_client()
    if not client.collection_exists(s.collection_name):
        return
    flt = None
    if sources:
        flt = qm.Filter(must=[qm.FieldCondition(key=_PAYLOAD_SOURCE_KEY, match=qm.MatchAny(any=list(sources)))])
    offset = None
    while True:
        points, offset = client.scroll(
            s.collection_name, scroll_filter=flt, limit=256, offset=offset,
            with_payload=with_payload, with_vectors=False,
        )
        yield from points
        if offset is None:
            break


def list_sources() -> list[dict]:
    """One row per indexed document: name, number of chunks, pages and characters."""
    agg: dict[str, dict] = {}
    for p in _scroll():
        payload = p.payload or {}
        m = payload.get("metadata", {})
        name = m.get("source_name", "unknown")
        row = agg.setdefault(name, {"source": name, "chunks": 0, "pages": set(), "characters": 0})
        row["chunks"] += 1
        row["characters"] += len(payload.get("page_content", ""))
        if m.get("page"):
            row["pages"].add(m["page"])
    return [{**r, "pages": len(r["pages"]) or None} for r in sorted(agg.values(), key=lambda r: r["source"])]


def get_chunks(sources: list[str] | None = None) -> list[Document]:
    """All indexed chunks (optionally limited to some documents). Used to build ground-truth eval sets."""
    docs = []
    for p in _scroll(sources=sources):
        payload = p.payload or {}
        meta = dict(payload.get("metadata", {}))
        meta["chunk_id"] = str(p.id)
        docs.append(Document(page_content=payload.get("page_content", ""), metadata=meta))
    return docs


def remove_source(name: str) -> None:
    delete_source(name)
    f = upload_dir() / safe_name(name)
    if f.exists():
        f.unlink()
