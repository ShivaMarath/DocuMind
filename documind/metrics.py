"""SQLite interaction log + pandas analytics for monitoring quality and usage."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

import pandas as pd

from .config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    search_query TEXT,
    answer TEXT,
    outcome TEXT,
    grounded INTEGER,
    n_retrieved INTEGER,
    n_relevant INTEGER,
    retrieval_attempts INTEGER,
    generation_attempts INTEGER,
    latency_ms REAL,
    prompt_version TEXT,
    provider TEXT,
    model TEXT,
    helpful INTEGER
)
"""


@contextmanager
def _conn(db_path: str | None = None):
    con = sqlite3.connect(db_path or get_settings().db_path)
    try:
        con.execute(_SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def log_interaction(record: dict, db_path: str | None = None) -> int:
    cols = [
        "question", "search_query", "answer", "outcome", "grounded", "n_retrieved", "n_relevant",
        "retrieval_attempts", "generation_attempts", "latency_ms", "prompt_version", "provider", "model",
    ]
    with _conn(db_path) as con:
        cur = con.execute(
            f"INSERT INTO interactions (ts, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
            [time.time(), *[record.get(c) for c in cols]],
        )
        return cur.lastrowid


def record_feedback(interaction_id: int, helpful: bool, db_path: str | None = None) -> None:
    with _conn(db_path) as con:
        con.execute("UPDATE interactions SET helpful=? WHERE id=?", (int(helpful), interaction_id))


def load_interactions(db_path: str | None = None) -> pd.DataFrame:
    with _conn(db_path) as con:
        return pd.read_sql_query("SELECT * FROM interactions ORDER BY ts", con)


def summarize(df: pd.DataFrame | None = None, db_path: str | None = None) -> dict:
    df = load_interactions(db_path) if df is None else df
    if df.empty:
        return {"total_queries": 0}
    rated = df["helpful"].dropna()
    return {
        "total_queries": int(len(df)),
        "answered_rate": round(float((df["outcome"] == "answered").mean()), 3),
        "no_context_rate": round(float((df["outcome"] == "no_context").mean()), 3),
        "ungrounded_rate": round(float((df["outcome"] == "ungrounded").mean()), 3),
        "self_correction_rate": round(
            float(((df["retrieval_attempts"] > 1) | (df["generation_attempts"] > 1)).mean()), 3
        ),
        "avg_relevant_docs": round(float(df["n_relevant"].mean()), 2),
        "latency_p50_ms": round(float(df["latency_ms"].quantile(0.5)), 1),
        "latency_p95_ms": round(float(df["latency_ms"].quantile(0.95)), 1),
        "feedback_count": int(len(rated)),
        "helpful_rate": round(float(rated.mean()), 3) if len(rated) else None,
    }
