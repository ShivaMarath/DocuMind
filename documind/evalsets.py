"""Storage and import/export of evaluation (ground-truth) sets as JSONL files."""
from __future__ import annotations

import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import pandas as pd

FIELDS = ["id", "question", "reference", "answerable", "expected_source", "expected_page",
          "expected_chunk_id", "evidence", "must_include", "origin"]


def sets_dir() -> Path:
    return Path(os.getenv("EVAL_SETS_DIR", "eval_sets"))


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "eval-set"


def _path(name: str) -> Path:
    return sets_dir() / f"{slug(name)}.jsonl"


def _clean(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip()) else v


def _to_bool(v, default=True) -> bool:
    v = _clean(v)
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() not in {"false", "no", "0", "n", "f"}
    return bool(v)


def _to_list(v) -> list[str]:
    v = _clean(v)
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in re.split(r"[;|]", str(v)) if x.strip()]


def normalize_item(raw: dict, default_origin: str = "manual") -> dict:
    q = _clean(raw.get("question"))
    if not q:
        raise ValueError("every row needs a non-empty 'question'")
    page = _clean(raw.get("expected_page"))
    answerable = _to_bool(raw.get("answerable"))
    return {
        "id": str(_clean(raw.get("id")) or uuid.uuid4().hex[:8]),
        "question": str(q).strip(),
        "reference": str(_clean(raw.get("reference")) or "") if answerable else "",
        "answerable": answerable,
        "expected_source": _clean(raw.get("expected_source")) if answerable else None,
        "expected_page": int(page) if page is not None else None,
        "expected_chunk_id": _clean(raw.get("expected_chunk_id")),
        "evidence": str(_clean(raw.get("evidence")) or ""),
        "must_include": _to_list(raw.get("must_include")),
        "origin": str(_clean(raw.get("origin")) or default_origin),
    }


def parse_upload(filename: str, data: bytes) -> list[dict]:
    """Parse a user-supplied Q&A file (.csv or .jsonl). Columns: question (required), reference,
    expected_source, answerable (true/false), must_include (a;b;c)."""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        rows = pd.read_csv(io.BytesIO(data)).to_dict("records")
    elif ext in {".jsonl", ".json"}:
        text = data.decode("utf-8")
        rows = json.loads(text) if ext == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        raise ValueError("Upload a .csv or .jsonl file")
    items = [normalize_item(r, default_origin="uploaded") for r in rows]
    if not items:
        raise ValueError("the file contains no rows")
    return items


def save_set(name: str, items: list[dict]) -> Path:
    sets_dir().mkdir(parents=True, exist_ok=True)
    path = _path(name)
    normalized = [normalize_item(i, default_origin=i.get("origin") or "manual") for i in items]
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in normalized) + "\n", encoding="utf-8")
    return path


def load_set(name: str) -> list[dict]:
    path = _path(name)
    if not path.exists():
        raise FileNotFoundError(f"No evaluation set named '{name}'. Existing: {[s['name'] for s in list_sets()]}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def list_sets() -> list[dict]:
    if not sets_dir().exists():
        return []
    out = []
    for p in sorted(sets_dir().glob("*.jsonl")):
        n = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        out.append({"name": p.stem, "questions": n, "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))})
    return out


def delete_set(name: str) -> None:
    _path(name).unlink(missing_ok=True)


# ---- table <-> items (for the editable grid in the UI) ----
def to_frame(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{**i, "must_include": "; ".join(i.get("must_include", []))} for i in items])
    return df.reindex(columns=FIELDS)


def from_frame(df: pd.DataFrame) -> list[dict]:
    items = []
    for rec in df.to_dict("records"):
        if not _clean(rec.get("question")):
            continue  # ignore blank rows added in the editor
        items.append(normalize_item(rec))
    return items
