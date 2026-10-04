"""FastAPI service:  python -m documind.cli serve   (docs at /docs)

Note: embedded Qdrant (no QDRANT_URL) allows only ONE process at a time. If you run the API and the
Streamlit UI together, use a Qdrant server (docker compose up -d) and set QDRANT_URL.
"""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import evalsets, kb
from .evalgen import generate_eval_set
from .evaluate import run_eval
from .graph import RAGPipeline
from .metrics import record_feedback, summarize

app = FastAPI(title="DocuMind AI", version="1.1.0")
_rag: RAGPipeline | None = None


def rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


class AskRequest(BaseModel):
    question: str
    history: list[dict] = []


class FeedbackRequest(BaseModel):
    interaction_id: int
    helpful: bool


class GenerateEvalRequest(BaseModel):
    name: str
    sources: list[str] | None = None
    n_answerable: int = 20
    n_unanswerable: int = 5
    seed: int = 42


class RunEvalRequest(BaseModel):
    set_name: str
    use_judge: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- knowledge base ----------
@app.post("/sources")
async def upload_sources(files: list[UploadFile] = File(...)):
    """Upload PDF / DOCX / MD / TXT files and index them. Re-uploading a filename replaces it."""
    payload = [(f.filename or "", await f.read()) for f in files]
    return kb.add_files(payload, vectorstore=rag().vs)


@app.get("/sources")
def list_sources():
    return kb.list_sources()


@app.delete("/sources/{name}")
def delete_source(name: str):
    kb.remove_source(name)
    return {"removed": name}


# ---------- question answering ----------
@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    return rag().ask(req.question, history=req.history)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    record_feedback(req.interaction_id, req.helpful)
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return summarize()


# ---------- ground-truth evaluation ----------
@app.get("/eval/sets")
def eval_sets():
    return evalsets.list_sets()


@app.get("/eval/sets/{name}")
def eval_set(name: str):
    try:
        return evalsets.load_set(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/eval/generate")
def eval_generate(req: GenerateEvalRequest):
    """Generate a ground-truth set from the indexed documents (can take a minute)."""
    try:
        items = generate_eval_set(rag(), kb.get_chunks(req.sources), req.n_answerable, req.n_unanswerable, req.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    evalsets.save_set(req.name, items)
    return {"name": evalsets.slug(req.name), "questions": len(items)}


@app.post("/eval/import")
async def eval_import(name: str, file: UploadFile = File(...)):
    try:
        items = evalsets.parse_upload(file.filename or "", await file.read())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, str(e))
    evalsets.save_set(name, items)
    return {"name": evalsets.slug(name), "questions": len(items)}


@app.post("/eval/run")
def eval_run(req: RunEvalRequest):
    try:
        items = evalsets.load_set(req.set_name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    out = run_eval(items, name=req.set_name, pipeline=rag(), use_judge=req.use_judge)
    return {"summary": out["summary"], "csv_path": out["csv_path"]}
