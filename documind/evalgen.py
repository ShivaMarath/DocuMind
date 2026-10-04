"""
Build a ground-truth evaluation set automatically from the user's own indexed documents.

Answerable questions : sample chunks (round-robin across documents) -> LLM writes a question, a reference
                       answer and verbatim keywords from that chunk. The source chunk is recorded as the
                       ground-truth evidence, so retrieval can be scored exactly (source + chunk level).
Unanswerable questions: LLM writes plausible-but-unanswered questions; each one is then checked against the
                       real index and dropped if any retrieved passage is judged relevant. These measure
                       whether the system abstains instead of hallucinating.
"""
from __future__ import annotations

import math
import random
import re
import uuid
from typing import Callable

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .config import get_settings
from .prompts import QA_GEN_PROMPT, UNANSWERABLE_PROMPT

ProgressCb = Callable[[str, int, int], None]

_LEAKY = re.compile(r"\b(the passage|this passage|the text|this text|the document|this document|the excerpt|above|according to the (passage|text))\b", re.I)


class GeneratedQA(BaseModel):
    skip: bool = Field(default=False, description="True if the passage has nothing worth asking about")
    question: str = ""
    reference: str = ""
    must_include: list[str] = Field(default_factory=list)


class UnanswerableQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list)


def usable_chunk(doc: Document, min_chars: int = 250) -> bool:
    text = doc.page_content.strip()
    if len(text) < min_chars:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / len(text) > 0.55  # skips tables of numbers, page furniture, etc.


def sample_chunks(chunks: list[Document], n: int, seed: int = 42) -> list[Document]:
    """Round-robin across documents so every uploaded file is represented."""
    rng = random.Random(seed)
    groups: dict[str, list[Document]] = {}
    for c in chunks:
        if usable_chunk(c):
            groups.setdefault(c.metadata.get("source_name", "?"), []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    picked: list[Document] = []
    while len(picked) < n and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(picked) < n:
                picked.append(groups[name].pop())
    return picked


def _norm(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def generate_answerable(rag, chunks: list[Document], n: int, seed: int = 42,
                        progress_cb: ProgressCb | None = None) -> list[dict]:
    """Draw chunks (round-robin across documents) in batches until ``n`` good questions exist or chunks run out.
    Batches are needed because the LLM skips boilerplate and some questions are filtered or duplicated."""
    if n <= 0:
        return []
    pool = sample_chunks(chunks, len(chunks), seed)      # every usable chunk, in round-robin order
    chain = QA_GEN_PROMPT | rag.llm.with_structured_output(GeneratedQA)
    items, seen, cursor = [], set(), 0
    while len(items) < n and cursor < len(pool):
        batch = pool[cursor: cursor + math.ceil((n - len(items)) * 1.5)]
        cursor += len(batch)
        results = chain.batch(
            [{"source": c.metadata.get("source_name"), "passage": c.page_content} for c in batch],
            config={"max_concurrency": 5}, return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) == len(results):
            raise errors[0]                              # e.g. bad API key / quota: fail loudly, not silently
        for chunk, res in zip(batch, results):
            if len(items) >= n:
                break
            if isinstance(res, Exception) or res.skip or not res.question.strip() or not res.reference.strip():
                continue
            if _LEAKY.search(res.question) or _norm(res.question) in seen:
                continue
            seen.add(_norm(res.question))
            passage = chunk.page_content.lower()
            keywords = [k for k in res.must_include if k.strip() and k.lower() in passage][:3]  # drop invented keywords
            items.append({
                "id": uuid.uuid4().hex[:8], "question": res.question.strip(), "reference": res.reference.strip(),
                "answerable": True, "expected_source": chunk.metadata.get("source_name"),
                "expected_page": chunk.metadata.get("page"), "expected_chunk_id": chunk.metadata.get("chunk_id"),
                "evidence": chunk.page_content, "must_include": keywords, "origin": "synthetic",
            })
            if progress_cb:
                progress_cb("answerable", len(items), n)
    return items


def generate_unanswerable(rag, chunks: list[Document], n: int, seed: int = 42,
                          progress_cb: ProgressCb | None = None) -> list[dict]:
    if n <= 0:
        return []
    excerpts_docs = sample_chunks(chunks, 8, seed + 1)
    if not excerpts_docs:
        return []
    excerpts = "\n\n".join(f"[{c.metadata.get('source_name')}] {c.page_content[:500]}" for c in excerpts_docs)
    ask = math.ceil(n * 2)  # over-ask: some will turn out to be answerable and get filtered
    out = (UNANSWERABLE_PROMPT | rag.llm.with_structured_output(UnanswerableQuestions)).invoke(
        {"n": ask, "excerpts": excerpts})
    items, seen, last_err = [], set(), None
    for q in out.questions:
        if len(items) >= n:
            break
        q = q.strip()
        if not q or _LEAKY.search(q) or _norm(q) in seen:
            continue
        seen.add(_norm(q))
        try:  # verify against the real index: if anything relevant is retrievable, it's not truly unanswerable
            docs = rag.vs.similarity_search(q, k=get_settings().top_k)
            grades = rag.chains["grade"].batch([{"question": q, "passage": d.page_content} for d in docs])
            if any(g.relevant for g in grades):
                continue
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        items.append({
            "id": uuid.uuid4().hex[:8], "question": q, "reference": "", "answerable": False,
            "expected_source": None, "expected_page": None, "expected_chunk_id": None,
            "evidence": "", "must_include": [], "origin": "synthetic_unanswerable",
        })
        if progress_cb:
            progress_cb("unanswerable", len(items), n)
    if not items and last_err:
        raise last_err  # every verification call failed (API/network problem) - don't report "no questions"
    return items


def generate_eval_set(rag, chunks: list[Document], n_answerable: int = 20, n_unanswerable: int = 5,
                      seed: int = 42, progress_cb: ProgressCb | None = None) -> list[dict]:
    if not chunks:
        raise ValueError("The knowledge base is empty. Upload documents first.")
    items = generate_answerable(rag, chunks, n_answerable, seed, progress_cb)
    items += generate_unanswerable(rag, chunks, n_unanswerable, seed, progress_cb)
    if not items:
        raise ValueError("Could not generate any questions - the documents may be too short or image-only.")
    return items
