"""
Offline evaluation (the "Quality Assurance" layer). Runs a ground-truth set through the RAG pipeline.

Ground-truth items come from documind.evalsets (generated from your PDFs, uploaded, or hand-edited):
  question, reference, answerable, expected_source, expected_chunk_id, must_include, ...

Metrics
  Retrieval : source hit-rate/MRR, chunk hit-rate/MRR (raw top-k, before LLM grading)
  Generation: LLM-judge score (1-5), keyword recall, grounded rate
  Safety    : abstention accuracy on unanswerable questions (hallucination guard)
  Ops       : latency p50/p95, self-correction rate, error count
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from pydantic import BaseModel, Field

from .config import get_settings
from .graph import RAGPipeline
from .ingestion import chunk_id
from .prompts import JUDGE_PROMPT


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = ""


def keyword_recall(answer: str, must_include: list[str]) -> float | None:
    if not must_include:
        return None
    a = answer.lower()
    return sum(k.lower() in a for k in must_include) / len(must_include)


def rank_of(retrieved: list, expected) -> int | None:
    """1-based rank of the first retrieved item equal to ``expected``; None if absent or no expectation."""
    if not expected:
        return None
    for i, r in enumerate(retrieved, 1):
        if r == expected:
            return i
    return None


def _hit_mrr(df: pd.DataFrame, expected_col: str, rank_col: str):
    sub = df[df[expected_col].notna() & df["answerable"]]
    if sub.empty:
        return None, None
    hit = round(float(sub[rank_col].notna().mean()), 3)
    mrr = round(float(sub[rank_col].map(lambda r: 1 / r if pd.notna(r) else 0).mean()), 3)
    return hit, mrr


def _mean(series: pd.Series, nd: int):
    return round(float(series.mean()), nd) if series.notna().any() else None


def summarize_results(df: pd.DataFrame) -> dict:
    errors = int((df["outcome"] == "error").sum())
    ok = df[df["outcome"] != "error"]
    summary: dict = {"n_questions": int(len(df)), "n_errors": errors}
    if ok.empty:
        return summary
    ans, unans = ok[ok["answerable"]], ok[~ok["answerable"]]
    src_hit, src_mrr = _hit_mrr(ok, "expected_source", "source_rank")
    chk_hit, chk_mrr = _hit_mrr(ok, "expected_chunk_id", "chunk_rank")
    summary.update({
        "n_answerable": int(len(ans)), "n_unanswerable": int(len(unans)),
        "source_hit_rate": src_hit, "source_mrr": src_mrr,
        "chunk_hit_rate": chk_hit, "chunk_mrr": chk_mrr,
        "avg_judge_score": _mean(ans["judge_score"], 2),
        "avg_keyword_recall": _mean(ans["keyword_recall"], 3),
        "answer_rate_on_answerable": round(float((ans["outcome"] == "answered").mean()), 3) if len(ans) else None,
        "abstention_accuracy_on_unanswerable": round(float((unans["outcome"] != "answered").mean()), 3) if len(unans) else None,
        "grounded_rate": round(float(ok["grounded"].mean()), 3),
        "self_correction_rate": round(float(((ok["retrieval_attempts"] > 1) | (ok["generation_attempts"] > 1)).mean()), 3),
        "latency_p50_ms": round(float(ok["latency_ms"].quantile(0.5)), 1),
        "latency_p95_ms": round(float(ok["latency_ms"].quantile(0.95)), 1),
    })
    return summary


def run_eval(items: list[dict], name: str = "adhoc", out_dir: str = "eval_runs",
             pipeline: RAGPipeline | None = None, use_judge: bool = True,
             progress_cb: Callable[[int, int, dict], None] | None = None) -> dict:
    """Run every item through the pipeline. Returns {"summary", "results" (DataFrame), "csv_path", "json_path"}."""
    s = get_settings()
    rag = pipeline or RAGPipeline()
    judge = (JUDGE_PROMPT | rag.llm.with_structured_output(JudgeScore)) if use_judge else None

    rows = []
    for i, item in enumerate(items, 1):
        q, answerable = item["question"], item.get("answerable", True)
        row = {"id": item.get("id"), "question": q, "answerable": answerable,
               "expected_source": item.get("expected_source"), "expected_chunk_id": item.get("expected_chunk_id"),
               "source_rank": None, "chunk_rank": None, "answer": "", "outcome": "error", "grounded": False,
               "judge_score": None, "keyword_recall": None, "retrieval_attempts": 0, "generation_attempts": 0,
               "latency_ms": None, "error": None}
        try:
            raw = rag.vs.similarity_search(q, k=s.top_k)  # raw retrieval quality, before LLM grading
            row["source_rank"] = rank_of([d.metadata.get("source_name") for d in raw], item.get("expected_source"))
            row["chunk_rank"] = rank_of([chunk_id(d) for d in raw], item.get("expected_chunk_id"))
            res = rag.ask(q, log=False)
            row.update(answer=res["answer"], outcome=res["outcome"], grounded=res["grounded"],
                       retrieval_attempts=res["retrieval_attempts"], generation_attempts=res["generation_attempts"],
                       latency_ms=res["latency_ms"],
                       keyword_recall=keyword_recall(res["answer"], item.get("must_include", [])))
            if judge and answerable and item.get("reference"):
                row["judge_score"] = judge.invoke(
                    {"question": q, "reference": item["reference"], "candidate": res["answer"]}).score
        except Exception as e:  # noqa: BLE001 - one failing question must not abort the run
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if progress_cb:
            progress_cb(i, len(items), row)

    df = pd.DataFrame(rows)
    summary = summarize_results(df)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path, json_path = out / f"{name}-{stamp}.csv", out / f"{name}-{stamp}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"set": name, "summary": summary}, indent=2))
    return {"summary": summary, "results": df, "csv_path": str(csv_path), "json_path": str(json_path)}
