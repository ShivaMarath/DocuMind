import pandas as pd

from documind.evaluate import keyword_recall, rank_of, summarize_results
from documind.metrics import load_interactions, log_interaction, record_feedback, summarize


def _rec(outcome="answered", **kw):
    base = dict(question="q", search_query="q", answer="a", outcome=outcome, grounded=1, n_retrieved=5,
                n_relevant=2, retrieval_attempts=1, generation_attempts=1, latency_ms=100.0,
                prompt_version="v1", provider="openai", model="m")
    return {**base, **kw}


def test_log_feedback_and_summary(tmp_path):
    db = str(tmp_path / "m.db")
    a = log_interaction(_rec(), db)
    log_interaction(_rec(outcome="no_context", latency_ms=300.0, retrieval_attempts=2), db)
    record_feedback(a, True, db)
    s = summarize(db_path=db)
    assert s["total_queries"] == 2
    assert s["answered_rate"] == 0.5
    assert s["self_correction_rate"] == 0.5
    assert s["helpful_rate"] == 1.0
    assert len(load_interactions(db)) == 2


def test_eval_helpers():
    assert rank_of(["a.pdf", "b.pdf"], "b.pdf") == 2
    assert rank_of(["a.pdf"], "z.pdf") is None
    assert rank_of(["a.pdf"], None) is None
    assert keyword_recall("Refund within 30 days", ["30 days", "annual"]) == 0.5
    assert keyword_recall("x", []) is None


def test_summarize_results():
    base = dict(grounded=True, judge_score=None, keyword_recall=None, retrieval_attempts=1,
                generation_attempts=1, latency_ms=100, expected_source=None, expected_chunk_id=None,
                source_rank=None, chunk_rank=None)
    df = pd.DataFrame([
        {**base, "answerable": True, "outcome": "answered", "judge_score": 5, "keyword_recall": 1.0,
         "expected_source": "a.pdf", "expected_chunk_id": "c1", "source_rank": 1, "chunk_rank": 2},
        {**base, "answerable": True, "outcome": "answered", "judge_score": 3,
         "expected_source": "a.pdf", "expected_chunk_id": "c2", "source_rank": 1, "chunk_rank": None},
        {**base, "answerable": False, "outcome": "no_context", "retrieval_attempts": 2, "latency_ms": 300},
        {**base, "answerable": True, "outcome": "error", "latency_ms": None},
    ])
    s = summarize_results(df)
    assert s["n_errors"] == 1 and s["n_questions"] == 4
    assert s["source_hit_rate"] == 1.0
    assert s["chunk_hit_rate"] == 0.5 and s["chunk_mrr"] == 0.25
    assert s["avg_judge_score"] == 4.0
    assert s["abstention_accuracy_on_unanswerable"] == 1.0
    assert s["self_correction_rate"] == round(1 / 3, 3)
