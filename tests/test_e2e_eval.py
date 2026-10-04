"""Whole flow with a real local Qdrant + PDF upload + generated ground truth + eval run (fake LLM)."""
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from documind import evalsets, kb
from documind.config import get_settings
from documind.evalgen import generate_eval_set
from documind.evaluate import run_eval
from documind.graph import RAGPipeline
from documind.vectorstore import get_vectorstore

from conftest import make_pdf


class FakeLLM(FakeListChatModel):
    structured: dict = {}

    def with_structured_output(self, schema, **kwargs):
        p = self.structured[schema.__name__]
        return RunnableLambda(lambda _x: schema(**p))


def test_pdf_to_ground_truth_to_eval(env):
    vs = get_vectorstore(DeterministicFakeEmbedding(size=64))
    text = ["Annual subscription plans can be refunded in full within 30 days of purchase by every customer.",
            "Monthly plans are non refundable but may be cancelled before the next billing date at any time.",
            "Priority support responds within four business hours while standard support takes two days."] * 3
    kb.add_files([("policy.pdf", make_pdf(text))], vectorstore=vs)
    assert kb.list_sources()[0]["source"] == "policy.pdf"

    llm = FakeLLM(responses=["refund window", "30 days [1]"], structured={
        "GeneratedQA": {"question": "How long is the refund period for yearly plans?", "reference": "30 days", "must_include": ["30 days"]},
        "UnanswerableQuestions": {"questions": ["Who is the CEO?"]},
        "DocGrade": {"relevant": True}, "Grounding": {"grounded": True}, "JudgeScore": {"score": 5}})
    rag = RAGPipeline(llm=llm, vectorstore=vs, settings=get_settings())

    # The unanswerable check uses the grader (relevant=True here), so use a grader that rejects for that step
    gen_llm = FakeLLM(responses=["x"], structured={**llm.structured, "DocGrade": {"relevant": False}})
    items = generate_eval_set(RAGPipeline(llm=gen_llm, vectorstore=vs, settings=get_settings()), kb.get_chunks(), 1, 1)
    assert [i["answerable"] for i in items] == [True, False]
    evalsets.save_set("pdf-eval", items)

    out = run_eval(evalsets.load_set("pdf-eval"), name="pdf-eval", out_dir=str(env / "runs"), pipeline=rag)
    df, sm = out["results"], out["summary"]
    assert sm["n_errors"] == 0 and sm["n_questions"] == 2
    # the question was generated from a chunk that is in the index, so exact chunk retrieval is scorable
    assert df.iloc[0]["chunk_rank"] is not None and sm["chunk_hit_rate"] in (0.0, 1.0)
    assert sm["avg_judge_score"] == 5.0
