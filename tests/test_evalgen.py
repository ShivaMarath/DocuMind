from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from documind.config import get_settings
from documind.evalgen import generate_eval_set, sample_chunks, usable_chunk
from documind.graph import RAGPipeline

TEXT = "The annual plan refund window is 30 days from the purchase date for all customers worldwide. " * 4


class FakeLLM(FakeListChatModel):
    structured: dict = {}

    def with_structured_output(self, schema, **kwargs):
        payload = self.structured[schema.__name__]
        # payload may be a list -> cycle through it so each call yields a different question
        state = {"i": 0}

        def _f(_x):
            p = payload[state["i"] % len(payload)] if isinstance(payload, list) else payload
            state["i"] += 1
            return schema(**p)
        return RunnableLambda(_f)


class FakeVS:
    def similarity_search(self, q, k=5):
        return [Document(page_content=TEXT, metadata={"source_name": "x"})]


def _chunks():
    return [Document(page_content=TEXT + str(i), metadata={"source_name": f"doc{i % 2}.pdf", "page": i, "chunk_id": f"id{i}"})
            for i in range(8)] + [Document(page_content="12 345 678", metadata={"source_name": "doc0.pdf", "chunk_id": "num"})]


def _rag(structured):
    structured = {"Grounding": {"grounded": True}, "DocGrade": {"relevant": False}, **structured}
    return RAGPipeline(llm=FakeLLM(responses=["x"], structured=structured), vectorstore=FakeVS(), settings=get_settings())


def test_chunk_filtering_and_round_robin_sampling():
    assert not usable_chunk(Document(page_content="short", metadata={}))
    assert not usable_chunk(Document(page_content="1 2 3 4 5 6 7 8 9 " * 30, metadata={}))
    picked = sample_chunks(_chunks(), 4, seed=1)
    assert len(picked) == 4 and {c.metadata["source_name"] for c in picked} == {"doc0.pdf", "doc1.pdf"}


def test_generate_answerable_with_provenance_and_filters():
    qa = [
        {"question": "How long is the refund window for annual plans?", "reference": "30 days.",
         "must_include": ["30 days", "invented-term"]},
        {"question": "What does the passage say about refunds?", "reference": "x"},      # leaks 'the passage' -> dropped
        {"skip": True},                                                                    # boilerplate -> dropped
        {"question": "Which customers get the annual refund?", "reference": "All customers worldwide.", "must_include": ["worldwide"]},
    ]
    rag = _rag({"GeneratedQA": qa, "UnanswerableQuestions": {"questions": []}, "DocGrade": {"relevant": False}})
    items = generate_eval_set(rag, _chunks(), n_answerable=2, n_unanswerable=0, seed=3)
    assert len(items) == 2 and all(i["answerable"] and i["origin"] == "synthetic" for i in items)
    first = items[0]
    assert first["must_include"] == ["30 days"]                       # invented keyword removed
    assert first["expected_chunk_id"].startswith("id") and first["expected_source"].endswith(".pdf")
    assert first["evidence"] and "the passage" not in " ".join(i["question"] for i in items)


def test_unanswerable_questions_are_verified_against_index():
    qs = {"questions": ["Who founded the company?", "What is the CEO's salary?"]}
    # grader says nothing relevant -> kept
    rag = _rag({"GeneratedQA": {"skip": True}, "UnanswerableQuestions": qs, "DocGrade": {"relevant": False}})
    items = generate_eval_set(rag, _chunks(), n_answerable=0, n_unanswerable=2)
    assert [i["answerable"] for i in items] == [False, False] and items[0]["expected_source"] is None
    # grader says the index DOES answer them -> all dropped -> error because nothing was produced
    rag = _rag({"GeneratedQA": {"skip": True}, "UnanswerableQuestions": qs, "DocGrade": {"relevant": True}})
    try:
        generate_eval_set(rag, _chunks(), n_answerable=0, n_unanswerable=2)
        assert False
    except ValueError:
        pass


def test_empty_kb_raises():
    try:
        generate_eval_set(_rag({}), [], 5, 1)
        assert False
    except ValueError as e:
        assert "empty" in str(e)


def test_api_failure_is_not_reported_as_empty_documents():
    class Boom(FakeLLM):
        def with_structured_output(self, schema, **kwargs):
            def _raise(_x):
                raise RuntimeError("401 invalid api key")
            return RunnableLambda(_raise)

    rag = RAGPipeline(llm=Boom(responses=["x"]), vectorstore=FakeVS(), settings=get_settings())
    try:
        generate_eval_set(rag, _chunks(), 2, 0)
        assert False
    except RuntimeError as e:
        assert "invalid api key" in str(e)
