"""End-to-end graph tests using fake LLM + fake vector store (no API keys, no network)."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from documind.config import get_settings
from documind.graph import RAGPipeline
from documind.prompts import NO_ANSWER


class FakeLLM(FakeListChatModel):
    structured: dict = {}

    def with_structured_output(self, schema, **kwargs):
        payload = self.structured[schema.__name__]
        return RunnableLambda(lambda _x: schema(**payload))


class FakeVS:
    def __init__(self, docs):
        self.docs = docs
        self.calls = 0

    def similarity_search_with_score(self, query, k=5):
        self.calls += 1
        return [(d, 0.9) for d in self.docs[:k]]


DOCS = [Document(page_content="Annual plans are refundable within 30 days.",
                 metadata={"source_name": "policy.md"})]


def _pipe(responses, structured, docs=DOCS):
    return RAGPipeline(llm=FakeLLM(responses=responses, structured=structured),
                       vectorstore=FakeVS(docs), settings=get_settings())


def test_happy_path():
    p = _pipe(["refund window annual", "30 days [1]"],
              {"DocGrade": {"relevant": True}, "Grounding": {"grounded": True}})
    r = p.ask("What's the refund window?", log=False)
    assert r["outcome"] == "answered" and r["answer"] == "30 days [1]"
    assert r["sources"][0]["source"] == "policy.md"
    assert r["retrieval_attempts"] == 1 and r["generation_attempts"] == 1


def test_irrelevant_docs_retries_then_falls_back():
    p = _pipe(["query"], {"DocGrade": {"relevant": False}, "Grounding": {"grounded": True}})
    r = p.ask("Who is the CEO?", log=False)
    assert r["outcome"] == "no_context" and r["answer"] == NO_ANSWER
    assert r["retrieval_attempts"] == get_settings().max_retrieval_attempts
    assert r["sources"] == []


def test_hallucination_triggers_regeneration_then_fallback():
    p = _pipe(["query", "made up answer"],
              {"DocGrade": {"relevant": True},
               "Grounding": {"grounded": False, "unsupported_claims": ["invented fact"]}})
    r = p.ask("q", log=False)
    assert r["outcome"] == "ungrounded" and r["answer"] == NO_ANSWER
    assert r["generation_attempts"] == get_settings().max_generation_attempts
