"""
LangGraph RAG pipeline with self-correction.

    rewrite_query -> retrieve -> grade_documents --(relevant)--> generate -> check_grounding --(grounded)--> END
         ^                            |                              ^               |
         |___(nothing relevant)_______|                              |___(ungrounded)|
                                      |                                              |
                                      +--(retries exhausted)--> fallback <-----------+
"""
from __future__ import annotations

import time
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import get_settings
from .prompts import (
    GENERATE_PROMPT,
    GRADE_PROMPT,
    GROUNDING_PROMPT,
    NO_ANSWER,
    PROMPT_VERSION,
    REWRITE_PROMPT,
)


class DocGrade(BaseModel):
    """Relevance verdict for a single retrieved passage."""

    relevant: bool = Field(description="True if the passage helps answer the question")
    reason: str = Field(default="", description="One short sentence explaining the verdict")


class Grounding(BaseModel):
    """Hallucination audit of a generated answer."""

    grounded: bool = Field(description="True only if every claim is supported by the context")
    unsupported_claims: list[str] = Field(default_factory=list)


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict]
    search_query: str
    documents: list[Document]
    n_retrieved: int
    answer: str
    grounded: bool
    feedback: str
    retrieval_attempts: int
    generation_attempts: int
    outcome: str
    trace: list[str]


def format_history(history: list[dict] | None, max_turns: int = 6) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-max_turns:])


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = f", p.{d.metadata['page']}" if d.metadata.get("page") else ""
        parts.append(f"[{i}] (source: {d.metadata.get('source_name', 'unknown')}{page})\n{d.page_content}")
    return "\n\n".join(parts)


class RAGPipeline:
    def __init__(self, llm=None, vectorstore=None, settings=None):
        self.settings = settings or get_settings()
        self._llm = llm
        self._vs = vectorstore
        self._chains = None
        self._graph = None

    # ---- lazy deps (lets tests inject fakes without API keys) ----
    @property
    def llm(self):
        if self._llm is None:
            from .llm import get_chat_model

            self._llm = get_chat_model()
        return self._llm

    @property
    def vs(self):
        if self._vs is None:
            from .vectorstore import get_vectorstore

            self._vs = get_vectorstore()
        return self._vs

    @property
    def chains(self):
        if self._chains is None:
            parser = StrOutputParser()
            self._chains = {
                "rewrite": REWRITE_PROMPT | self.llm | parser,
                "grade": GRADE_PROMPT | self.llm.with_structured_output(DocGrade),
                "generate": GENERATE_PROMPT | self.llm | parser,
                "ground": GROUNDING_PROMPT | self.llm.with_structured_output(Grounding),
            }
        return self._chains

    # ---- nodes ----
    def rewrite_query(self, state: RAGState) -> dict:
        previous = state.get("search_query", "") if state.get("retrieval_attempts") else ""
        query = self.chains["rewrite"].invoke({
            "question": state["question"],
            "history": format_history(state.get("history")),
            "previous_query": previous or "(none)",
        }).strip().strip('"')
        return {"search_query": query or state["question"],
                "trace": state.get("trace", []) + [f"rewrite -> {query!r}"]}

    def retrieve(self, state: RAGState) -> dict:
        results = self.vs.similarity_search_with_score(state["search_query"], k=self.settings.top_k)
        docs = []
        for doc, score in results:
            doc.metadata["score"] = round(float(score), 4)
            docs.append(doc)
        return {"documents": docs, "n_retrieved": len(docs),
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
                "trace": state.get("trace", []) + [f"retrieve -> {len(docs)} chunks"]}

    def grade_documents(self, state: RAGState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {"documents": [], "trace": state.get("trace", []) + ["grade -> nothing to grade"]}
        grades = self.chains["grade"].batch(
            [{"question": state["question"], "passage": d.page_content} for d in docs]
        )
        kept = [d for d, g in zip(docs, grades) if g.relevant]
        return {"documents": kept,
                "trace": state.get("trace", []) + [f"grade -> kept {len(kept)}/{len(docs)}"]}

    def generate(self, state: RAGState) -> dict:
        feedback = state.get("feedback", "")
        answer = self.chains["generate"].invoke({
            "context": format_context(state["documents"]),
            "question": state["question"],
            "history": format_history(state.get("history")),
            "no_answer": NO_ANSWER,
            "feedback": f"IMPORTANT correction from a previous attempt: {feedback}" if feedback else "",
        }).strip()
        return {"answer": answer, "generation_attempts": state.get("generation_attempts", 0) + 1,
                "trace": state.get("trace", []) + ["generate"]}

    def check_grounding(self, state: RAGState) -> dict:
        verdict = self.chains["ground"].invoke({
            "context": format_context(state["documents"]),
            "question": state["question"],
            "answer": state["answer"],
        })
        feedback = ""
        if not verdict.grounded:
            feedback = "Remove or fix these unsupported claims: " + "; ".join(verdict.unsupported_claims)
        return {"grounded": verdict.grounded, "feedback": feedback,
                "trace": state.get("trace", []) + [f"grounding -> {'pass' if verdict.grounded else 'FAIL'}"]}

    def fallback(self, state: RAGState) -> dict:
        outcome = "ungrounded" if state.get("answer") else "no_context"
        return {"answer": NO_ANSWER, "grounded": True, "outcome": outcome,
                "trace": state.get("trace", []) + [f"fallback ({outcome})"]}

    def finalize(self, state: RAGState) -> dict:
        return {"outcome": "answered" if state["answer"].strip() != NO_ANSWER else "no_context"}

    # ---- routing ----
    def route_after_grade(self, state: RAGState) -> str:
        if state.get("documents"):
            return "generate"
        if state.get("retrieval_attempts", 0) < self.settings.max_retrieval_attempts:
            return "rewrite_query"
        return "fallback"

    def route_after_grounding(self, state: RAGState) -> str:
        if state.get("grounded"):
            return "finalize"
        if state.get("generation_attempts", 0) < self.settings.max_generation_attempts:
            return "generate"
        return "fallback"

    # ---- graph ----
    @property
    def graph(self):
        if self._graph is None:
            g = StateGraph(RAGState)
            for name in ["rewrite_query", "retrieve", "grade_documents", "generate",
                         "check_grounding", "fallback", "finalize"]:
                g.add_node(name, getattr(self, name))
            g.add_edge(START, "rewrite_query")
            g.add_edge("rewrite_query", "retrieve")
            g.add_edge("retrieve", "grade_documents")
            g.add_conditional_edges("grade_documents", self.route_after_grade,
                                    {"generate": "generate", "rewrite_query": "rewrite_query",
                                     "fallback": "fallback"})
            g.add_edge("generate", "check_grounding")
            g.add_conditional_edges("check_grounding", self.route_after_grounding,
                                    {"finalize": "finalize", "generate": "generate", "fallback": "fallback"})
            g.add_edge("fallback", END)
            g.add_edge("finalize", END)
            self._graph = g.compile()
        return self._graph

    # ---- public API ----
    def ask(self, question: str, history: list[dict] | None = None, log: bool = True) -> dict:
        t0 = time.perf_counter()
        state = self.graph.invoke({"question": question, "history": history or [], "trace": []})
        latency_ms = (time.perf_counter() - t0) * 1000

        docs = state.get("documents", []) if state["outcome"] == "answered" else []
        sources = [{"n": i, "source": d.metadata.get("source_name"), "page": d.metadata.get("page"),
                    "score": d.metadata.get("score"), "snippet": d.page_content[:240]}
                   for i, d in enumerate(docs, 1)]
        result = {
            "question": question,
            "answer": state["answer"],
            "outcome": state["outcome"],
            "grounded": state.get("grounded", False),
            "sources": sources,
            "search_query": state.get("search_query"),
            "n_retrieved": state.get("n_retrieved", 0),
            "n_relevant": len(state.get("documents", [])),
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "generation_attempts": state.get("generation_attempts", 0),
            "latency_ms": round(latency_ms, 1),
            "trace": state.get("trace", []),
        }
        if log:
            from .metrics import log_interaction

            result["interaction_id"] = log_interaction({
                **result, "prompt_version": PROMPT_VERSION,
                "provider": self.settings.provider, "model": self.settings.chat_model,
                "grounded": int(result["grounded"]),
            })
        return result
