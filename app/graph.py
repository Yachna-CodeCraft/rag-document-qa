"""RAG pipeline as a LangGraph state machine.

    START -> retrieve -> [relevant chunks found?]
                            yes -> generate -> END
                            no  -> fallback -> END
"""
from typing import TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from app import config
from app.llm import get_llm
from app.prompts import PROMPTS, DEFAULT_PROMPT, NOT_FOUND_MESSAGE
from app.vectorstore import get_vectorstore


class RAGState(TypedDict, total=False):
    question: str
    source: Optional[str]            # optional: restrict to one document
    hits: list[tuple[Document, float]]
    answer: str


def retrieve(state: RAGState) -> RAGState:
    kwargs = {"k": config.TOP_K}
    if state.get("source"):
        kwargs["filter"] = {"source": state["source"]}
    results = get_vectorstore().similarity_search_with_relevance_scores(
        state["question"], **kwargs
    )
    hits = [(d, s) for d, s in results if s >= config.MIN_RELEVANCE]
    return {"hits": hits}


def route(state: RAGState) -> str:
    return "generate" if state.get("hits") else "fallback"


def generate(state: RAGState) -> RAGState:
    context = "\n\n".join(
        f"[{d.metadata['source']}, p.{d.metadata['page']}]\n{d.page_content}"
        for d, _ in state["hits"]
    )
    chain = PROMPTS[DEFAULT_PROMPT] | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


def fallback(state: RAGState) -> RAGState:
    return {"answer": NOT_FOUND_MESSAGE, "hits": []}


def _build():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"generate": "generate", "fallback": "fallback"})
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


rag_graph = _build()


def ask(question: str, source: Optional[str] = None) -> dict:
    out = rag_graph.invoke({"question": question, "source": source})
    sources = [
        {
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "score": round(float(s), 3),
            "snippet": d.page_content[:300],
        }
        for d, s in out.get("hits", [])
    ]
    return {"answer": out["answer"], "sources": sources}
