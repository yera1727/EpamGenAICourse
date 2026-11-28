"""RAG pipeline: user question -> embedding -> vector search -> prompt with context -> LLM answer."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .embeddings import EmbeddingsClient
from .llm import ChatResponse, LLMClient, Message
from .vector_store import SearchHit, VectorStore

SYSTEM_PROMPT = """You are Orbitra Assist, the customer support assistant of the smart-home brand Orbitra Home \
(products: RoboVac R7 robot vacuum, BrewMate C3 coffee machine, AirPure A5 air purifier, ThermoSmart T2 thermostat).

Rules:
1. Answer ONLY with information from the numbered context documents provided in the user message. \
Do not use outside knowledge about other brands or "typical" devices.
2. Cite the documents you used with their numbers in square brackets, e.g. [1] or [2][4], right after the facts.
3. If the context does not contain the answer, say clearly that you do not have this information in the \
Orbitra documentation and suggest contacting Orbitra support. Never guess or invent codes, numbers or procedures.
4. If the question is not about Orbitra products, politely say that you can only help with Orbitra Home products.
5. Be concise and practical: use numbered steps for procedures and mention exact values (intervals, limits, codes).
6. Answer in the language of the user's question."""

NO_RAG_SYSTEM_PROMPT = "You are a helpful customer support assistant. Be concise."


@dataclass
class RAGResult:
    question: str
    answer: str
    hits: list[SearchHit]
    timings: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


def format_context(hits: list[SearchHit]) -> str:
    """Render retrieved chunks as numbered documents for the prompt."""
    if not hits:
        return "(no relevant documents were found)"
    blocks = []
    for i, hit in enumerate(hits, start=1):
        p = hit.payload
        blocks.append(f"[{i}] source: {p.get('doc_id')} › {p.get('section')}\n{p.get('text', '').strip()}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    return (
        "Context documents:\n\n"
        f"{format_context(hits)}\n\n"
        "=====\n"
        f"Customer question: {question.strip()}\n\n"
        "Answer using only the context documents above and cite them by number."
    )


class RAGPipeline:
    def __init__(
        self,
        embedder: EmbeddingsClient | None = None,
        store: VectorStore | None = None,
        llm: LLMClient | None = None,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder or EmbeddingsClient()
        self.store = store or VectorStore()
        self.llm = llm or LLMClient()
        self.top_k = top_k

    # ------------------------------------------------------------------ steps

    def retrieve(self, question: str, product_id: str | None = None, top_k: int | None = None) -> tuple[list[SearchHit], dict[str, float]]:
        """Embed the question and find the most similar chunks. Returns (hits, timings)."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        t0 = time.perf_counter()
        vector = self.embedder.embed_query(question)
        t1 = time.perf_counter()
        hits = self.store.search(vector, top_k=top_k or self.top_k, product_id=product_id)
        t2 = time.perf_counter()
        return hits, {"embedding_s": t1 - t0, "search_s": t2 - t1}

    @staticmethod
    def build_messages(question: str, hits: list[SearchHit]) -> list[Message]:
        """System rules + (retrieved context and the original question) in one context window."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, hits)},
        ]

    # ------------------------------------------------------------------ end-to-end

    def answer(self, question: str, product_id: str | None = None, top_k: int | None = None) -> RAGResult:
        hits, timings = self.retrieve(question, product_id, top_k)
        response: ChatResponse = self.llm.chat(self.build_messages(question, hits))
        timings["llm_s"] = response.latency_s
        return RAGResult(question=question, answer=response.text, hits=hits, timings=timings, usage=response.usage)

    def stream_answer(
        self, question: str, product_id: str | None = None, top_k: int | None = None
    ) -> tuple[list[SearchHit], dict[str, float], Iterator[str]]:
        """Retrieve first, then return a generator that streams the LLM answer."""
        hits, timings = self.retrieve(question, product_id, top_k)
        return hits, timings, self.llm.stream(self.build_messages(question, hits))

    def answer_without_rag(self, question: str) -> ChatResponse:
        """Baseline for comparison: the same LLM without any retrieved context."""
        return self.llm.ask(question, system=NO_RAG_SYSTEM_PROMPT)
