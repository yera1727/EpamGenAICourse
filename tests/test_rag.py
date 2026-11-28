"""Offline unit tests for the RAG pipeline (embeddings, vector DB and LLM are fakes)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from orbitra_rag.llm import ChatResponse
from orbitra_rag.rag import SYSTEM_PROMPT, RAGPipeline, build_user_prompt
from orbitra_rag.vector_store import SearchHit


def hit(i: int, text: str) -> SearchHit:
    return SearchHit(id=str(i), score=0.9 - i / 10, payload={"doc_id": f"doc-{i}", "section": f"S{i}", "text": text})


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, vector, top_k=5, product_id=None):
        self.calls.append({"vector": vector, "top_k": top_k, "product_id": product_id})
        return [hit(1, "E06 means LiDAR blocked."), hit(2, "E04 means cliff sensor dirty.")][:top_k]


class FakeLLM:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return ChatResponse(text="It is the LiDAR [1].", model="fake", latency_s=0.5, usage={"total_tokens": 10})

    def stream(self, messages):
        self.messages = messages
        yield from ["It is ", "the LiDAR [1]."]

    def ask(self, prompt, system=None):
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return ChatResponse(text="Probably the fan.", model="fake", latency_s=0.1)


def make_pipeline():
    embedder, store, llm = FakeEmbedder(), FakeStore(), FakeLLM()
    return RAGPipeline(embedder=embedder, store=store, llm=llm, top_k=5), embedder, store, llm


class RAGPipelineTest(unittest.TestCase):
    def test_answer_runs_all_steps(self):
        rag, embedder, store, llm = make_pipeline()
        result = rag.answer("What is E06?", product_id="robovac-r7", top_k=2)

        self.assertEqual(embedder.queries, ["What is E06?"])  # question -> embedding
        self.assertEqual(store.calls[0]["vector"], [0.1, 0.2])  # embedding -> vector search
        self.assertEqual(store.calls[0]["product_id"], "robovac-r7")
        self.assertEqual(store.calls[0]["top_k"], 2)
        self.assertEqual(result.answer, "It is the LiDAR [1].")
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(set(result.timings), {"embedding_s", "search_s", "llm_s"})

    def test_prompt_contains_question_and_context_in_one_window(self):
        rag, _, _, llm = make_pipeline()
        rag.answer("What is E06?")
        system, user = llm.messages
        self.assertEqual(system["content"], SYSTEM_PROMPT)
        self.assertIn("What is E06?", user["content"])
        self.assertIn("[1] source: doc-1 › S1", user["content"])
        self.assertIn("E06 means LiDAR blocked.", user["content"])
        self.assertIn("[2] source: doc-2 › S2", user["content"])

    def test_stream_answer(self):
        rag, *_ = make_pipeline()
        hits, timings, stream = rag.stream_answer("What is E06?")
        self.assertEqual(len(hits), 2)
        self.assertEqual("".join(stream), "It is the LiDAR [1].")

    def test_no_hits_is_explicit_in_prompt(self):
        self.assertIn("no relevant documents", build_user_prompt("q", []))

    def test_baseline_has_no_context(self):
        rag, _, store, llm = make_pipeline()
        rag.answer_without_rag("What is E06?")
        self.assertEqual(store.calls, [])
        self.assertNotIn("Context documents", llm.messages[1]["content"])

    def test_empty_question_rejected(self):
        rag, *_ = make_pipeline()
        with self.assertRaises(ValueError):
            rag.answer("  ")


if __name__ == "__main__":
    unittest.main()
