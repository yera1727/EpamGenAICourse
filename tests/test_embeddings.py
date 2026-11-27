"""Offline unit tests for EmbeddingsClient (the embeddings API is replaced by a fake).

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import httpx
import openai

from orbitra_rag.config import get_settings
from orbitra_rag.embeddings import EmbeddingsClient, cosine_similarity

DIM = 8


class FakeEmbeddingsAPI:
    """Mimics `client.embeddings.create`, returning items in reverse order to test sorting."""

    def __init__(self, dim: int = DIM, fail_times: int = 0) -> None:
        self.dim = dim
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            request = httpx.Request("POST", "http://fake")
            raise openai.RateLimitError("throttled", response=httpx.Response(429, request=request), body=None)
        items = [
            SimpleNamespace(index=i, embedding=[float(len(text))] + [float(i)] * (self.dim - 1))
            for i, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(items)))


def make_client(api: FakeEmbeddingsAPI, batch_size: int = 2, send_dimensions: bool = False) -> EmbeddingsClient:
    settings = dataclasses.replace(
        get_settings(), embedding_dim=DIM, embedding_batch_size=batch_size, embedding_send_dimensions=send_dimensions
    )
    return EmbeddingsClient(settings=settings, client=SimpleNamespace(embeddings=api))


class EmbeddingsClientTest(unittest.TestCase):
    def test_batches_and_preserves_order(self):
        api = FakeEmbeddingsAPI()
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        vectors = make_client(api, batch_size=2).embed_documents(texts)

        self.assertEqual(len(api.calls), 3)  # 2 + 2 + 1
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(all(len(v) == DIM for v in vectors))

    def test_missing_index_keeps_response_order(self):
        class NoIndexAPI(FakeEmbeddingsAPI):  # Gemini-style response: index is None, input order
            def create(self, **kwargs):
                return SimpleNamespace(
                    data=[SimpleNamespace(index=None, embedding=[float(len(t))] * DIM) for t in kwargs["input"]]
                )

        vectors = make_client(NoIndexAPI(), batch_size=10).embed_documents(["a", "bb", "ccc"])
        self.assertEqual([v[0] for v in vectors], [1.0, 2.0, 3.0])

    def test_embed_query_normalises_whitespace(self):
        api = FakeEmbeddingsAPI()
        make_client(api).embed_query("  error \n  E06  ")
        self.assertEqual(api.calls[0]["input"], ["error E06"])

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            make_client(FakeEmbeddingsAPI()).embed_query("   ")

    def test_dimension_mismatch_detected(self):
        with self.assertRaises(RuntimeError):
            make_client(FakeEmbeddingsAPI(dim=DIM + 1)).embed_query("hello")

    def test_dimensions_parameter_is_optional(self):
        api = FakeEmbeddingsAPI()
        make_client(api).embed_query("x")
        make_client(api, send_dimensions=True).embed_query("x")
        self.assertNotIn("dimensions", api.calls[0])
        self.assertEqual(api.calls[1]["dimensions"], DIM)

    def test_retries_on_rate_limit(self):
        api = FakeEmbeddingsAPI(fail_times=1)
        client = make_client(api)
        client._embed_batch.retry.wait = lambda *_: 0  # no real sleeping in tests
        self.assertEqual(len(client.embed_query("x")), DIM)
        self.assertEqual(len(api.calls), 2)

    def test_cosine_similarity(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([0, 0], [1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
