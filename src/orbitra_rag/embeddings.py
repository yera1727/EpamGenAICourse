"""Embeddings client: converts text chunks and user queries into dense vectors.

The same client (and therefore the same model) is used both for indexing the dataset and for
embedding user questions - otherwise the vectors would not be comparable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings, get_settings
from .providers import create_openai_client

logger = logging.getLogger(__name__)

# Errors worth retrying: throttling, timeouts, network problems and 5xx responses.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class EmbeddingsClient:
    """Thin wrapper around the embeddings API with batching, retries and validation."""

    def __init__(self, settings: Settings | None = None, client: openai.OpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.embedding_model
        self.dim = self.settings.embedding_dim
        self.batch_size = self.settings.embedding_batch_size
        self._client = client or create_openai_client(self.settings)

    # ------------------------------------------------------------------ public API

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many texts, preserving order. Splits the input into API-sized batches."""
        cleaned = [self._prepare(t) for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            logger.debug("Embedding batch %d-%d of %d", start, start + len(batch), len(cleaned))
            vectors.extend(self._embed_batch(batch))
        return vectors

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _prepare(text: str) -> str:
        text = " ".join(text.split())  # collapse whitespace / newlines
        if not text:
            raise ValueError("Cannot embed an empty string")
        return text

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=2, min=2, max=60),  # free tiers have per-minute quotas
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": batch}
        if self.settings.embedding_send_dimensions:
            kwargs["dimensions"] = self.dim
        response = self._client.embeddings.create(**kwargs)

        # OpenAI may return items out of order - sort by index to be safe.
        # Some OpenAI-compatible APIs (e.g. Gemini) leave `index` empty and return items in input order.
        data = list(response.data)
        if all(item.index is not None for item in data):
            data.sort(key=lambda item: item.index)
        vectors = [item.embedding for item in data]

        if len(vectors) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} embeddings, got {len(vectors)}")
        for vec in vectors:
            if len(vec) != self.dim:
                raise RuntimeError(
                    f"Model '{self.model}' returned vectors of size {len(vec)}, but EMBEDDING_DIM={self.dim}. "
                    "Fix EMBEDDING_DIM in .env (or enable EMBEDDING_SEND_DIMENSIONS)."
                )
        return vectors


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
