"""Qdrant vector store: collection management, upsert and vector search.

Talks to Qdrant over its REST API (https://api.qdrant.tech/api-reference) using httpx.
We intentionally do not use `qdrant-client`: it always imports `grpcio`, whose native DLL is
blocked by Windows Application Control on the development machine, and REST is all we need.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings, get_settings

# Payload fields that get a keyword index, so filtered search stays fast.
INDEXED_FIELDS = ("product_id", "doc_type", "doc_id")

# Namespace for deterministic point IDs: the same chunk_id always maps to the same UUID,
# which makes re-ingestion idempotent (points are overwritten, not duplicated).
_ID_NAMESPACE = uuid.UUID("6f1c2b8e-4d1a-4c5e-9a43-0b7f0e0c9a11")

UPSERT_BATCH = 128


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class VectorStoreError(RuntimeError):
    pass


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, settings: Settings | None = None, collection: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection = collection or self.settings.qdrant_collection
        self.dim = self.settings.embedding_dim
        self._http = httpx.Client(base_url=self.settings.qdrant_url, timeout=30)

    # ------------------------------------------------------------------ http helper

    def _request(self, method: str, path: str, json: dict | None = None, ok_404: bool = False) -> Any:
        response = self._http.request(method, path, json=json)
        if ok_404 and response.status_code == 404:
            return None
        if response.is_error:
            raise VectorStoreError(f"Qdrant {method} {path} -> {response.status_code}: {response.text}")
        return response.json().get("result")

    @property
    def _base(self) -> str:
        return f"/collections/{self.collection}"

    # ------------------------------------------------------------------ server / collection

    def server_info(self) -> dict[str, Any]:
        response = self._http.get("/")
        response.raise_for_status()
        return response.json()

    def exists(self) -> bool:
        return bool(self._request("GET", f"{self._base}/exists")["exists"])

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection (cosine distance, HNSW index) and payload indexes if needed."""
        exists = self.exists()
        if exists and recreate:
            self.drop_collection()
            exists = False

        if exists:
            current_size = self._request("GET", self._base)["config"]["params"]["vectors"]["size"]
            if current_size != self.dim:
                raise VectorStoreError(
                    f"Collection '{self.collection}' has vector size {current_size}, "
                    f"but EMBEDDING_DIM={self.dim}. Recreate it with --recreate."
                )
            return

        self._request(
            "PUT",
            self._base,
            {
                "vectors": {"size": self.dim, "distance": "Cosine"},
                "hnsw_config": {"m": 16, "ef_construct": 100},
            },
        )
        for field in INDEXED_FIELDS:
            self._request(
                "PUT", f"{self._base}/index?wait=true", {"field_name": field, "field_schema": "keyword"}
            )

    def drop_collection(self) -> None:
        self._request("DELETE", self._base, ok_404=True)

    def count(self) -> int:
        return self._request("POST", f"{self._base}/points/count", {"exact": True})["count"]

    def info(self) -> dict[str, Any]:
        info = self._request("GET", self._base)
        vectors = info["config"]["params"]["vectors"]
        return {
            "collection": self.collection,
            "status": info["status"],
            "points": info.get("points_count"),
            "vector_size": vectors["size"],
            "distance": vectors["distance"],
            "payload_indexes": sorted(info.get("payload_schema", {}).keys()),
        }

    # ------------------------------------------------------------------ data

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        if not (len(chunk_ids) == len(vectors) == len(payloads)):
            raise ValueError("chunk_ids, vectors and payloads must have the same length")
        points = [
            {"id": point_id(cid), "vector": vec, "payload": {"chunk_id": cid, **payload}}
            for cid, vec, payload in zip(chunk_ids, vectors, payloads)
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            self._request("PUT", f"{self._base}/points?wait=true", {"points": points[start : start + UPSERT_BATCH]})

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        product_id: str | None = None,
        doc_type: str | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Vector similarity search with optional metadata filters.

        `product_id` also matches cross-product documents (product_id == "all").
        """
        must: list[dict[str, Any]] = []
        if product_id:
            must.append({"key": "product_id", "match": {"any": [product_id, "all"]}})
        if doc_type:
            must.append({"key": "doc_type", "match": {"value": doc_type}})

        body: dict[str, Any] = {"query": vector, "limit": top_k, "with_payload": True}
        if must:
            body["filter"] = {"must": must}
        if score_threshold is not None:
            body["score_threshold"] = score_threshold

        result = self._request("POST", f"{self._base}/points/query", body)
        return [
            SearchHit(id=str(p["id"]), score=p["score"], payload=p.get("payload") or {})
            for p in result["points"]
        ]
