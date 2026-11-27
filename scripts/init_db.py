"""Step 3: verify the local Qdrant instance and create the knowledge-base collection.

Usage:
    python scripts/init_db.py              # create collection if missing + smoke test
    python scripts/init_db.py --recreate   # drop and recreate the collection

The smoke test runs vector search on a temporary collection with hand-made 4-dim vectors,
so it proves the DB works without calling any embedding API.
"""

from __future__ import annotations

import argparse
import sys

from orbitra_rag.config import get_settings
from orbitra_rag.vector_store import VectorStore


def smoke_test(settings) -> None:
    store = VectorStore(settings, collection="orbitra_smoke_test")
    store.dim = 4
    store.ensure_collection(recreate=True)
    try:
        store.upsert(
            chunk_ids=["vacuum", "coffee", "thermostat"],
            vectors=[[0.9, 0.1, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [0.0, 0.0, 0.2, 0.9]],
            payloads=[
                {"text": "robot vacuum", "product_id": "robovac-r7", "doc_type": "manual"},
                {"text": "coffee machine", "product_id": "brewmate-c3", "doc_type": "manual"},
                {"text": "thermostat", "product_id": "thermosmart-t2", "doc_type": "faq"},
            ],
        )
        hits = store.search([0.8, 0.2, 0.0, 0.0], top_k=3)
        print("  query ~ 'vacuum' ->", [(h.payload["text"], round(h.score, 3)) for h in hits])
        assert hits[0].payload["text"] == "robot vacuum", "nearest neighbour should be 'robot vacuum'"

        filtered = store.search([0.8, 0.2, 0.0, 0.0], top_k=3, doc_type="faq")
        print("  same query, filter doc_type=faq ->", [h.payload["text"] for h in filtered])
        assert [h.payload["text"] for h in filtered] == ["thermostat"]
    finally:
        store.drop_collection()
    print("  smoke test passed: vector search + payload filtering work")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args()

    settings = get_settings()
    store = VectorStore(settings)

    try:
        server = store.server_info()
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot reach Qdrant at {settings.qdrant_url}: {exc}")
        print("Start it with:  docker compose up -d")
        return 1
    print(f"Connected to Qdrant {server['version']} at {settings.qdrant_url}")

    print("Running smoke test...")
    smoke_test(settings)

    store.ensure_collection(recreate=args.recreate)
    print("Knowledge-base collection ready:")
    for key, value in store.info().items():
        print(f"  {key}: {value}")
    print(f"Dashboard: {settings.qdrant_url}/dashboard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
